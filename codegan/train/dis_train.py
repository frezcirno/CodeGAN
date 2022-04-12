import argparse
import logging
import os
import sys
from typing import Literal
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from .. import tokenize
from .gen_train import GenTrainer

from .utils import Trainer, add_general_arguments, eval_dis_acc, eval_dis_loss, fakegen2, init_run_dir, is_notebook, save_model, setup_gpu, setup_logging, load_dataset
from ..utils import set_seed
from ..utils.cache import cache_result
from ..utils.dist import is_distributed, is_master, local_rank, rank, world_size
from ..utils.meter import MaxMeter, BatchAvgMeter, Meaner, MinMeter
from ..discriminator import Discriminator

logger = logging.getLogger(__name__)


class DisTrainer(Trainer):
    def __init__(self, args, run_dir, load_path, device, parallel=True):
        super().__init__(args, run_dir, device)

        self.model = Discriminator(
            args.src_max_len,
            args.tgt_max_len,
            args.vocab_size,
            args.dis_hidden_size,
        )

        if load_path:
            self.model = self.load_model(self.model, load_path, device)

        self.model.to(device)

        if parallel:
            self.model = self.build_parallel(
                self.model,
                find_unused_parameters=False,
                device=device
            )

        self.register_path("latest", "dis.bin")
        self.register_path("best_loss", "dis_bestloss_%f.bin")
        self.register_path("best_acc", "dis_bestacc_%f.bin")

        self.best_acc = MaxMeter()

    def save_checkpoint(self, type: Literal['latest|best_acc|best_loss'], *args):
        path = self.get_path(type, *args)
        save_model(self.model, path)

    def train_epoch(self, dataloader):
        self.model.train()
        loss_meter = Meaner()
        with tqdm(dataloader, "BCEloss 00.0000", dynamic_ncols=True) as bar:
            for batch in bar:
                source_ids = self.to_model_device(batch[0])
                target_ids = self.to_model_device(batch[1])
                labels = self.to_model_device(batch[2])

                prob = self.model(source_ids, target_ids)
                loss = F.binary_cross_entropy(prob, labels, reduction='sum')
                # mean() to average on multi-gpu
                if loss.size():
                    loss = loss.mean()

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

                avg_loss = loss_meter.update(loss.item(), labels.size(0))
                bar.set_description(f"BCEloss {avg_loss:.4f}")

    def prepare_optimizer(self):
        self.opt = optim.Adam(self.model.parameters(), lr=self.learning_rate,
                              eps=self.adam_epsilon, weight_decay=self.weight_decay)

    def train(self, train_dataset, valid_dataset, test_dataset):
        self.prepare_optimizer()

        dataloader, sampler = self.train_dataloader(train_dataset)

        for self.epoch in range(self.train_epochs):
            logger.info("epoch %d", self.epoch)

            if is_distributed():
                sampler.set_epoch(self.epoch)
            self.train_epoch(dataloader)
            self.save_checkpoint("latest")

            self.eval_epoch(valid_dataset, test_dataset)

    def eval_epoch(self, valid_dataset, test_dataset):
        loss = self.eval_loss(valid_dataset)
        logger.info("+ BCEloss = %f", loss)
        best_loss = self.best_loss.update(loss)
        if self.best_loss.is_best():
            self.save_checkpoint('best_loss', best_loss)

        acc = self.eval_acc(test_dataset)
        logger.info("+ Accu = %f", acc)
        best_acc = self.best_acc.update(acc)
        if self.best_acc.is_best():
            self.save_checkpoint('best_acc', best_acc)

    def eval_loss(self, dataset):
        self.model.eval()
        avg_loss = eval_dis_loss(self.model, self.device, dataset,
                                 self.eval_batch_size, self.num_workers)
        return avg_loss

    def eval_acc(self, dataset):
        self.model.eval()
        return eval_dis_acc(self.model, self.device, dataset,
                            self.eval_batch_size, self.num_workers)

    def eval(self, train_dataset, valid_dataset, test_dataset):
        loss = self.eval_loss(valid_dataset)
        logger.info("+ BCEloss = %f", loss)
        acc = self.eval_acc(test_dataset)
        logger.info("+ Accu = %f", acc)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        GenTrainer.add_arguments(parser)
        parser.add_argument('--gen_load_path', type=str)
        parser.add_argument('--gen_batch_size', type=int, default=256)
        parser.add_argument('--gen_num_train_batchs', type=int, default=2000)
        parser.add_argument('--gen_num_valid_batchs', type=int, default=200)
        parser.add_argument('--gen_num_test_batchs', type=int, default=200)
        parser.add_argument('--gen_beam_search', action='store_true')

        parser.add_argument("--dis_hidden_size", type=int, default=768)


if __name__ == '__main__':
    run_dir = init_run_dir("dis")
    setup_logging(run_dir)

    parser = argparse.ArgumentParser()
    add_general_arguments(parser)
    Trainer.add_arguments(parser)
    DisTrainer.add_arguments(parser)

    TRAIN_ARGS = '''--help'''.split()
    args = parser.parse_args(TRAIN_ARGS if is_notebook() else sys.argv[1:])

    logger.info(" ".join(sys.argv))
    logger.info(str(args))

    set_seed(args.seed)

    if not torch.cuda.is_available():
        logger.error("cuda is not available")
        exit(-1)

    _device, _ = setup_gpu(args.device_ids, args.occupy)
    logger.info(f"Using device {_device}")

    train_dataset, valid_dataset, test_dataset = load_dataset(args.data, args.src_max_len, args.tgt_max_len)

    @cache_result("cache/dataset_for_dis.bin", format="torch")
    def make_dataset_for_dis(train_dataset, valid_dataset, test_dataset):
        _gen = GenTrainer(args, run_dir, args.gen_load_path, _device, parallel=False)
        train_dataset = fakegen2(_gen.model, _device, train_dataset,
                                 args.gen_num_train_batchs,
                                 args.gen_batch_size,
                                 args.gen_beam_search,
                                 args.num_workers)
        valid_dataset = fakegen2(_gen.model, _device, valid_dataset,
                                 args.gen_num_valid_batchs,
                                 args.gen_batch_size,
                                 args.gen_beam_search,
                                 args.num_workers)
        test_dataset = fakegen2(_gen.model, _device, test_dataset,
                                args.gen_num_test_batchs,
                                args.gen_batch_size,
                                args.gen_beam_search,
                                args.num_workers)

        return train_dataset, valid_dataset, test_dataset

    train_dataset, valid_dataset, test_dataset = make_dataset_for_dis(train_dataset, valid_dataset, test_dataset)
    logger.info("train dataset: %d samples", len(train_dataset))
    logger.info("valid dataset: %d samples", len(valid_dataset))
    logger.info("test dataset: %d samples", len(test_dataset))

    trainer = DisTrainer(args, run_dir, args.load_path, _device)
    trainer.run(train_dataset, valid_dataset, test_dataset)
