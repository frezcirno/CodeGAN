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

from codegan.train.gen_train import GenTrainer


from .common import load_dataset
from .utils import Trainer, add_general_arguments, eval_dis_acc, eval_dis_loss, fakegen2, init_run_dir, is_notebook, save_model, setup_gpu, setup_logging, validate_device_ids
from ..utils import set_seed
from ..utils.dist import is_distributed, local_rank, rank, world_size
from ..utils.meter import MaxMeter, BatchAvgMeter, MinMeter
from ..utils.memory import occupy_mem
from ..generator.codebert import Generator
from ..discriminator import Discriminator

logger = logging.getLogger(__name__)


class DisTrainer(Trainer):
    def __init__(self, args, run_dir, device, parallel=True):
        super().__init__(args, run_dir)

        self.model = Discriminator(
            args.src_max_len,
            args.tgt_max_len,
            args.vocab_size,
            args.hidden_size,
        )

        if args.load_path:
            self.model = self.load_model(self.model, args.load_path, device)

        self.gen.to(device)

        if parallel:
            self.model = self.build_parallel(
                self.model,
                find_unused_parameters=False,
                load_path=args.load_path,
                device=device
            )

        self.device = device

        self.register_path("latest", os.path.join(run_dir, "dis.bin"))
        self.register_path("best_loss", os.path.join(run_dir, "dis_bestloss_%d_%f.bin"))
        self.register_path("best_acc", os.path.join(run_dir, "dis_bestacc_%d_%f.bin"))

        self.best_loss = MinMeter()
        self.best_acc = MaxMeter()

    @classmethod
    def load_model(cls, raw_model, load_path, map_location):
        logger.info(f"Load model from {load_path}")
        weights = torch.load(load_path,
                             map_location=torch.device(map_location))
        raw_model.load_state_dict(weights)
        return raw_model

    @classmethod
    def add_argument(cls, parser):
        # Dis
        parser.add_argument("--dis_train_sample", type=int, default=5000)
        parser.add_argument("--dis_valid_sample", type=int, default=1000)
        parser.add_argument("--fakegen_batch_size", type=int, default=64)
        parser.add_argument(
            "--dis_load_path", help="The path to the discriminator model",
        )
        parser.add_argument("--dis_early_stop_loss", default=1e-5)

    def save_checkpoint(self, type: Literal['latest|best_acc|best_loss'], *args):
        path = self.get_path(type, *args)
        save_model(self._dis, path)

    def to_dis_device(self, x):
        return x.to(self.device)

    def dis_train(self):
        self.model.train()

    def dis_eval(self):
        self.model.eval()

    def train_epoch(self, dataloader):
        self.dis_train()
        loss_meter = BatchAvgMeter()
        with tqdm(dataloader, "BCEloss 00.0000", dynamic_ncols=True) as bar:
            for batch in bar:
                source_ids = self.to_dis_device(batch[0])
                target_ids = self.to_dis_device(batch[1])
                labels = self.to_dis_device(batch[2])

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
        self.opt = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=self.adam_epsilon
        )

    def train(self, train_dataset, valid_dataset, test_dataset):
        self.prepare_optimizer()

        if is_distributed():
            sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            sampler = RandomSampler(train_dataset)

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
        )

        for self.epoch in range(self.train_epochs):
            logger.info(f"epoch {self.epoch}")

            if is_distributed():
                sampler.set_epoch(self.epoch)
            self.train_epoch(dataloader)
            self.save_checkpoint("latest")

            self.eval_epoch(valid_dataset, test_dataset)

    def eval_epoch(self, valid_dataset, test_dataset):
        loss = self.eval_loss(valid_dataset)
        logger.info(f"+ Eval loss: {loss:.5f}")
        best_loss = self.best_loss.update(loss)
        if self.best_loss.is_best():
            self.save_checkpoint('best_loss', best_loss)

        acc = self.eval_acc(test_dataset)
        logger.info(f"+ Eval accuracy: {acc:.5f}")
        best_acc = self.best_acc.update(acc)
        if self.best_acc.is_best():
            self.save_checkpoint('best_acc', best_acc)

    def eval_loss(self, dataset):
        self.dis_eval()
        avg_loss = eval_dis_loss(self.model, self.device, dataset,
                                 self.batch_size, self.num_workers)
        logger.info("+ BCEloss = %f", avg_loss)
        return avg_loss

    def eval_acc(self, dataset):
        self.dis_eval()
        acc = eval_dis_acc(self.model, self.device, dataset,
                           self.batch_size, self.num_workers)
        logger.info("+ Accu = %f", acc)
        return acc

    def eval(self, train_dataset, valid_dataset, test_dataset):
        loss = self.eval_loss(valid_dataset)
        logger.info("+ BCEloss = %f", loss)
        acc = self.eval_acc(test_dataset)
        logger.info("+ Accu = %f", acc)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        pass


if __name__ == '__main__':
    run_dir = init_run_dir("dis")
    setup_logging(run_dir)

    parser = argparse.ArgumentParser()
    add_general_arguments(parser)
    Trainer.add_arguments(parser)
    DisTrainer.add_arguments(parser)

    GenTrainer.add_arguments(parser)
    parser.add_argument('--gen_load_path', type=str)
    parser.add_argument('--gen_batch_size', type=int, default=64)
    parser.add_argument('--gen_beam_search', action='store_true')
    parser.add_argument('--gen_num_train_batchs', type=int, default=2000)
    parser.add_argument('--gen_num_valid_batchs', type=int, default=200)
    parser.add_argument('--gen_num_test_batchs', type=int, default=200)

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
    logger.info("train dataset: %d samples", len(train_dataset))
    logger.info("valid dataset: %d samples", len(valid_dataset))
    logger.info("test dataset: %d samples", len(test_dataset))

    _gen = GenTrainer(args, run_dir, _device, parallel=False)
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
    _gen = None

    trainer = DisTrainer(args, run_dir, _device)
    trainer.run(train_dataset, valid_dataset, test_dataset)
