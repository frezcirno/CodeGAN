import logging
import argparse
from pathlib import Path
import os
import sys
from typing import Literal
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from .common import load_dataset
from .utils import Trainer, add_general_arguments, build_parallel, eval_gen_bleu, eval_gen_loss, init_run_dir, is_notebook, save_model, setup_gpu, setup_logging, validate_device_ids
from ..utils import occupy_mem, set_seed
from ..utils.meter import MaxMeter, BatchAvgMeter, MinMeter
from ..utils.dist import is_distributed, local_rank, rank, world_size
from ..generator.codebert import Generator

logger = logging.getLogger(__name__)


class GenTrainer(Trainer):
    def __init__(self, args, run_dir, device, parallel=True):
        super().__init__(args, run_dir)

        self.args = args
        self.model = Generator(
            args.hidden_size,
            args.vocab_size,
            args.beam_size,
            args.tgt_max_len,
        )

        if args.load_path:
            self.model = self.load_model(self.model, args.load_path, device)

        self.model.to(device)

        if parallel:
            self.model = build_parallel(self.model, False, device)

        self.device = device

        self.prepare_checkpoints()

        self.best_loss = MinMeter()
        self.best_bleu = MaxMeter()

    @classmethod
    def load_model(cls, raw_model, load_path, map_location):
        logger.info(f"Load model from {load_path}")
        weights = torch.load(load_path,
                             map_location=torch.device(map_location))
        if 'encoder.pooler.dense.weight' in weights:
            del weights['encoder.pooler.dense.weight']
            del weights['encoder.pooler.dense.bias']
        raw_model.load_state_dict(weights)
        return raw_model

    def gen_train(self):
        self.model.train()

    def gen_eval(self):
        self.model.eval()

    def to_gen_device(self, x):
        return x.to(self.device)

    def prepare_optimizer(self):
        # TODO: Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.opt = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )

    def prepare_checkpoints(self):
        self.register_path('latest', "gen.bin")
        self.register_path('best_loss', "gen_bestloss_%f.bin")
        self.register_path('best_bleu', "gen_bestbleu_%f.bin")
        self.register_path('output_file', "gen.output")
        self.register_path('gold_file', "gen.gold")

    def save_checkpoint(self, type: Literal['latest|best_loss|best_bleu'], *args):
        path = self.get_path(type, *args)
        save_model(self.model, path)

    def prepare_scheduler(self, train_dataset):
        t_total = len(train_dataset) / self.batch_size * self.train_epochs

        logger.info("+ Total train steps = %d", t_total)
        logger.info("+ Warmup steps = %d", int(t_total * 0.1))

        self.sch = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
        )

    def train_epoch(self, dataloader):
        self.gen_train()
        avg_meter = BatchAvgMeter()
        with tqdm(dataloader, dynamic_ncols=True) as train_bar:
            for batch in train_bar:
                source_ids = self.to_gen_device(batch[0])
                source_mask = self.to_gen_device(batch[1])
                target_ids = self.to_gen_device(batch[2])
                target_mask = self.to_gen_device(batch[3])

                loss, num = self.model(source_ids, source_mask,
                                       target_ids, target_mask)
                # TODO: is it right?
                if loss.size():
                    loss = loss.sum()  # mean() to average on multi-gpu.
                    num = num.sum()
                loss.backward()

                # Update parameters
                self.opt.step()
                self.opt.zero_grad()
                self.sch.step()

                loss = avg_meter.update(loss.item(), num.item())
                train_bar.set_description(f"loss {loss:.2f}")

    def train(self, train_dataset, valid_dataset, test_dataset):
        # Start training
        logger.info("Do train generator:")
        logger.info("+ train_dataset = %d", len(train_dataset))
        logger.info("+ Batch size = %d", self.batch_size)
        logger.info("+ Train epochs = %d", self.train_epochs)
        logger.info("+ Learning rate = %e", self.learning_rate)
        logger.info("+ Adam epsilon = %e", self.adam_epsilon)

        self.prepare_optimizer()
        self.prepare_scheduler(train_dataset)

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
            self.save_checkpoint('latest')

            self.eval_epoch(valid_dataset, test_dataset)

    def eval(self, train_dataset, valid_dataset, test_dataset):
        loss = self.eval_loss(valid_dataset)
        logger.info(f"+ Eval loss: {loss:.5f}")
        dev_bleu = self.eval_bleu(test_dataset)
        logger.info("+ bleu-4 = %f", dev_bleu)

    def eval_epoch(self, valid_dataset, test_dataset):
        loss = self.eval_loss(valid_dataset)
        logger.info(f"+ Eval loss: {loss:.5f}")
        best_loss = self.best_loss.update(loss)
        if self.best_loss.is_best():
            self.save_checkpoint('best_loss', best_loss)

        dev_bleu = self.eval_bleu(test_dataset)
        logger.info("+ bleu-4 = %f", dev_bleu)
        best_bleu = self.best_bleu.update(dev_bleu)
        if self.best_bleu.is_best():
            self.save_checkpoint('best_bleu', best_bleu)

    def eval_loss(self, dataset) -> float:
        self.gen_eval()
        return eval_gen_loss(
            self.model,
            self.device,
            dataset,
            self.batch_size,
            self.num_workers
        )

    def eval_bleu(self, dataset) -> float:
        self.gen_eval()
        return eval_gen_bleu(
            self.model,
            self.device,
            dataset,
            self.get_path("output_file"),
            self.get_path("gold_file"),
            self.eval_batch_size,
            self.num_workers
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--src_max_len",
            type=int,
            default=256,
            help="The maximum total source sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--tgt_max_len",
            type=int,
            default=32,
            help="The maximum total target sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--hidden_size", type=int, default=768, help="The path to the generator model")
        parser.add_argument("--vocab_size", type=int, default=50265, help="The path to the generator model")
        parser.add_argument("--beam_size", type=int, default=10)


if __name__ == '__main__':
    run_dir = init_run_dir("gen")
    setup_logging(run_dir)

    parser = argparse.ArgumentParser()
    add_general_arguments(parser)
    Trainer.add_arguments(parser)
    GenTrainer.add_arguments(parser)

    TRAIN_ARGS = '''--help'''.split()

    args = parser.parse_args(TRAIN_ARGS if is_notebook() else sys.argv[1:])

    logger.info(" ".join(sys.argv))
    logger.info(str(args))

    set_seed(args.seed)

    _device, _ = setup_gpu(args.device_ids)
    logger.info(f"Using device {_device}")

    train_dataset, valid_dataset, test_dataset = load_dataset(args.data, args.src_max_len, args.tgt_max_len)
    logger.info("train dataset: %d samples", len(train_dataset))
    logger.info("valid dataset: %d samples", len(valid_dataset))
    logger.info("test dataset: %d samples", len(test_dataset))

    trainer = GenTrainer(args, run_dir, _device)
    trainer.run(train_dataset, valid_dataset, test_dataset)
