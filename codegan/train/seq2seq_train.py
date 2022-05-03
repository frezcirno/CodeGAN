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

from codegan.baseline.seq2seq import Seq2Seq

from .utils import Trainer, add_general_arguments, evaluate_metrics, eval_gen_loss, init_run_dir, is_notebook, save_model, setup_gpu, setup_logging
from ..utils import occupy_mem, set_seed
from ..utils.meter import MaxMeter, BatchAvgMeter, MinMeter
from ..utils.dist import is_distributed, local_rank, rank, world_size

logger = logging.getLogger(__name__)


class Seq2SeqTrainer(Trainer):
    def __init__(self, args, run_dir, device, parallel=True):
        super().__init__(args, run_dir, device)

        self.args = args
        self.model = Seq2Seq(
            args.vocab_size,
            args.embed_size,
            args.max_length,
            args.src_hidden_size,
            args.tgt_hidden_size,
            with_attention=args.with_attention,
        )

        if args.load_path:
            self.model = self.load_model(self.model, args.load_path, device)

        self.model.to(device)

        if parallel:
            self.model = self.build_parallel(self.model, False, device)

        self.prepare_checkpoints()
        self.best_bleu = MaxMeter()

    def prepare_optimizer(self):
        self.opt = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)

    def prepare_checkpoints(self):
        self.register_path('latest', "seqattn.bin")
        self.register_path('best_loss', "seqattn_bestloss_%f.bin")
        self.register_path('best_bleu', "seqattn_bestbleu_%f.bin")
        self.register_path('output_file', f"seqattn_{local_rank()}.output" if is_distributed() else "seqattn.output")
        self.register_path('gold_file', f"seqattn_{local_rank()}.gold" if is_distributed() else "seqattn.gold")

    def save_checkpoint(self, type: Literal['latest|best_loss|best_bleu'], *args):
        path = self.get_path(type, *args)
        save_model(self.model, path)

    def train_epoch(self, dataloader):
        self.model.train()
        avg_meter = BatchAvgMeter()
        with tqdm(dataloader, dynamic_ncols=True) as train_bar:
            for batch in train_bar:
                source_ids = self.to_model_device(batch[0])
                source_mask = self.to_model_device(batch[1])
                target_ids = self.to_model_device(batch[2])
                target_mask = self.to_model_device(batch[3])

                loss, num = self.model(source_ids, source_mask, target_ids, target_mask)
                # TODO: is it right?
                if loss.size():
                    loss = loss.sum()  # mean() to average on multi-gpu.
                    num = num.sum()
                loss.backward()

                # Update parameters
                self.opt.step()
                self.opt.zero_grad()

                loss = avg_meter.update(loss.item(), num.item())
                train_bar.set_description(f"loss {loss:.2f}")

    def train(self, train_dataset, valid_dataset, test_dataset):
        self.prepare_optimizer()

        dataloader, sampler = self.train_dataloader(train_dataset)

        for self.epoch in range(self.train_epochs):
            logger.info("epoch %d", self.epoch)
            if is_distributed():
                sampler.set_epoch(self.epoch)
            self.train_epoch(dataloader)
            self.save_checkpoint('latest')

            self.eval_epoch(valid_dataset, test_dataset)

    def eval(self, train_dataset, valid_dataset, test_dataset):
        loss = self.eval_loss(valid_dataset)
        logger.info(f"+ Eval loss: {loss:.5f}")

        metrics = self.eval_metrics(test_dataset)
        logger.info("+ metrics = %s", metrics)

    def eval_loss(self, dataset) -> float:
        self.model.eval()

        return eval_gen_loss(
            self.model,
            self.device,
            dataset,
            self.eval_batch_size,
            self.num_workers
        )

    def eval_metrics(self, dataset) -> float:
        self.model.eval()

        return evaluate_metrics(
            self.model,
            self.device,
            dataset,
            self.get_path("output_file"),
            self.get_path("gold_file"),
            self.eval_batch_size,
            self.num_workers
        )

    def eval_epoch(self, valid_dataset, test_dataset):
        loss = self.eval_loss(valid_dataset)
        logger.info(f"+ Eval loss: {loss:.5f}")
        best_loss = self.best_loss.update(loss)
        if self.best_loss.is_best():
            self.save_checkpoint('best_loss', best_loss)

        metrics = self.eval_metrics(test_dataset)
        logger.info("+ metrics = %s", metrics)
        dev_bleu = metrics['bleu4']
        best_bleu = self.best_bleu.update(dev_bleu)
        if self.best_bleu.is_best():
            self.save_checkpoint('best_bleu', best_bleu)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--max_length",
            type=int,
            default=32,
            help="The maximum total target sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--vocab_size", type=int, default=50265)
        parser.add_argument("--embed_size", type=int, default=256)
        parser.add_argument("--src_hidden_size", type=int, default=512)
        parser.add_argument("--tgt_hidden_size", type=int, default=512)
        parser.add_argument("--with_attention", action="store_true")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_general_arguments(parser)
    Seq2SeqTrainer.add_arguments(parser)
    Trainer.add_arguments(parser)

    TRAIN_ARGS = '''--help'''.split()

    args = parser.parse_args(TRAIN_ARGS if is_notebook() else sys.argv[1:])

    run_dir = init_run_dir("seq2seq_attn" if args.with_attention else "seq2seq")
    setup_logging(run_dir)

    logger.info(" ".join(sys.argv))
    logger.info(str(args))

    set_seed(args.seed)

    _device, _ = setup_gpu(args.device_ids)
    logger.info(f"Using device {_device}")

    train_dataset, valid_dataset, test_dataset = torch.load(args.data)
    logger.info("train dataset: %d samples", len(train_dataset))
    logger.info("valid dataset: %d samples", len(valid_dataset))
    logger.info("test dataset: %d samples", len(test_dataset))

    trainer = Seq2SeqTrainer(args, run_dir, _device)
    trainer.run(train_dataset, valid_dataset, test_dataset)
