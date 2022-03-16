# %%
import logging
import argparse
from pathlib import Path
import os
from os.path import join as path_join
import sys
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset
from helpers import (
    DisTrainer, GanTrainer, is_distributed, is_notebook, remember_result, set_seed, cache_result,
    to_features, to_pad_features, series_to_tensor, sample_dataset
)
from model import Generator, Discriminator
import tokenizer
# import pytorch_lightning as pl

# %%
parser = argparse.ArgumentParser()

# General
parser.add_argument(
    "--output_dir", default="current", help="output path, to save models, etc.",
)
parser.add_argument(
    "--data_dir", default="data_objs", help="output path, to save data objects, etc.",
)
parser.add_argument(
    "--data", default="data_objs/jd.parquet", help="The path to the data parquet file",
)
parser.add_argument(
    "--source_length",
    type=int,
    default=256,
    help="The maximum total source sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--target_length",
    type=int,
    default=32,
    help="The maximum total target sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument("--do_lower_case", action="store_true", default=True)
parser.add_argument(
    "--seed", type=int, default=42, help="random seed for initialization"
)
parser.add_argument("--gen_use_cpu", action="store_true")
parser.add_argument("--gen_device_ids", nargs="+", type=int, default=[0])
parser.add_argument("--dis_use_cpu", action="store_true")
parser.add_argument("--dis_device_ids", nargs="+", type=int, default=[0])
parser.add_argument("--gen_no_ddp", action="store_true")
parser.add_argument("--dis_no_ddp", action="store_true")

parser.add_argument(
    "--local_rank", type=int, default=-1, help="For distributed training: local_rank",
)
parser.add_argument("--local_ranks", nargs="+", type=int, default=[])

# Gen
parser.add_argument("--do_gen_train", action="store_true")
parser.add_argument("--do_gen_eval", action="store_true")
parser.add_argument("--gen_batch_size", type=int, default=128)
parser.add_argument("--eval_batch_size", type=int, default=64)
parser.add_argument(
    "--gen_load_path", help="The path to the generator model",
)
parser.add_argument("--gen_train_epochs", type=int, default=10)
parser.add_argument("--gen_num_workers", type=int, default=4)
parser.add_argument("--gen_learning_rate", type=float, default=5e-5)
parser.add_argument("--gen_adam_epsilon", type=float, default=1e-8)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--beam_size", type=int, default=10)

# Dis
parser.add_argument("--do_dis_train", action="store_true")
parser.add_argument("--dis_fakegen_train_sample", type=int, default=5000)
parser.add_argument("--dis_fakegen_valid_sample", type=int, default=1000)
parser.add_argument("--fakegen_batch_size", type=int)
parser.add_argument("--fakegen_num_workers", type=int, default=4)
parser.add_argument(
    "--dis_load_path", help="The path to the discriminator model",
)
parser.add_argument("--dis_train_epochs", type=int, default=10)
parser.add_argument("--dis_num_workers", type=int, default=4)
parser.add_argument("--dis_batch_size", type=int, default=64)
parser.add_argument("--dis_learning_rate", type=float, default=5e-5)
parser.add_argument("--dis_adam_epsilon", type=float, default=1e-8)
parser.add_argument("--dis_early_stop_loss", default=1e-5)

# Gan
parser.add_argument("--do_gan_train", action="store_true")
parser.add_argument("--do_gan_eval", action="store_true")
parser.add_argument("--gan_batch_size", type=int)
parser.add_argument("--gan_train_epochs", type=int, default=30)
parser.add_argument("--gan_num_workers", type=int, default=4)
parser.add_argument("--gan_rollnum", type=int, default=20)
parser.add_argument("--gan_g_steps", type=int, default=100,
                    help="Generator train steps, one batch one step.")
parser.add_argument("--gan_teach", action="store_true",
                    help="Use teacher forcing after every step.")
parser.add_argument("--gan_d_steps", type=int, default=5,
                    help="Discriminator train steps, do gan_d_sample x gan_d_epochs samples one step.")
parser.add_argument("--gan_d_sample", type=int, default=1000)
parser.add_argument("--gan_d_epochs", type=int, default=3)
parser.add_argument("--gan_d_batch_size", type=int, default=64)
parser.add_argument("--gan_d_num_workers", type=int, default=4)

args = parser.parse_args(
    args="--gen_load_path checkpoints/3.8_3/gan_gen_bestbleu_0_19.520000.bin --dis_load_path checkpoints/3.8_3/gan_dis_bestbleu_0_19.520000.bin --do_gan_train --gan_teach --gan_train_epochs 10 --gan_batch_size 2 --gan_rollnum 20 --gan_g_step 100 --gan_d_sample 1000 --gan_d_step 10 --gan_d_epochs 10 --fakegen_batch_size 32 --gen_device_ids 3 --dis_device_ids 3 --output_dir 3.16-1".split()
    if is_notebook()
    else sys.argv[1:]
)

args.hidden_size = 768
args.vocab_size = 50265


# For using `torchrun`
try:
    args.local_rank = int(os.environ["LOCAL_RANK"])
except KeyError:
    pass

master = args.local_rank == 0
standalone_or_master = not is_distributed() or master

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)
data_dir = Path(args.data_dir)
data_dir.mkdir(exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            path_join(args.output_dir, datetime.now().strftime(
                f"%Y-%m-%d_%H-%M-%S_{args.local_rank}.log"
                if is_distributed()
                else "%Y-%m-%d_%H-%M-%S.log"
            )),
            "w",
            "utf-8",
        ),
    ],
    level=logging.INFO,
)

logging.info(" ".join(sys.argv))
logging.info(str(args))

if is_distributed():
    if args.local_ranks:
        args.local_rank = args.local_ranks[args.local_rank]

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    # device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    logging.info("Distributed training: %d", is_distributed())

    gen_device = args.local_rank
    dis_device = args.local_rank


else:
    ngpu = torch.cuda.device_count()
    logging.info("GPU count = %d", ngpu)

    args.gen_device_ids = [
        dev_id for dev_id in args.gen_device_ids if dev_id < ngpu
    ]
    if not args.gen_device_ids:
        args.gen_device_ids.append(0)

    args.dis_device_ids = [
        dev_id for dev_id in args.dis_device_ids if dev_id < ngpu
    ]
    if not args.dis_device_ids:
        args.dis_device_ids.append(0)

    gen_device = args.gen_device_ids[0]
    dis_device = args.dis_device_ids[0]


if args.gen_use_cpu:
    gen_device = "cpu"
    args.gen_device_ids = []


if args.dis_use_cpu:
    dis_device = "cpu"
    args.dis_device_ids = []


# %%
set_seed(args.seed)


@remember_result
def make_jd():
    return pd.read_parquet(args.data)


@remember_result
@cache_result(data_dir / "feats")
def make_feats():
    jd = make_jd()
    return to_features(jd)


@remember_result
@cache_result(data_dir / "pad_feats")
def make_pad_feats():
    feats = make_feats()
    return to_pad_features(feats,
                           args.source_length,
                           args.target_length)


@remember_result
@cache_result(data_dir / "train_feats")
def make_train_feats():
    jd = make_jd()
    pad_feats = make_pad_feats()
    return pad_feats[jd.partition == "train"]


@remember_result
@cache_result(data_dir / "valid_feats")
def make_valid_feats():
    jd = make_jd()
    pad_feats = make_pad_feats()
    return pad_feats[jd.partition == "valid"]


@remember_result
@cache_result(data_dir / "jd_bleu")
def make_jd_bleu():
    jd = make_jd()
    return sample_dataset(jd[jd.partition == "valid"])


@remember_result
@cache_result(data_dir / "bleu_feats")
def make_bleu_feats():
    pad_feats = make_pad_feats()
    jd_bleu = make_jd_bleu()
    return pad_feats.loc[jd_bleu.index]


@remember_result
@cache_result(data_dir / "train_dataset", "pickle")
def make_train_dataset():
    feats = make_train_feats()
    return TensorDataset(
        series_to_tensor(feats.code_ids),
        series_to_tensor(feats.code_mask),
        series_to_tensor(feats.doc_ids),
        series_to_tensor(feats.doc_mask),
    )


@remember_result
@cache_result(data_dir / "valid_dataset", "pickle")
def make_valid_dataset():
    feats = make_valid_feats()
    return TensorDataset(
        series_to_tensor(feats.code_ids),
        series_to_tensor(feats.code_mask),
        series_to_tensor(feats.doc_ids),
        series_to_tensor(feats.doc_mask),
    )


@remember_result
@cache_result(data_dir / "bleu_dataset", "pickle")
def make_bleu_dataset():
    feats = make_bleu_feats()
    return TensorDataset(
        series_to_tensor(feats.code_ids),
        series_to_tensor(feats.code_mask),
    )


jd_bleu = make_jd_bleu()
bleu_feats = make_bleu_feats()
train_dataset = make_train_dataset()
valid_dataset = make_valid_dataset()
bleu_dataset = make_bleu_dataset()

logging.info("train dataset: %d samples", len(train_dataset))
# logging.info("test dataset: %d samples", len(test_dataset))
logging.info("valid dataset: %d samples", len(valid_dataset))
logging.info("bleu dataset: %d samples", len(bleu_dataset))

# %%
# [Load model]
# config = RobertaConfig.from_pretrained("microsoft/codebert-base")


def build_gen(args, load_path="", device=0, device_ids=[]):
    _model = Generator(
        args.hidden_size,
        args.vocab_size,
        args.beam_size,
        args.target_length,
        device,
    )
    if load_path:
        logging.info("Load generator from: %s", load_path)
        _model.load_state_dict(
            torch.load(load_path,
                       map_location=torch.device(device))
        )

    logging.info("Load generator on device %d", device)
    _model.to(device)

    if is_distributed():
        logging.info("Build DDP generator")
        model = DDP(
            _model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
    elif len(device_ids) > 1:
        logging.info("Build DP generator")
        model = DP(_model, device_ids)
        model.device = device
    else:
        logging.info("Build generator")
        model = _model

    return _model, model


def build_dis(args, load_path="", device=0, device_ids=[]):
    _model = Discriminator(
        args.source_length,
        args.target_length,
        args.vocab_size,
        args.hidden_size,
        device,
    )
    if load_path:
        logging.info("Load discriminator from: %s", load_path)
        _model.load_state_dict(
            torch.load(load_path,
                       map_location=torch.device(device))
        )
    _model.to(device)

    if is_distributed():
        logging.info("Build DDP discriminator")
        model = DDP(
            _model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
    elif len(device_ids) > 1:
        logging.info("Build DP discriminator")
        model = DP(_model, device_ids)
        model.device = device
    else:
        logging.info("Build discriminator")
        model = _model

    return _model, model


"""
Generator: 1995 MiB
Discriminator: 556 MiB
Total: 2551 MiB
"""
if args.do_gen_train or args.do_gen_eval or args.do_dis_train or args.do_dis_eval or args.do_gan_train or args.do_gan_eval:
    _gen, gen = build_gen(args, args.gen_load_path,
                          gen_device, args.gen_device_ids)


if args.do_dis_train or args.do_gan_train:
    _dis, dis = build_dis(args, args.dis_load_path,
                          dis_device, args.dis_device_ids)

# %%
if args.do_dis_train or args.do_dis_eval:
    dis_train = DisTrainer(args, _dis, dis, _gen)

    if args.do_dis_train:
        dis_train.train(train_dataset, valid_dataset)
    else:
        dis_train.eval(valid_dataset)

# %%
if args.do_gan_train or args.do_gan_eval:
    gan_train = GanTrainer(args, _gen, gen, _dis,
                           dis, jd_bleu.index, jd_bleu.redocstring)
    if args.do_gan_train:
        gan_train.train(train_dataset, valid_dataset, bleu_dataset)
    else:
        gan_train.eval(train_dataset)

# %%
