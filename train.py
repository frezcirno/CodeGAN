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
    DisTrainer, GanTrainer, GenTrainer, is_distributed, is_notebook, local_rank, rank, remember_result, set_seed,
    cache_result, to_features, to_pad_features, series_to_tensor, sample_dataset, world_size)
from memory import occupy_mem
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
parser.add_argument("--device_ids", nargs="+", type=int, default=[])
parser.add_argument("--gen_device_ids", nargs="+", type=int, default=[])
parser.add_argument("--dis_device_ids", nargs="+", type=int, default=[])
parser.add_argument("--gen_no_ddp", action="store_true")
parser.add_argument("--dis_no_ddp", action="store_true")
parser.add_argument("--num_workers", type=int, default=4)

# Gen
parser.add_argument("--do_gen_train", action="store_true")
parser.add_argument("--do_gen_eval", action="store_true")
parser.add_argument("--gen_batch_size", type=int, default=128)
parser.add_argument("--eval_batch_size", type=int, default=64)
parser.add_argument(
    "--gen_load_path", help="The path to the generator model",
)
parser.add_argument("--gen_train_epochs", type=int, default=10)
parser.add_argument("--gen_learning_rate", type=float, default=5e-5)
parser.add_argument("--gen_adam_epsilon", type=float, default=1e-8)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--beam_size", type=int, default=10)

# Dis
parser.add_argument("--do_dis_train", action="store_true")
parser.add_argument("--do_dis_eval", action="store_true")
parser.add_argument("--dis_train_sample", type=int, default=5000)
parser.add_argument("--dis_valid_sample", type=int, default=1000)
parser.add_argument("--fakegen_batch_size", type=int, default=64)
parser.add_argument(
    "--dis_load_path", help="The path to the discriminator model",
)
parser.add_argument("--dis_train_epochs", type=int, default=10)
parser.add_argument("--dis_batch_size", type=int, default=64)
parser.add_argument("--dis_learning_rate", type=float, default=5e-5)
parser.add_argument("--dis_adam_epsilon", type=float, default=1e-8)
parser.add_argument("--dis_early_stop_loss", default=1e-5)

# Gan
parser.add_argument("--do_gan_train", action="store_true")
parser.add_argument("--do_gan_eval", action="store_true")
parser.add_argument("--gan_batch_size", type=int, default=16)
parser.add_argument("--gan_train_epochs", type=int, default=30)
parser.add_argument("--gan_rollnum", type=int, default=20)
parser.add_argument("--gan_g_steps", type=int, default=100,
                    help="Generator train steps, one batch one step.")
parser.add_argument("--gan_teach", action="store_true",
                    help="Use teacher forcing after every step.")
parser.add_argument("--gan_d_steps", type=int, default=5,
                    help="Discriminator train steps, do gan_d_sample x gan_d_epochs samples one step.")
parser.add_argument("--gan_d_sample", type=int, default=1000)
parser.add_argument("--gan_d_epochs", type=int, default=6)
parser.add_argument("--gan_d_batch_size", type=int, default=64)

DIS_TRAIN_ARGS = "--gen_load_path checkpoints/3.8_3/gan_gen_bestbleu_0_19.520000.bin --dis_load_path dis_bestloss_3_0.350139.bin --do_dis_train --dis_train_sample 500 --dis_valid_sample 100 --fakegen_batch_size 64 --device_ids 2 --output_dir 3.18-1".split()

GAN_TRAIN_ARGS = "--gen_load_path checkpoints/3.8_3/gan_gen_bestbleu_0_19.520000.bin --dis_load_path checkpoints/3.8_3/gan_dis_bestbleu_0_19.520000.bin --do_gan_train --gan_teach --gan_train_epochs 10 --gan_batch_size 2 --gan_rollnum 20 --gan_g_step 100 --gan_d_sample 1000 --gan_d_step 10 --gan_d_epochs 10 --fakegen_batch_size 32 --device_ids 3 --output_dir current".split()

args = parser.parse_args(DIS_TRAIN_ARGS if is_notebook() else sys.argv[1:])

args.hidden_size = 768
args.vocab_size = 50265


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
                f"%Y-%m-%d_%H-%M-%S_{local_rank()}.log"
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


def validate_device_ids(device_ids):
    ngpu = torch.cuda.device_count()

    return [dev_id for dev_id in device_ids if dev_id < ngpu] or [0]


logging.info("GPU count = %d", torch.cuda.device_count())

if is_distributed():
    logging.info(
        f"Distributed training: rank {rank()} world_size {world_size()} local_rank {local_rank()}")

    device_ids = validate_device_ids(args.device_ids)
    local_device_id = device_ids[local_rank()]
    logging.info(f"Using device {local_device_id}")

    gen_device_ids = [local_device_id]
    gen_device = local_device_id
    dis_device_ids = [local_device_id]
    dis_device = local_device_id
    occupy_mem(local_device_id)
    torch.cuda.set_device(local_device_id)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl", init_method='env://')

else:
    gen_device_ids = validate_device_ids(
        args.gen_device_ids or args.device_ids)
    gen_device = gen_device_ids[0]

    for device in gen_device_ids:
        occupy_mem(device)

    dis_device_ids = validate_device_ids(
        args.dis_device_ids or args.device_ids)
    dis_device = dis_device_ids[0]

    for device in dis_device_ids:
        occupy_mem(device)


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


def build_parallel(raw_model, args, load_path="", device=0, device_ids=[]):
    if load_path:
        logging.info(f"Load model from {load_path}")
        weights = torch.load(load_path,
                             map_location=torch.device(device))
        if 'encoder.pooler.dense.weight' in weights:
            del weights['encoder.pooler.dense.weight']
            del weights['encoder.pooler.dense.bias']
        raw_model.load_state_dict(weights)

    raw_model.to(device)

    if is_distributed():
        logging.info("Using DistributedDataParallel")
        model = DDP(
            raw_model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=args.do_gan_train,
            broadcast_buffers=False,
        )
    elif len(device_ids) > 1:
        logging.info("Using DataParallel")
        model = DP(raw_model, device_ids)
        model.device = device
    else:
        model = raw_model

    return model


"""
Generator: 1995 MiB
Discriminator: 556 MiB
Total: 2551 MiB
"""
if args.do_gen_train or args.do_gen_eval or args.do_dis_train or args.do_dis_eval or args.do_gan_train or args.do_gan_eval:
    _gen = Generator(
        args.hidden_size,
        args.vocab_size,
        args.beam_size,
        args.target_length,
        gen_device,
    )

    gen = build_parallel(_gen, args, args.gen_load_path,
                         gen_device, gen_device_ids)


if args.do_dis_train or args.do_dis_eval or args.do_gan_train:
    _dis = Discriminator(
        args.source_length,
        args.target_length,
        args.vocab_size,
        args.hidden_size,
        dis_device,
    )

    dis = build_parallel(_dis, args, args.dis_load_path,
                         dis_device, dis_device_ids)


# %%
if args.do_gen_train or args.do_gen_eval:
    gen_train = GenTrainer(args, _gen, gen, jd_bleu.index, jd_bleu.docstring)

    if args.do_gen_train:
        gen_train.train(train_dataset, valid_dataset, bleu_dataset)
    else:
        gen_train.eval(valid_dataset, bleu_dataset)

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
                           dis, jd_bleu.index, jd_bleu.docstring)
    if args.do_gan_train:
        gan_train.train(train_dataset, valid_dataset, bleu_dataset)
    else:
        gan_train.eval(train_dataset)
