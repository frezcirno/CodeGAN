# %%
import logging
import argparse
from pathlib import Path
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
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    Subset,
    ConcatDataset,
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)
import bleu
from helpers import *
from model import Generator, Discriminator, Rollout
# import pytorch_lightning as pl

"""
Generator : 1997MB
Discriminator : 1601MB
"""

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
                    default=True, help="Use teacher forcing after every step.")
parser.add_argument("--gan_d_steps", type=int, default=5,
                    help="Discriminator train steps, do gan_d_sample x gan_d_epochs samples one step.")
parser.add_argument("--gan_d_sample", type=int, default=1000)
parser.add_argument("--gan_d_epochs", type=int, default=3)
parser.add_argument("--gan_d_batch_size", type=int, default=64)
parser.add_argument("--gan_d_num_workers", type=int, default=4)

args = parser.parse_args(
    args=[
        "--gen_device_ids",
        "2",
        "--dis_device_ids",
        "0", "1", "2",
        "--gen_load_path",
        "checkpoints/2.19/gen/gen_bestbleu_5_19.220000.bin",
        "--dis_load_path",
        "",
        "--do_dis_train",
        "--gan_batch_size",
        "2",
        "--gan_train_epochs",
        "4",
        "--gan_num_workers",
        "1",
        "--gan_rollnum",
        "5",
        "--gan_g_steps",
        "500",
        "--gan_teach",
        "--gan_d_steps",
        "25",
        "--gan_d_batch_size",
        "64",
    ]
    if is_notebook()
    else sys.argv[1:]
)

tokenizer = RobertaTokenizer.from_pretrained(
    path_join(args.data_dir, "tokenizer"),
    do_lower_case=args.do_lower_case
)

args.bos_token_id = 0
args.pad_token_id = 1
args.eos_token_id = 2
args.hidden_size = 768
args.vocab_size = 50265


# For using `torchrun`
try:
    args.local_rank = int(os.environ["LOCAL_RANK"])
except KeyError:
    pass

distributing = args.local_rank != -1
master = args.local_rank == 0
standalone_or_master = not distributing or master

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            datetime.now().strftime(
                f"%Y-%m-%d_%H-%M-%S_{args.local_rank}.log"
                if distributing
                else "%Y-%m-%d_%H-%M-%S.log"
            ),
            "w",
            "utf-8",
        ),
    ],
    level=logging.INFO,
)

logging.info(" ".join(sys.argv))
logging.info(str(args))

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)
data_dir = Path(args.data_dir)
data_dir.mkdir(exist_ok=True)

if distributing:
    if args.local_ranks:
        args.local_rank = args.local_ranks[args.local_rank]

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    logging.info("Distributed training: %d", distributing)

    gen_device = args.local_rank
    dis_device = args.local_rank


else:
    args.ngpu = torch.cuda.device_count()
    logging.info("GPU count = %d", args.ngpu)

    args.gen_device_ids = [
        dev_id for dev_id in args.gen_device_ids if dev_id < args.ngpu
    ]
    if not args.gen_device_ids:
        args.gen_device_ids.append(0)

    args.dis_device_ids = [
        dev_id for dev_id in args.dis_device_ids if dev_id < args.ngpu
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
# set_seed(args.seed)


@remember_result
def make_jd():
    return pd.read_parquet(args.data)


@remember_result
@cache_result(data_dir / "feats")
def make_feats():
    jd = make_jd()
    return to_features(jd, tokenizer)


@remember_result
@cache_result(data_dir / "pad_feats")
def make_pad_feats():
    feats = make_feats()
    return to_pad_features(feats,
                           args.source_length,
                           args.target_length,
                           args.bos_token_id,
                           args.eos_token_id,
                           args.pad_token_id)


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


def build_gen(args, load_path="", device=0, distributing=False, device_ids=[]):
    _model = Generator(
        args.hidden_size,
        args.vocab_size,
        args.beam_size,
        args.target_length,
        args.bos_token_id,
        args.eos_token_id,
        args.pad_token_id,
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

    if distributing:
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


def build_dis(args, load_path="", device=0, distributing=False, device_ids=[]):
    _model = Discriminator(
        args.source_length,
        args.target_length,
        args.vocab_size,
        args.hidden_size,
        args.bos_token_id,
        args.eos_token_id,
        args.pad_token_id,
        device,
    )
    if load_path:
        logging.info("Load discriminator from: %s", load_path)
        _model.load_state_dict(
            torch.load(load_path,
                       map_location=torch.device(device))
        )
    _model.to(device)

    if distributing:
        logging.info("Build DDP discriminator")
        model = DDP(
            _model,
            device_ids=[device],
            output_device=device,
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


if args.do_gen_train or args.do_gen_eval or args.do_dis_train or args.do_gan_train or args.do_gan_eval:
    _gen, gen = build_gen(args, args.gen_load_path,
                          gen_device, distributing, args.gen_device_ids)

    # TODO: Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in gen.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in gen.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    g_optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.gen_learning_rate,
        eps=args.gen_adam_epsilon,
    )


if args.do_dis_train or args.do_gan_train:
    _dis, dis = build_dis(args, args.dis_load_path,
                          dis_device, distributing, args.dis_device_ids)

    d_optimizer = optim.Adam(
        dis.parameters(), lr=args.dis_learning_rate, eps=args.dis_adam_epsilon
    )

# %%
if args.do_gen_train or args.do_gen_eval:
    # Start training
    logging.info("Do train generator:")
    logging.info("+ Num examples = %d", len(train_dataset))
    logging.info("+ Batch size = %d", args.gen_batch_size)
    logging.info("+ Train epochs = %d", args.gen_train_epochs)
    logging.info("+ Learning rate = %e", args.gen_learning_rate)
    logging.info("+ Adam epsilon = %e", args.gen_adam_epsilon)
    logging.info("+ Distributed workers = %d", args.gen_num_workers)

    if not distributing:
        gen_train_sampler = RandomSampler(train_dataset)
    else:
        gen_train_sampler = DistributedSampler(train_dataset)

    gen_train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.gen_batch_size,
        # shuffle=True,
        num_workers=args.gen_num_workers,
        pin_memory=True,
        sampler=gen_train_sampler,
    )

    gen_valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.gen_num_workers,
        pin_memory=True,
    )

    gen_valid_dataloader_bleu = DataLoader(
        bleu_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.gen_num_workers,
        pin_memory=True,
    )

    # TODO: warm up
    t_total = len(gen_train_dataloader) * args.gen_train_epochs
    gen_scheduler = get_linear_schedule_with_warmup(
        g_optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
    )
    logging.info("+ Total train steps = %d", t_total)
    logging.info("+ Warmup steps = %d", int(t_total * 0.1))
    nb_tr_examples = 0
    nb_tr_steps = 0
    tr_loss = 0
    global_step = 0
    best_bleu = 0
    best_loss = 1e6

    train_loss = 0
    checkpoint_last = path_join(args.output_dir, "gen.bin")
    checkpoint_best_ppl = path_join(args.output_dir, "gen_bestppl_%d_%f.bin")
    checkpoint_best_bleu = path_join(args.output_dir, "gen_bestbleu_%d_%f.bin")
    output_file = path_join(args.output_dir, "gen.output")
    gold_file = path_join(args.output_dir, "gen.gold")

    logging.info("+ checkpoint_last = %s", checkpoint_last)
    logging.info("+ checkpoint_best_ppl = %s", checkpoint_best_ppl)
    logging.info("+ checkpoint_best_bleu = %s", checkpoint_best_bleu)


for epoch in range(
    args.gen_train_epochs if args.do_gen_train else 1 if args.do_gen_eval else 0
):
    # Train
    gen.train()
    if args.do_gen_train:
        with tqdm(gen_train_dataloader) as train_bar:
            for batch in train_bar:
                source_ids = batch[0].to(gen_device)
                source_mask = batch[1].to(gen_device)
                target_ids = batch[2].to(gen_device)
                target_mask = batch[3].to(gen_device)

                context, memory_key_padding_mask = _gen.get_context(
                    source_ids, source_mask)
                # [batch_size x source_length x args.hidden_size]

                loss, _, _ = gen(context, memory_key_padding_mask,
                                 target_ids, target_mask)
                if loss.size():
                    loss = loss.mean()  # mean() to average on multi-gpu.
                loss.backward()

                tr_loss += loss.item()
                train_loss = round(tr_loss / (nb_tr_steps + 1), 4)
                train_bar.set_description(f"EP {epoch} loss {train_loss:.2f}")
                nb_tr_examples = nb_tr_examples + args.source_length
                nb_tr_steps += 1

                # Update parameters
                g_optimizer.step()
                g_optimizer.zero_grad()
                gen_scheduler.step()
                global_step += 1

        # Save last checkpoint
        if standalone_or_master:
            save_model(gen, checkpoint_last)

    # Eval
    if args.do_gen_eval and standalone_or_master:
        # Eval model with dev dataset
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        logging.info("Do evaluation:")
        logging.info("+ Num examples = %d", len(valid_dataset))
        logging.info("+ Batch size = %d", args.gen_batch_size)

        # Start Evaling model
        # Save best checkpoint for best ppl
        gen.eval()
        eval_loss, tokens_num = 0, 0
        for batch in tqdm(gen_valid_dataloader, "eval"):
            source_ids = batch[0].to(gen_device)
            source_mask = batch[1].to(gen_device)
            target_ids = batch[2].to(gen_device)
            target_mask = batch[3].to(gen_device)

            with torch.no_grad():
                context, memory_key_padding_mask = _gen.get_context(
                    source_ids, source_mask
                )
                # [batch_size x source_length x args.hidden_size]

                loss, num, _ = gen(
                    context, memory_key_padding_mask, target_ids, target_mask
                )
                loss *= num
                if loss.size():
                    num = num.sum()
                    loss = loss.mean()  # mean() to average on multi-gpu.
            eval_loss += loss.item()
            tokens_num += num.item()

        gen.train()
        eval_loss = eval_loss / tokens_num
        logging.info("Evaluation result:")
        logging.info("+ eval_loss = %s", round(eval_loss, 5))
        logging.info("+ global_step = %s", global_step + 1)
        logging.info("+ train_loss (avg.) = %s", round(train_loss, 5))

        if eval_loss < best_loss:
            logging.info("+ Best loss: %s", round(eval_loss, 5))
            best_loss = eval_loss

            save_model(gen, checkpoint_best_ppl % (epoch, best_loss))

        # Save best checkpoint for best bleu
        gen.eval()
        predicts = []
        for batch in tqdm(gen_valid_dataloader_bleu, "bleu"):
            source_ids = batch[0].to(gen_device)
            source_mask = batch[1].to(gen_device)
            with torch.no_grad():
                preds = _gen.beam_predict(source_ids, source_mask)
                predicts += tensors_to_text(tokenizer, preds)

        gen.train()
        predictions = write_output(
            output_file, gold_file, bleu_feats.index, predicts, jd_bleu.redocstring,
        )

        goldMap, predictionMap = bleu.computeMaps(predictions, gold_file)
        dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        logging.info("+ bleu-4 = %f", dev_bleu)

        if dev_bleu > best_bleu:
            logging.info("+ Best bleu = %s", dev_bleu)
            best_bleu = dev_bleu
            save_model(gen, checkpoint_best_bleu % (epoch, best_bleu))

# %%


def fakegen(dataset):
    '''when in distributing, every process will generate and return a part of the mixed data'''
    if distributing:
        sampler = DistributedSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    if not args.fakegen_batch_size:
        args.fakegen_batch_size = 64 * 3

    dataloader = DataLoader(
        dataset,
        batch_size=args.fakegen_batch_size,
        num_workers=args.fakegen_num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    return make_fake_dataset2(_gen, dataloader)


if args.do_dis_train:
    logging.info("Do discriminator train:")

    logging.info("+ generate train sample = %d", args.dis_fakegen_train_sample)
    logging.info("+ generate valid sample = %d", args.dis_fakegen_valid_sample)

    train_dataset_sub = Subset(
        train_dataset, range(args.dis_fakegen_train_sample))

    valid_dataset_sub = Subset(
        valid_dataset, range(args.dis_fakegen_valid_sample))

    path = path_join(args.data_dir,
                     f"fake_valid_dataset_{args.dis_fakegen_valid_sample}")
    fake_valid_dataset = cache_call(
        path, fakegen, format="torch")(valid_dataset_sub)
    dis_valid_dataset = mix_dataset(
        valid_dataset_sub, fake_valid_dataset, [0, 2])

    min_loss = 1e6
    dis_last_path = path_join(args.output_dir, "dis.bin")
    dis_best_path = path_join(args.output_dir, "dis_bestppl_%d_%f.bin")

    logging.info("+ Train dataset = %d", 2 * args.dis_fakegen_train_sample)
    logging.info("+ Valid dataset = %d", 2 * args.dis_fakegen_valid_sample)
    logging.info("+ dis_batch_size = %s", args.dis_batch_size)
    logging.info("+ Learning rate = %s", args.dis_learning_rate)
    logging.info("+ Adam epsilon = %e", args.dis_adam_epsilon)
    logging.info("+ dis_last_path = %s", dis_last_path)
    logging.info("+ dis_best_path = %s", dis_best_path)

for epoch in range(args.dis_train_epochs if args.do_dis_train else 0):

    dis_train_dataset = fakegen(train_dataset_sub)

    mixed_train_dataloader = DataLoader(
        dis_train_dataset,
        batch_size=args.dis_batch_size,
        num_workers=args.dis_num_workers,
        pin_memory=True,
    )

    dis.train()
    with tqdm(mixed_train_dataloader) as bar:
        for batch in bar:
            source_ids = batch[0].to(dis_device)
            target_ids = batch[1].to(dis_device)
            labels = batch[2].to(dis_device)

            pred = dis(source_ids, target_ids)
            loss = F.binary_cross_entropy(pred, labels)
            # mean() to average on multi-gpu
            if loss.size():
                loss = loss.mean()
            bar.set_description(f"EP {epoch} loss {loss.item():.2f}")

            loss.backward()
            d_optimizer.step()
            d_optimizer.zero_grad()

    if standalone_or_master:
        save_model(dis, dis_last_path)

    all_loss, loss_num = 0, 0

    mixed_valid_dataloader = DataLoader(
        dis_valid_dataset,
        batch_size=args.dis_batch_size,
        num_workers=args.dis_num_workers,
        pin_memory=True,
    )

    dis.eval()
    with torch.no_grad():
        with tqdm(mixed_valid_dataloader) as bar:
            for batch in bar:
                source_ids = batch[0].to(dis_device)
                target_ids = batch[1].to(dis_device)
                labels = batch[2].to(dis_device)

                pred = dis(source_ids, target_ids)
                loss = F.binary_cross_entropy(pred, labels)
                # mean() to average on multi-gpu
                if loss.size():
                    loss = loss.mean()
                all_loss += loss.item()
                loss_num += 1
                bar.set_description(f"dis eval loss {loss.item():.2f}")

    all_loss /= loss_num
    logging.info("+ eval loss = %f", all_loss)

    if all_loss < min_loss:
        logging.info("+ Best loss !!")
        min_loss = all_loss
        if standalone_or_master:
            save_model(dis, dis_best_path % (epoch, min_loss))

# %%
if args.do_gan_train or args.do_gan_eval:
    logging.info("Do GAN train:")

    logging.info("+ train sample = %d", len(train_dataset))
    logging.info("+ Generator learning rate = %s", args.gen_learning_rate)
    logging.info("+ Generator adam epsilon = %e", args.gen_adam_epsilon)
    logging.info("+ Discriminator learning rate = %s", args.dis_learning_rate)
    logging.info("+ Discriminator adam epsilon = %e", args.dis_adam_epsilon)

    if distributing:
        gan_train_sampler = DistributedSampler(train_dataset)
    else:
        gan_train_sampler = RandomSampler(train_dataset)

    gan_train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.gan_batch_size,
        num_workers=args.gan_num_workers,
        pin_memory=True,
        sampler=gan_train_sampler,
    )

    logging.info("+ train dataset = %d", len(train_dataset))
    logging.info("+ valid dataset = %d", len(valid_dataset))
    logging.info("+ bleu dataset = %d", len(bleu_dataset))

    logging.info("+ g-steps = %d", args.gan_g_steps)
    logging.info("+ d-steps = %d", args.gan_d_steps)
    logging.info("+ d-sample = %d", args.gan_d_sample)
    logging.info("+ d-epochs = %d", args.gan_d_epochs)

    logging.info("+ GAN batch size = %d", args.gan_batch_size)
    logging.info("+ GAN num workers = %d", args.gan_num_workers)

    t_total = len(gan_train_dataloader) * args.gan_train_epochs
    if args.gan_teach:
        t_total = t_total * 2
    logging.info("+ Teacher forcing = %s", args.gan_teach)
    gan_gen_scheduler = get_linear_schedule_with_warmup(
        g_optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
    )
    logging.info("+ Total train steps = %d", t_total)
    logging.info("+ Rollout num = %d", args.gan_rollnum)
    logging.info("+ Warmup steps = %d", int(t_total * 0.1))


if args.do_gan_train:
    rollout = Rollout(gen, dis, args.target_length)

    gan_gen_path = path_join(args.output_dir, "gan_gen.bin")
    gan_dis_path = path_join(args.output_dir, "gan_dis.bin")
    gan_gen_best_ppl = path_join(args.output_dir, "gan_gen_bestppl_%d_%f.bin")
    gan_dis_best_ppl = path_join(args.output_dir, "gan_dis_bestppl_%d_%f.bin")
    gan_gen_best_bleu = path_join(
        args.output_dir, "gan_gen_bestbleu_%d_%f.bin")
    gan_dis_best_bleu = path_join(
        args.output_dir, "gan_dis_bestbleu_%d_%f.bin")

    logging.info("+ gan_gen_path = %s", gan_gen_path)
    logging.info("+ gan_dis_path = %s", gan_dis_path)
    logging.info("+ gan_gen_best_ppl = %s", gan_gen_best_ppl)
    logging.info("+ gan_dis_best_ppl = %s", gan_dis_best_ppl)
    logging.info("+ gan_gen_best_bleu = %s", gan_gen_best_bleu)
    logging.info("+ gan_dis_best_bleu = %s", gan_dis_best_bleu)

    best_bleu = 0
    best_loss = 1e6

    train_iter = iter(itertools.cycle(gan_train_dataloader))
    train_iter2 = iter(itertools.cycle(gan_train_dataloader))

    gan_d_dataset = Subset(train_dataset, range(args.gan_d_sample))

if args.do_gan_train or args.do_gan_eval:
    gan_output_file = path_join(args.output_dir, "gan_dev.output")
    gan_gold_file = path_join(args.output_dir, "gan_dev.gold")

    logging.info("+ gan_output_file = %s", gan_output_file)
    logging.info("+ gan_gold_file = %s", gan_gold_file)


def gan_train(train_iter):
    batch = next(train_iter)

    source_ids = batch[0].to(gen_device)
    source_mask = batch[1].to(gen_device)

    context, memory_key_padding_mask = _gen.get_context(
        source_ids, source_mask)
    # [batch_size x source_length x args.hidden_size]

    pre_target_ids, _ = gen(context, memory_key_padding_mask)
    pre_target_ids, pre_target_mask = get_target_mask(
        pre_target_ids, args.bos_token_id, args.eos_token_id, args.pad_token_id,
    )
    rewards = rollout.get_reward(
        source_ids,
        context,
        memory_key_padding_mask,
        pre_target_ids,
        pre_target_mask,
        rollnum=args.gan_rollnum,
    )
    loss = gen(context, memory_key_padding_mask,
               pre_target_ids, rewards=rewards)

    if loss.size():
        loss = loss.mean()

    loss.backward()
    g_optimizer.step()
    g_optimizer.zero_grad()
    gan_gen_scheduler.step()

    return loss.item()


def gan_gen_train(train_iter2):
    batch = next(train_iter2)

    source_ids = batch[0].to(gen_device)
    source_mask = batch[1].to(gen_device)
    target_ids = batch[2].to(gen_device)
    target_mask = batch[3].to(gen_device)

    context, memory_key_padding_mask = _gen.get_context(
        source_ids, source_mask)
    # [batch_size x source_length x args.hidden_size]

    rewards = torch.ones_like(target_ids) * target_mask
    tloss = gen(context, memory_key_padding_mask,
                target_ids, rewards=rewards)

    if tloss.size():
        tloss = tloss.mean()

    tloss.backward()
    g_optimizer.step()
    g_optimizer.zero_grad()
    gan_gen_scheduler.step()

    return tloss.item()


def gan_dis_step(dis_train_dataset):
    dataloader = DataLoader(
        dis_train_dataset,
        batch_size=args.gan_d_batch_size,
        num_workers=args.gan_d_num_workers,
        pin_memory=True,
    )

    # Train dis for k epochs
    for d_epoch in trange(args.gan_d_epochs, desc="d-step epoch"):
        d_loss_acc = 0.0
        d_loss_count = 0
        d_loss_avg = 0.0
        with tqdm(dataloader, "loss 0.0000") as bar:
            for batch in bar:
                source_ids = batch[0].to(dis_device)
                target_ids = batch[1].to(dis_device)
                label = batch[2].to(dis_device)

                pred = dis(source_ids, target_ids)
                loss = F.binary_cross_entropy(pred, label)
                if loss.size():
                    loss = loss.mean()  # mean() to average on multi-gpu

                loss.backward()
                d_optimizer.step()
                d_optimizer.zero_grad()

                loss = loss.item()
                d_loss_acc += loss
                d_loss_count += 1
                d_loss_avg = d_loss_acc / d_loss_count
                bar.set_description(f"loss {d_loss_avg:.4f}")

        logging.info("d-epoch train loss %f", d_loss_avg)


def get_loss(valid_dataset):
    eval_loss = 0
    tokens_num = 0
    gan_valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.gan_num_workers,
        pin_memory=True,
    )

    for batch in tqdm(gan_valid_dataloader, "validation"):
        source_ids = batch[0].to(gen_device)
        source_mask = batch[1].to(gen_device)
        target_ids = batch[2].to(gen_device)
        target_mask = batch[3].to(gen_device)

        with torch.no_grad():
            context, memory_key_padding_mask = _gen.get_context(
                source_ids, source_mask
            )
            # [batch_size x source_length x args.hidden_size]

            loss, num, _ = gen(
                context, memory_key_padding_mask, target_ids, target_mask
            )
            loss *= num
            if loss.size():
                num = num.sum()
                loss = loss.mean()  # mean() to average on multi-gpu.
        eval_loss += loss.item()
        tokens_num += num.item()

    eval_loss = eval_loss / tokens_num
    return eval_loss


def get_bleu(bleu_dataset):
    # Save best checkpoint for best bleu
    gan_valid_dataloader_bleu = DataLoader(
        bleu_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.gan_num_workers,
        pin_memory=True,
    )
    predicts = []
    for batch in tqdm(gan_valid_dataloader_bleu, "bleu"):
        source_ids = batch[0]
        source_mask = batch[1]
        with torch.no_grad():
            preds = _gen.beam_predict(
                source_ids.to(gen_device),
                source_mask.to(gen_device))
            predicts += tensors_to_text(tokenizer, preds)

    predictions = write_output(
        gan_output_file,
        gan_gold_file,
        bleu_feats.index,
        predicts,
        jd_bleu.redocstring,
    )

    goldMap, predictionMap = bleu.computeMaps(predictions, gan_gold_file)
    dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    return dev_bleu


for epoch in trange(args.gan_train_epochs if args.do_gan_train else 1 if args.do_gan_eval else 0, desc="Epoch"):
    if args.do_gan_train:
        # G-step
        gen.to(gen_device)
        dis.to(dis_device)
        gen.train()
        dis.eval()
        g_loss_acc = 0.0
        g_loss_count = 0
        g_loss_avg = 0.0
        with trange(args.gan_g_steps, desc="g-step loss 0.0000") as g_step_bar:
            g_step = iter(g_step_bar)
            while True:
                try:
                    next(g_step)
                except StopIteration:
                    break

                loss = gan_train(train_iter)
                g_loss_acc += loss
                g_loss_count += 1
                g_loss_avg = g_loss_acc / g_loss_count

                g_step_bar.set_description(f"g-step loss {g_loss_avg:.4f}")


        if args.gan_teach:
            dis.to("cpu")
            gen.to(gen_device)
            logging.info("Do teacher forcing:")
            with trange(args.gan_g_steps, desc="g-step 0.0000") as g_step_bar:
                g_step = iter(g_step_bar)
                while True:
                    try:
                        next(g_step)
                    except StopIteration:
                        break

                    tloss = gan_gen_train(train_iter2)
                    g_loss_acc += tloss
                    g_loss_count += 1
                    g_loss_avg = g_loss_acc / g_loss_count

                    g_step_bar.set_description(f"g-step loss {g_loss_avg:.4f}")

        logging.info("g-step train avg loss: %f", g_loss_avg)

        save_model(gen, gan_gen_path)

        # D-step
        gen.eval()
        dis.train()
        for d_step in trange(args.gan_d_steps, desc="d-step"):
            # (re)generate fake dataset
            gen.to(gen_device)
            dis_train_dataset = fakegen(gan_d_dataset)
            gen.to("cpu")

            dis.to(dis_device)
            gan_dis_step(dis_train_dataset)
            dis.to("cpu")

        save_model(dis, gan_dis_path)

    if (args.do_gan_train or args.do_gan_eval) and standalone_or_master:
        # Eval G with dev dataset
        logging.info("Do evaluation:")
        logging.info("+ Valid dataset = %d", len(valid_dataset))
        logging.info("+ batch size = %d", args.eval_batch_size)
        logging.info("+ num workers = %d", args.gan_num_workers)

        if args.do_gan_train:
            dis.to("cpu")
        gen.eval()
        gen.to(gen_device)

        eval_loss = get_loss(valid_dataset)
        logging.info("+ eval loss = %f", eval_loss)

        if args.do_gan_train and eval_loss < best_loss:
            logging.info("+ Best loss !!")
            best_loss = eval_loss

            save_model(gen, gan_gen_best_ppl % (epoch, best_loss))
            save_model(dis, gan_dis_best_ppl % (epoch, best_loss))

        logging.info("+ Bleu dataset = %d", len(bleu_dataset))
        logging.info("+ batch size = %d", args.eval_batch_size)
        logging.info("+ num workers = %d", args.gan_num_workers)

        dev_bleu = get_bleu(bleu_dataset)
        logging.info("+ bleu-4 = %f", dev_bleu)

        if args.do_gan_train and dev_bleu > best_bleu:
            logging.info("+ Best bleu !!")
            best_bleu = dev_bleu

            save_model(gen, gan_gen_best_bleu % (epoch, best_bleu))
            save_model(dis, gan_dis_best_bleu % (epoch, best_bleu))


# %%
