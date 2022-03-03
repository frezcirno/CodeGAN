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
import pytorch_lightning as pl

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
parser.add_argument("--gen_device_ids", nargs="+", type=int, default=[0])
parser.add_argument("--dis_device_ids", nargs="+", type=int, default=[0])
parser.add_argument("--gen_no_ddp", action="store_true")
parser.add_argument("--dis_no_ddp", action="store_true")

parser.add_argument(
    "--local_rank", type=int, default=-1, help="For distributed training: local_rank",
)
parser.add_argument(
    "--rank_offset", type=int, default=0, help="For distributed training: rank_offset",
)

# Gen
parser.add_argument("--do_gen_train", action="store_true")
parser.add_argument("--do_gen_eval", action="store_true")
parser.add_argument("--gen_batch_size", type=int, default=128)
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
parser.add_argument("--dis_fakegen_batch_size", type=int, default=64)
parser.add_argument("--dis_fakegen_num_workers", type=int, default=4)
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
parser.add_argument("--gan_batch_size", type=int, default=1)
parser.add_argument("--gan_train_epochs", type=int, default=30)
parser.add_argument("--gan_num_workers", type=int, default=4)
parser.add_argument("--gan_rollnum", type=int, default=5)
parser.add_argument("--gan_g_steps", type=int, default=1000)
parser.add_argument("--gan_teach", action="store_true", default=True)
parser.add_argument("--gan_d_steps", type=int, default=30)
parser.add_argument("--gan_d_batch_size", type=int, default=64)
parser.add_argument("--gan_d_num_workers", type=int, default=4)
parser.add_argument("--gan_d_sample", type=int, default=1000)
parser.add_argument("--gan_d_epochs", type=int, default=5)

args = parser.parse_args(
    args=[
        "--gen_device_ids",
        "1",
        "--dis_device_ids",
        "1",
        "--gen_load_path",
        "checkpoints/2.19/gen/gen_bestbleu_5_19.220000.bin",
        "--dis_load_path",
        "checkpoints/2.19/dis/dis_bestppl_1_0.473262.bin",
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

args.hidden_size = 768
args.vocab_size = 50265

# For using `torchrun`
try:
    args.local_rank = int(os.environ["LOCAL_RANK"])
except KeyError:
    pass

if args.local_rank != -1:
    args.local_rank += args.rank_offset

distributing = args.local_rank != -1
master = args.local_rank == args.rank_offset
standalone_or_master = not distributing or master

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            datetime.now().strftime(
                f"%Y-%m-%d_%H-%M-%S_{args.local_rank}.log"
                if args.local_rank != -1
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

if args.local_rank != -1:
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    logging.info("Distributed training: %d", args.local_rank != -1)

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


# %%
set_seed(args.seed)

tokenizer = RobertaTokenizer.from_pretrained(
    "microsoft/codebert-base", do_lower_case=args.do_lower_case
)

args.bos_token_id = 0
args.pad_token_id = 1
args.eos_token_id = 2


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
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
    elif len(device_ids) > 1:
        logging.info("Build DP discriminator")
        model = DP(_model, device_ids)
    else:
        logging.info("Build discriminator")
        model = _model

    return _model, model


if args.do_gen_train or args.do_gen_eval or args.do_dis_train or args.do_gan_train:
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

    if args.local_rank == -1:
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
        batch_size=args.gen_batch_size,
        num_workers=args.gen_num_workers,
        pin_memory=True,
    )

    gen_valid_dataloader_bleu = DataLoader(
        bleu_dataset,
        batch_size=args.gen_batch_size,
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
        train_bar = tqdm(gen_train_dataloader)
        for batch in train_bar:
            source_ids = batch[0].to(gen_device)
            source_mask = batch[1].to(gen_device)
            target_ids = batch[2].to(gen_device)
            target_mask = batch[3].to(gen_device)

            g_optimizer.zero_grad()
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
if args.do_dis_train:
    logging.info("Do discriminator train:")

    logging.info("+ generate train sample = %d", args.dis_fakegen_train_sample)
    logging.info("+ generate valid sample = %d", args.dis_fakegen_valid_sample)

    train_dataset_sub = Subset(
        train_dataset, range(args.dis_fakegen_train_sample))

    valid_dataset_sub = Subset(
        valid_dataset, range(args.dis_fakegen_valid_sample))

    def __make(dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=args.dis_fakegen_batch_size,
            num_workers=args.dis_fakegen_num_workers,
            pin_memory=True,
        )
        return make_fake_dataset(_gen, dataloader)

    path = path_join(args.data_dir,
                     f"fake_train_dataset_{args.dis_fakegen_train_sample}")
    fake_train_dataset = cache_call(
        path, __make, format="torch")(train_dataset_sub)

    path = path_join(args.data_dir,
                     f"fake_valid_dataset_{args.dis_fakegen_valid_sample}")
    fake_valid_dataset = cache_call(
        path, __make, format="torch")(valid_dataset_sub)

    dis_train_dataset = mix_dataset(
        train_dataset_sub, fake_train_dataset, [0, 2])
    dis_valid_dataset = mix_dataset(
        valid_dataset_sub, fake_valid_dataset, [0, 2])

    min_loss = 1e6
    dis_last_path = path_join(args.output_dir, "dis.bin")
    dis_best_path = path_join(args.output_dir, "dis_bestppl_%d_%f.bin")

    logging.info("+ Train dataset = %d", len(dis_train_dataset))
    logging.info("+ Valid dataset = %d", len(dis_valid_dataset))
    logging.info("+ dis_batch_size = %s", args.dis_batch_size)
    logging.info("+ Learning rate = %s", args.dis_learning_rate)
    logging.info("+ Adam epsilon = %e", args.dis_adam_epsilon)
    logging.info("+ dis_last_path = %s", dis_last_path)
    logging.info("+ dis_best_path = %s", dis_best_path)

    if args.local_rank == -1:
        dis_train_sampler = RandomSampler(dis_train_dataset)
    else:
        dis_train_sampler = DistributedSampler(dis_train_dataset)

    mixed_train_dataloader = DataLoader(
        dis_train_dataset,
        batch_size=args.dis_batch_size,
        num_workers=args.dis_num_workers,
        pin_memory=True,
        sampler=dis_train_sampler,
    )
    mixed_valid_dataloader = DataLoader(
        dis_valid_dataset,
        batch_size=args.dis_batch_size,
        num_workers=args.dis_num_workers,
        pin_memory=True,
    )

for epoch in range(args.dis_train_epochs if args.do_dis_train else 0):
    dis.train()
    bar = tqdm(mixed_train_dataloader)
    for batch in bar:
        source_ids = batch[0].to(dis_device)
        target_ids = batch[1].to(dis_device)
        labels = batch[2].to(dis_device)

        d_optimizer.zero_grad()
        pred = dis(source_ids, target_ids)
        loss = F.binary_cross_entropy(pred, labels)
        # mean() to average on multi-gpu
        if loss.size():
            loss = loss.mean()
        bar.set_description(f"EP {epoch} loss {loss.item():.2f}")

        loss.backward()
        d_optimizer.step()

    if standalone_or_master:
        save_model(dis, dis_last_path)

    all_loss, loss_num = 0, 0

    dis.eval()
    with torch.no_grad():
        bar = tqdm(mixed_valid_dataloader)
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

    if args.local_rank == -1:
        gan_train_sampler = RandomSampler(train_dataset)
    else:
        gan_train_sampler = DistributedSampler(train_dataset)

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
    logging.info("+ Warmup steps = %d", int(t_total * 0.1))

    rollout = Rollout(gen, dis, args.target_length)

    gan_gen_path = path_join(args.output_dir, "gan_gen.bin")
    gan_dis_path = path_join(args.output_dir, "gan_dis.bin")
    gan_gen_best_ppl = path_join(args.output_dir, "gan_gen_bestppl_%d_%f.bin")
    gan_dis_best_ppl = path_join(args.output_dir, "gan_dis_bestppl_%d_%f.bin")
    gan_gen_best_bleu = path_join(
        args.output_dir, "gan_gen_bestbleu_%d_%f.bin")
    gan_dis_best_bleu = path_join(
        args.output_dir, "gan_dis_bestbleu_%d_%f.bin")
    gan_output_file = path_join(args.output_dir, "gan_dev.output")
    gan_gold_file = path_join(args.output_dir, "gan_dev.gold")

    logging.info("+ gan_gen_path = %s", gan_gen_path)
    logging.info("+ gan_dis_path = %s", gan_dis_path)
    logging.info("+ gan_gen_best_ppl = %s", gan_gen_best_ppl)
    logging.info("+ gan_dis_best_ppl = %s", gan_dis_best_ppl)
    logging.info("+ gan_gen_best_bleu = %s", gan_gen_best_bleu)
    logging.info("+ gan_dis_best_bleu = %s", gan_dis_best_bleu)
    logging.info("+ gan_output_file = %s", gan_output_file)
    logging.info("+ gan_gold_file = %s", gan_gold_file)

    best_bleu = 0
    best_loss = 1e6

    train_cycle_iter = iter(itertools.cycle(gan_train_dataloader))

    gan_d_dataset = Subset(train_dataset, range(args.gan_d_sample))

    gan_d_dataloader = DataLoader(
        gan_d_dataset,
        batch_size=args.dis_fakegen_batch_size,
        num_workers=args.dis_fakegen_num_workers,
        pin_memory=True,
    )

for epoch in range(args.gan_train_epochs if args.do_gan_train else 0):

    # G-step
    gen.train()
    g_step_bar = trange(args.gan_g_steps)
    g_step = iter(g_step_bar)
    while True:
        try:
            next(g_step)
        except StopIteration:
            break

        batch = next(train_cycle_iter)

        source_ids = batch[0].to(gen_device)
        source_mask = batch[1].to(gen_device)

        g_optimizer.zero_grad()

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

        g_step_bar.set_description(f"EP {epoch} g-step loss {loss.item():.2f}")

        loss.backward()
        g_optimizer.step()
        gan_gen_scheduler.step()

        if args.gan_teach:
            # teacher forcing
            target_ids = batch[2].to(gen_device)
            target_mask = batch[3].to(gen_device)

            g_optimizer.zero_grad()

            context, memory_key_padding_mask = _gen.get_context(
                source_ids, source_mask)
            # [batch_size x source_length x args.hidden_size]

            rewards = torch.ones_like(target_ids) * \
                target_mask  # right answer!
            tloss = gen(context, memory_key_padding_mask,
                        target_ids, rewards=rewards)

            if tloss.size():
                tloss = tloss.mean()

            g_step_bar.set_description(
                f"EP {epoch} g-step {loss.item():.2f} {tloss.item():.2f}"
            )

            tloss.backward()
            g_optimizer.step()
            gan_gen_scheduler.step()

    save_model(gen, gan_gen_path)

    # D-step
    gen.eval()
    dis.train()
    for d_step in trange(args.gan_d_steps):
        # (re)generate fake dataset

        fake_train_dataset = make_fake_dataset(_gen, gan_d_dataloader)
        dis_train_dataset = ConcatDataset([gan_d_dataset, fake_train_dataset])

        sampler = RandomSampler(dis_train_dataset)
        mixed_valid_dataloader = DataLoader(
            dis_train_dataset,
            sampler=sampler,
            batch_size=args.gan_d_batch_size,
            num_workers=args.gan_d_num_workers,
            pin_memory=True,
        )

        # Train dis for k epochs
        for d_epoch in args.gan_d_epochs:
            dis_train_dataset_bar = tqdm(dis_train_dataset)
            for batch in dis_train_dataset_bar:
                source_ids = batch[0].to(dis_device)
                target_ids = batch[1].to(dis_device)
                label = batch[2].to(dis_device)

                d_optimizer.zero_grad()

                pred = dis(source_ids, target_ids)
                loss = F.binary_cross_entropy(pred, label)
                if loss.size():
                    loss = loss.mean()  # mean() to average on multi-gpu

                dis_train_dataset_bar.set_description(
                    f"{epoch}:{d_step}:{d_epoch} loss {loss.item():.2f}"
                )

                loss.backward()
                d_optimizer.step()

    save_model(dis, gan_dis_path)

    if args.do_gan_eval and standalone_or_master:
        # Eval G with dev dataset
        logging.info("Do evaluation:")
        logging.info("+ Valid dataset = %d", len(valid_dataset))

        gen.eval()
        eval_loss = 0
        tokens_num = 0
        gan_valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.gan_batch_size,
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
        logging.info("+ eval loss = %f", eval_loss)

        if args.do_gan_train and eval_loss < best_loss:
            logging.info("+ Best loss !!")
            best_loss = eval_loss

            save_model(gen, gan_gen_best_ppl % (epoch, best_loss))
            save_model(dis, gan_dis_best_ppl % (epoch, best_loss))

        # Save best checkpoint for best bleu
        gen.eval()
        gan_valid_dataloader_bleu = DataLoader(
            bleu_dataset,
            batch_size=args.gan_batch_size,
            num_workers=args.gan_num_workers,
            pin_memory=True,
        )
        predicts = []
        for batch in tqdm(gan_valid_dataloader_bleu, "bleu"):
            source_ids = batch[0].to(gen_device)
            source_mask = batch[1].to(gen_device)
            with torch.no_grad():
                preds = _gen.beam_predict(source_ids, source_mask)
                predicts += tensors_to_text(tokenizer, preds)

        gen.train()
        predictions = write_output(
            gan_output_file,
            gan_gold_file,
            bleu_feats.index,
            predicts,
            jd_bleu.redocstring,
        )

        goldMap, predictionMap = bleu.computeMaps(predictions, gan_gold_file)
        dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        logging.info("+ bleu-4 = %f", dev_bleu)

        if args.do_gan_train and dev_bleu > best_bleu:
            logging.info("+ Best bleu !!")
            best_bleu = dev_bleu

            save_model(gen, gan_gen_best_bleu % (epoch, best_bleu))
            save_model(dis, gan_dis_best_bleu % (epoch, best_bleu))


# %%

