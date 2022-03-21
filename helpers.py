import logging
import os
import _pickle as cPickle
import random
from typing import Literal
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
from torch.utils.data import TensorDataset, ConcatDataset
from tqdm import tqdm, trange
from pandas.io.parquet import to_parquet
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SubsetRandomSampler,
    TensorDataset,
    Subset,
    ConcatDataset,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
import bleu
from dataset import CombineDataset, ConstDataset, SamplingSubset, SelectDataset
from meter import MinMeter, MaxMeter, BatchAvgMeter
import tokenizer
from tokenizer import str_to_ids, tensors_to_text


# writer = SummaryWriter()


def to_features(df):
    def make_features(s):
        code_tokens, code_ids = str_to_ids(s.recode)

        doc_tokens, doc_ids = str_to_ids(s.redocstring)

        return {
            "code_tokens": code_tokens,
            "doc_tokens": doc_tokens,
            "code_ids": code_ids,
            "doc_ids": doc_ids,
        }

    return df.apply(make_features, axis=1, result_type="expand")


def padding(ids, padding_to):
    ids = [tokenizer.bos_token_id] + \
        list(ids[: padding_to - 2]) + [tokenizer.eos_token_id]
    padding_length = padding_to - len(ids)
    mask = [1] * len(ids) + [0] * padding_length
    ids += [tokenizer.pad_token_id] * padding_length
    return ids, mask


def to_pad_features(df, source_length, target_length):
    def pad_features(s):
        code_ids, code_mask = padding(s.code_ids, source_length)
        doc_ids, doc_mask = padding(s.doc_ids, target_length)

        return {
            "code_ids": code_ids,
            "code_mask": code_mask,
            "doc_ids": doc_ids,
            "doc_mask": doc_mask,
        }

    return df.apply(pad_features, axis=1, result_type="expand")


def sample_dataset(df):
    return df.sample(n=min(1000, len(df)))


def series_to_tensor(col):
    return torch.tensor(np.array(col.to_list()))


def cache_call(path, func, format="parquet"):
    def pickle_dump(obj, path):
        with open(path, "wb") as f:
            cPickle.dump(obj, f)

    def pickle_load(path):
        with open(path, "rb") as f:
            return cPickle.load(f)

    FORMAT = {
        "parquet": (".parquet", to_parquet, pd.read_parquet),
        "numpy": (".npy", lambda obj, path: np.save(path, obj), np.load),
        "torch": (".bin", torch.save, torch.load),
        "pickle": (".pkl", pickle_dump, pickle_load),
    }

    postfix, saver, loader = FORMAT[format]

    path += postfix

    def new_func(*args, **kwargs):
        if not os.path.exists(path):
            res = func(*args, **kwargs)
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            saver(res, path)
            return res
        return loader(path)

    return new_func


def remember_result(func):
    def new_func():
        scope = func
        if hasattr(scope, "res"):
            return scope.res
        scope.res = res = func()
        return res
    return new_func


def cache_result(path: str, format="parquet"):
    def pickle_dump(obj, path):
        with open(path, "wb") as f:
            cPickle.dump(obj, f)

    def pickle_load(path):
        with open(path, "rb") as f:
            return cPickle.load(f)

    FORMAT = {
        "parquet": (".parquet", to_parquet, pd.read_parquet),
        "numpy": (".npy", lambda obj, path: np.save(path, obj), np.load),
        "torch": (".bin", torch.save, torch.load),
        "pickle": (".pkl", pickle_dump, pickle_load),
    }

    postfix, saver, loader = FORMAT[format]

    path = str(path)
    path += postfix

    def decorate(func):
        def new_func(*args, **kwargs):
            if os.path.exists(path):
                return loader(path)
            res = func(*args, **kwargs)
            try:
                if os.path.dirname(path):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                saver(res, path)
            except Exception as ex:
                print("cache_result failed:", ex)
            return res
        return new_func
    return decorate


def save_model(model, path):
    if is_distributed() and not is_master():
        return

    # Only save the model it-self
    model_to_save = model.module if hasattr(model, "module") else model
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    logging.info(f"saving model to {path}")
    torch.save(model_to_save.state_dict(), path)


def write_output(output_file, gold_file, indexs, predicts, golds):
    predictions = []

    with open(output_file, "w") as f, open(gold_file, "w") as fgold:
        for idx, pred, gold in zip(indexs, predicts, golds):
            predictions.append(str(idx) + "\t" + pred)
            f.write(str(idx) + "\t" + pred + "\n")
            fgold.write(str(idx) + "\t" + gold + "\n")
    return predictions


def env_get(key: str) -> str:
    return os.environ.get(key)


def is_distributed() -> bool:
    return os.environ.get("LOCAL_RANK") != None


def local_rank() -> int:
    return int(env_get("LOCAL_RANK"))


def rank() -> int:
    return int(env_get("RANK"))


def world_size() -> int:
    return int(env_get("WORLD_SIZE"))


def is_master() -> bool:
    return os.environ.get("LOCAL_RANK") == "0"


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def make_gan_dataset(_gen, dataloader) -> TensorDataset:
    all_source_ids = []
    all_target_ids = []
    all_labels = []

    device = _gen.device

    _gen.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, "Making fake dataset"):
            source_ids = batch[0]
            source_mask = batch[1]
            target_ids = batch[2]

            batch_size = source_ids.size(0)

            all_source_ids.append(source_ids)
            all_target_ids.append(target_ids)
            all_labels.append(torch.ones(batch_size, dtype=torch.float))

            g_target_ids = _gen.beam_predict(
                source_ids.to(device), source_mask.to(device), "cpu"
            )

            all_source_ids.append(source_ids)
            all_target_ids.append(g_target_ids)
            all_labels.append(torch.zeros(batch_size, dtype=torch.float))

    all_source_ids = torch.cat(all_source_ids)
    all_target_ids = torch.cat(all_target_ids)
    all_labels = torch.cat(all_labels)

    return TensorDataset(all_source_ids, all_target_ids, all_labels)


def mix_dataset(real_dataset, fake_dataset, keep_col=[0, 1, 2]):
    """
    Combine two dataset.

    Input:
        real_dataset: (source_ids, source_mask, target_ids, target_mask)
        fake_dataset: (source_ids, source_mask, target_ids)
    Output:
        mixed_dataset: (source_ids, source_mask, target_ids, label)
    """
    real_dataset = CombineDataset(
        SelectDataset(real_dataset, keep_col),
        ConstDataset(torch.tensor(1.0), len(real_dataset))
    )
    fake_dataset = CombineDataset(
        SelectDataset(fake_dataset, keep_col),
        ConstDataset(torch.tensor(0.0), len(fake_dataset))
    )

    return ConcatDataset([real_dataset, fake_dataset])


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta

    def __call__(self, val_loss):
        """ return True if should stop training """

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def fakegen(args, dataset, num_samples, _gen, train=True):
    '''when in distributed, every process will generate and return a part of the mixed data'''
    if train:
        if is_distributed():
            indices = torch.randperm(len(dataset))[:num_samples]
            subset = Subset(dataset, indices)
            sampler = DistributedSampler(subset, shuffle=True)
        else:
            indices = torch.randperm(len(dataset))[:num_samples]
            sampler = SubsetRandomSampler(indices)
    else:
        indices = torch.randperm(len(dataset))[:num_samples]
        dataset = Subset(dataset, indices)
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.fakegen_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    logging.info("+ make dataset for gan %s: batch_size = %d, num_samples = %d, total samples ~= %d",
                 "train" if train else "eval", args.fakegen_batch_size, num_samples,
                 args.fakegen_batch_size * len(dataloader))

    return make_gan_dataset(_gen, dataloader)


def eval_gen_bleu(args, _gen, bleu_dataset, gan_output_file, gan_gold_file, indices, gold):
    gen_device = _gen.device
    dataloader = DataLoader(
        bleu_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    predicts = []
    for batch in tqdm(dataloader, "bleu"):
        source_ids = batch[0]
        source_mask = batch[1]
        with torch.no_grad():
            preds = _gen.beam_predict(source_ids.to(gen_device),
                                      source_mask.to(gen_device))
            predicts += tensors_to_text(preds)

    predictions = write_output(
        gan_output_file,
        gan_gold_file,
        indices,
        predicts,
        gold,
    )

    goldMap, predictionMap = bleu.computeMaps(predictions, gan_gold_file)
    return bleu.bleuFromMaps(goldMap, predictionMap)[0]


def eval_gen_loss(args, _gen, gen, valid_dataset):
    gen_device = _gen.device
    gen_valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    avg_meter = BatchAvgMeter()
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
                loss = loss.sum()

        eval_loss = avg_meter.update(loss.item(), num.item())

    return eval_loss


def eval_dis_loss(args, dis, dataset):
    dis_device = dis.device
    loss_meter = BatchAvgMeter()
    dataloader = DataLoader(
        dataset,
        batch_size=args.dis_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    with torch.no_grad():
        with tqdm(dataloader, "BCEloss 00.0000") as bar:
            for batch in bar:
                source_ids = batch[0].to(dis_device)
                target_ids = batch[1].to(dis_device)
                labels = batch[2].to(dis_device)

                prob = dis(source_ids, target_ids)
                loss = F.binary_cross_entropy(prob, labels, reduction='sum')
                # mean() to average on multi-gpu
                if loss.size():
                    loss = loss.mean()

                avg_loss = loss_meter.update(loss.item(), labels.size(0))
                bar.set_description(f"BCEloss {avg_loss:.4f}")

    return avg_loss


def eval_dis_acc(args, dis, dataset):
    dis_device = dis.device
    acc_meter = BatchAvgMeter()
    dataloader = DataLoader(
        dataset,
        batch_size=args.dis_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    with torch.no_grad():
        with tqdm(dataloader, "Accu 00.0000") as bar:
            for batch in bar:
                source_ids = batch[0].to(dis_device)
                target_ids = batch[1].to(dis_device)
                labels = batch[2]

                prob = dis(source_ids, target_ids).to("cpu").gt(0.5)

                right = labels.eq(prob).sum()
                total = source_ids.size(0)

                acc = acc_meter.update(right, total)
                bar.set_description(f"Accu {acc:.4f}")
    return acc


class GenTrainer():
    def __init__(self, args, _gen, gen, bleu_index, bleu_gold):
        self.args = args
        self._gen = _gen
        self.gen = gen
        self.bleu_index = bleu_index
        self.bleu_gold = bleu_gold

        self.gen_device = _gen.device

        self.prepare_optimizer()
        self.prepare_checkpoints()

        self.best_loss = MinMeter()
        self.best_bleu = MaxMeter()

    def gen_train(self):
        self.gen.train()

    def gen_eval(self):
        self.gen.eval()

    def to_gen_device(self, x):
        return x.to(self.gen_device)

    def prepare_optimizer(self):
        # TODO: Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.gen.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.gen.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.g_opt = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.gen_learning_rate,
            eps=self.args.gen_adam_epsilon,
        )

    def prepare_checkpoints(self):
        from os.path import join as path_join

        self.checkpoints = {
            'latest': path_join(self.args.output_dir, "gen.bin"),
            'best_loss': path_join(self.args.output_dir, "gen_bestloss_%d_%f.bin"),
            'best_bleu': path_join(self.args.output_dir, "gen_bestbleu_%d_%f.bin"),
            'output_file': path_join(self.args.output_dir, "gen.output"),
            'gold_file': path_join(self.args.output_dir, "gen.gold"),
        }

        logging.info("checkpoints: %s", self.checkpoints)

    def save_checkpoint(self, type: Literal['latest|best_loss|best_bleu'], *args):
        path = self.checkpoints[type]
        if args:
            path = path % args
        save_model(self._gen, path)

    def prepare_scheduler(self, train_dataset):
        t_total = len(train_dataset) / self.args.gen_batch_size * \
            self.args.gen_train_epochs

        logging.info("+ Total train steps = %d", t_total)
        logging.info("+ Warmup steps = %d", int(t_total * 0.1))

        self.g_sch = get_linear_schedule_with_warmup(
            self.g_opt, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
        )

    def train_epoch(self, train_dataset):
        self.prepare_scheduler(train_dataset)

        if is_distributed():
            gen_train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            gen_train_sampler = RandomSampler(train_dataset)

        gen_train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.gen_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=gen_train_sampler,
        )

        avg_meter = BatchAvgMeter()
        self.gen_train()
        with tqdm(gen_train_dataloader) as train_bar:
            for batch in train_bar:
                source_ids = self.to_gen_device(batch[0])
                source_mask = self.to_gen_device(batch[1])
                target_ids = self.to_gen_device(batch[2])
                target_mask = self.to_gen_device(batch[3])

                context, memory_key_padding_mask = self._gen.get_context(
                    source_ids, source_mask)
                # [batch_size x source_length x args.hidden_size]

                loss, num, _ = self.gen(context, memory_key_padding_mask,
                                        target_ids, target_mask)
                if loss.size():
                    loss = loss.sum()  # mean() to average on multi-gpu.
                    num = num.sum()
                loss.backward()

                # Update parameters
                self.g_opt.step()
                self.g_opt.zero_grad()
                self.g_sch.step()

                avg_loss = avg_meter.update(loss.item(), num.item())
                train_bar.set_description(f"loss {avg_loss:.2f}")

        self.save_checkpoint('latest', self.epoch)

    def train(self, train_dataset, valid_dataset, bleu_dataset):
        # Start training
        logging.info("Do train generator:")
        logging.info("+ train_dataset = %d", len(train_dataset))
        logging.info("+ Batch size = %d", self.args.gen_batch_size)
        logging.info("+ Train epochs = %d", self.args.gen_train_epochs)
        logging.info("+ Learning rate = %e", self.args.gen_learning_rate)
        logging.info("+ Adam epsilon = %e", self.args.gen_adam_epsilon)

        for self.epoch in range(self.args.gen_train_epochs):
            self.train_epoch(train_dataset)
            self.eval_epoch(valid_dataset, bleu_dataset)

    def eval(self, valid_dataset, bleu_dataset):
        loss = self.eval_loss(valid_dataset)
        logging.info(f"+ Eval loss: {loss:.5f}")
        dev_bleu = self.eval_bleu(bleu_dataset)
        logging.info("+ bleu-4 = %f", dev_bleu)

    def eval_epoch(self, valid_dataset, bleu_dataset):
        loss = self.eval_loss(valid_dataset)
        logging.info(f"+ Eval loss: {loss:.5f}")
        if self.best_loss.update(loss) == loss:
            self.save_checkpoint('best_loss', self.epoch, loss)

        dev_bleu = self.eval_bleu(bleu_dataset)
        logging.info("+ bleu-4 = %f", dev_bleu)
        if self.best_bleu.update(dev_bleu) == dev_bleu:
            self.save_checkpoint('best_bleu', self.epoch, dev_bleu)

    def eval_loss(self, valid_dataset):
        self.gen_eval()
        return eval_gen_loss(self.args, self._gen, self.gen, valid_dataset)

    def eval_bleu(self, bleu_dataset):
        return eval_gen_bleu(
            self.args,
            self._gen,
            bleu_dataset,
            self.checkpoints["bleu_output"],
            self.checkpoints["bleu_gold"],
            self.bleu_index,
            self.bleu_gold,
        )


class DisTrainer():
    def __init__(self, args, _dis, dis, _gen):
        self.args = args
        self._dis = _dis
        self.dis = dis
        self._gen = _gen
        self.dis_device = _dis.device

        from os.path import join as path_join
        self.checkpoints = {
            "dis_last_path": path_join(args.output_dir, "dis.bin"),
            "dis_best_path": path_join(args.output_dir, "dis_bestloss_%d_%f.bin"),
            "dis_best_acc": path_join(args.output_dir, "dis_bestacc_%d_%f.bin"),
        }
        logging.info("checkpoint paths: %s", self.checkpoints)

        self.__prepare_optimizer()

        self.best_loss = MinMeter()
        self.best_acc = MaxMeter()

    def save_model(self, type: Literal['latest|best_acc|best_loss'], loss=0):
        if type == 'latest':
            path = self.checkpoints['dis_last_path']
        elif type == 'best_acc':
            path = self.checkpoints['dis_best_acc'] % (self.epoch, loss)
        else:
            path = self.checkpoints['dis_best_path'] % (self.epoch, loss)

        save_model(self._dis, path)

    def to_dis_device(self, x):
        return x.to(self.dis_device)

    def dis_train(self):
        self.dis.train()

    def dis_eval(self):
        self.dis.eval()

    def __prepare_optimizer(self):
        logging.info("+ Learning rate = %s", self.args.dis_learning_rate)
        logging.info("+ Adam epsilon = %e", self.args.dis_adam_epsilon)
        self.d_opt = optim.Adam(
            self.dis.parameters(), lr=self.args.dis_learning_rate, eps=self.args.dis_adam_epsilon
        )

    def train_epoch(self, dis_train_dataset):
        loss_meter = BatchAvgMeter()
        dataloader = DataLoader(
            dis_train_dataset,
            batch_size=self.args.dis_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        logging.info("+ train batch size = %d", self.args.dis_batch_size)
        self.dis_train()
        with tqdm(dataloader, "BCEloss 00.0000") as bar:
            for batch in bar:
                source_ids = self.to_dis_device(batch[0])
                target_ids = self.to_dis_device(batch[1])
                labels = self.to_dis_device(batch[2])

                prob = self.dis(source_ids, target_ids)
                loss = F.binary_cross_entropy(prob, labels, reduction='sum')
                # mean() to average on multi-gpu
                if loss.size():
                    loss = loss.mean()

                loss.backward()
                self.d_opt.step()
                self.d_opt.zero_grad()

                avg_loss = loss_meter.update(loss.item(), labels.size(0))
                bar.set_description(f"BCEloss {avg_loss:.4f}")

    def train(self, train_dataset, valid_dataset):
        logging.info("Do discriminator train:")
        logging.info("+ train_dataset = %d", len(train_dataset))
        logging.info("+ valid_dataset = %d", len(valid_dataset))

        dis_valid_dataset = fakegen(
            self.args, valid_dataset, self.args.dis_valid_sample, self._gen)

        for self.epoch in range(self.args.dis_train_epochs):
            # change dataset per X epochs
            if self.epoch % 10 == 0:
                dis_train_dataset = fakegen(
                    self.args,
                    train_dataset,
                    self.args.dis_train_sample,
                    self._gen,
                    True
                )

            self.train_epoch(dis_train_dataset)
            self.save_model("latest")

            loss = self.eval_loss(dis_valid_dataset)
            if self.best_loss.update(loss) == loss:
                logging.info("+ Best loss !!")
                self.save_model('best_loss', loss)

            acc = self.eval_acc(dis_valid_dataset)
            if self.best_acc.update(acc) == acc:
                logging.info("+ Best acc !!")
                self.save_model('best_acc', acc)

    def eval_loss(self, dataset):
        self.dis_eval()
        avg_loss = eval_dis_loss(self.args, self.dis, dataset)
        logging.info("+ BCEloss = %f", avg_loss)
        return avg_loss

    def eval_acc(self, dataset):
        self.dis_eval()
        acc = eval_dis_acc(self.args, self.dis, dataset)
        logging.info("+ Accu = %f", acc)
        return acc

    def eval(self, valid_dataset):
        dis_valid_dataset = fakegen(
            self.args,
            valid_dataset,
            self.args.dis_valid_sample,
            self._gen,
            train=False,
        )
        self.eval_loss(dis_valid_dataset)
        self.eval_acc(dis_valid_dataset)


class GanTrainer():
    def __init__(self, args, _gen, gen, _dis, dis, bleu_index, bleu_gold):
        self.args = args
        self._gen = _gen
        self.gen = gen
        self._dis = _dis
        self.dis = dis
        self.rollout = Rollout(gen, dis, args.target_length)

        self.gen_device = _gen.device
        self.dis_device = _dis.device

        self.bleu_index = bleu_index
        self.bleu_gold = bleu_gold

        for p in self._dis.src_embedding.parameters():
            p.requires_grad = False
        for p in self._dis.tgt_embedding.parameters():
            p.requires_grad = False

        self.__prepare_path()
        self.__prepare_eval()
        self.__prepare_optimizer_gen()
        self.__prepare_optimizer_dis()

        self.epoch = -1

    def to_gen_device(self, x):
        return x.to(self.gen_device)

    def to_dis_device(self, x):
        return x.to(self.dis_device)

    def gen_to_cpu(self):
        self.gen.to("cpu")

    def gen_to_gpu(self):
        self.gen.to(self.gen_device)

    def dis_to_cpu(self):
        self.dis.to("cpu")

    def dis_to_gpu(self):
        self.dis.to(self.dis_device)

    def gen_train(self):
        self.gen.train()

    def gen_eval(self):
        self.gen.eval()

    def dis_train(self):
        self.dis.train()

    def dis_eval(self):
        self.dis.eval()

    def save_models(self, type: Literal['latest|best_loss|best_bleu'], val=0):
        if type == "latest":
            gen_path = self.checkpoints['latest_gen']
            dis_path = self.checkpoints['latest_dis']
        elif type == "best_loss":
            gen_path = self.checkpoints['best_loss_gen'] % (self.epoch, val)
            dis_path = self.checkpoints['best_loss_dis'] % (self.epoch, val)
        else:
            gen_path = self.checkpoints['best_bleu_gen'] % (self.epoch, val)
            dis_path = self.checkpoints['best_bleu_dis'] % (self.epoch, val)

        save_model(self._gen, gen_path)
        save_model(self._dis, dis_path)

    def __prepare_path(self):
        from os.path import join as path_join

        self.checkpoints = {
            "latest_gen": path_join(self.args.output_dir, "gan_gen.bin"),
            "latest_dis": path_join(self.args.output_dir, "gan_dis.bin"),
            "best_loss_gen": path_join(self.args.output_dir, "gan_gen_bestloss_%d_%f.bin"),
            "best_loss_dis": path_join(self.args.output_dir, "gan_dis_bestloss_%d_%f.bin"),
            "best_bleu_gen": path_join(self.args.output_dir, "gan_gen_bestbleu_%d_%f.bin"),
            "best_bleu_dis": path_join(self.args.output_dir, "gan_dis_bestbleu_%d_%f.bin"),
            "bleu_output": path_join(self.args.output_dir, "gan_dev_%d.output"),
            "bleu_gold": path_join(self.args.output_dir, "gan_dev_%d.gold")
        }

        logging.info("Checkpoint paths: %s", self.checkpoints)

    def __prepare_optimizer_gen(self):
        logging.info("+ Generator learning rate = %s", self.args.gen_learning_rate)
        logging.info("+ Generator adam epsilon = %e", self.args.gen_adam_epsilon)
        # TODO: Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.gen.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.gen.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.g_opt = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.gen_learning_rate,
            eps=self.args.gen_adam_epsilon,
        )

        t_total = self.args.gan_g_steps * self.args.gan_train_epochs
        if self.args.gan_teach:
            t_total = t_total * 2

        logging.info("+ Total train steps = %d", t_total)
        logging.info("+ Warmup steps = %d", int(t_total * 0.1))

        self.g_sch = get_linear_schedule_with_warmup(
            self.g_opt, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total)

    def __prepare_optimizer_dis(self):
        logging.info("+ Discriminator learning rate = %s", self.args.dis_learning_rate)
        logging.info("+ Discriminator adam epsilon = %e", self.args.dis_adam_epsilon)
        self.d_opt = optim.Adam(
            filter(lambda p: p.requires_grad, self.dis.parameters()),
            lr=self.args.dis_learning_rate,
            eps=self.args.dis_adam_epsilon
        )

    def train_step_gen(self, batch):
        source_ids = self.to_gen_device(batch[0])
        source_mask = self.to_gen_device(batch[1])

        """672 MiB"""
        context, memory_key_padding_mask = self._gen.get_context(source_ids, source_mask)
        # [batch_size x source_length x args.hidden_size]

        pre_target_ids, pre_target_mask = self.gen(context, memory_key_padding_mask)
        rewards = self.rollout.get_reward(
            source_ids,
            context,
            memory_key_padding_mask,
            pre_target_ids,
            pre_target_mask,
            rollnum=self.args.gan_rollnum,
        )
        # get pg_loss
        loss = self.gen(context, memory_key_padding_mask, pre_target_ids, rewards=rewards)

        if loss.size():
            loss = loss.mean()

        loss.backward()
        self.g_opt.step()
        self.g_opt.zero_grad()
        self.g_sch.step()

        return loss.item(), source_ids.size(0)

    def train_step_gen_tf(self, batch):
        source_ids = self.to_gen_device(batch[0])
        source_mask = self.to_gen_device(batch[1])
        target_ids = self.to_gen_device(batch[2])
        target_mask = self.to_gen_device(batch[3])

        context, memory_key_padding_mask = self._gen.get_context(source_ids, source_mask)
        # [batch_size x source_length x args.hidden_size]

        rewards = torch.ones_like(target_ids) * target_mask
        tloss = self.gen(context, memory_key_padding_mask, target_ids, rewards=rewards)

        if tloss.size():
            tloss = tloss.mean()

        tloss.backward()
        self.g_opt.step()
        self.g_opt.zero_grad()
        self.g_sch.step()

        return tloss.item(), source_ids.size(0)

    def train_step_dis(self, dis_train_dataset):
        dataloader = DataLoader(
            dis_train_dataset,
            batch_size=self.args.gan_d_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        # Train dis for k epochs
        for d_epoch in trange(self.args.gan_d_epochs, desc="d-step epoch"):
            loss_meter = BatchAvgMeter()
            with tqdm(dataloader, "BCEloss 0.0000") as bar:
                for batch in bar:
                    source_ids = self.to_dis_device(batch[0])
                    target_ids = self.to_dis_device(batch[1])
                    labels = self.to_dis_device(batch[2])

                    prob = self.dis(source_ids, target_ids)
                    loss = F.binary_cross_entropy(prob, labels, reduction='sum')
                    # mean() to average on multi-gpu
                    if loss.size():
                        loss = loss.mean()

                    loss.backward()
                    self.d_opt.step()
                    self.d_opt.zero_grad()

                    d_loss_avg = loss_meter.update(loss.item(), labels.size(0))
                    bar.set_description(f"BCEloss {d_loss_avg:.4f}")

            logging.info("d-epoch train BCEloss %f", d_loss_avg)

    def train_epoch_gen(self, train_iter):
        # G-steps
        self.gen_to_gpu()
        self.gen_train()
        self.dis_to_gpu()
        self.dis_eval()
        g_loss = BatchAvgMeter()
        g_tloss = BatchAvgMeter()
        with trange(self.args.gan_g_steps, desc="g-step 00.0000 00.0000") as g_step_bar:
            g_step = iter(g_step_bar)
            while True:
                try:
                    next(g_step)
                except StopIteration:
                    break

                batch = next(train_iter)
                batch_loss, batch_samples = self.train_step_gen(batch)
                g_loss_avg = g_loss.update(batch_loss, batch_samples)

                if self.args.gan_teach:
                    tloss, tsamples = self.train_step_gen_tf(batch)
                    g_tloss_avg = g_tloss.update(tloss, tsamples)

                    g_step_bar.set_description(f"g-step {g_loss_avg:.4f} {g_tloss_avg:.4f}")
                else:
                    g_step_bar.set_description(f"g-step {g_loss_avg:.4f} -")

        logging.info("g-step train avg loss (gan only): %f", g_loss_avg)
        if self.args.gan_teach:
            logging.info("g-step train avg loss (teach only): %f", g_tloss_avg)
            logging.info("g-step train avg loss: %f", (g_loss_avg + g_tloss_avg) / 2)
            # writer.add_scalar("GAN/Gen/Loss", (g_loss_avg + g_tloss_avg) / 2, self.epoch)
            pass
        else:
            # writer.add_scalar("GAN/Gen/Loss", g_loss_avg, self.epoch)
            pass

    def sample_for_dis(self, train_subset):
        '''
        Notice: The order of the return dataset is equal to the train_subset,
        as shuffle is used in train_step_dis(), so not shuffle here.
        '''
        if is_distributed():
            sampler = DistributedSampler(train_subset)
        else:
            sampler = SequentialSampler(train_subset)

        dataloader = DataLoader(
            train_subset,
            batch_size=self.args.fakegen_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
        )

        logging.info("+ Gen fake: batch_size = %d, samples ~= %d", self.args.fakegen_batch_size,
                     self.args.fakegen_batch_size * len(dataloader))
        return make_gan_dataset(self._gen, dataloader)

    def train_epoch_dis(self, train_subset):
        # D-steps
        self.gen_eval()
        self.dis_train()
        for d_step in trange(self.args.gan_d_steps, desc="d-step"):
            # (re)generate fake dataset
            self.gen_to_gpu()
            dis_train_dataset = self.sample_for_dis(train_subset)
            self.gen_to_cpu()

            self.dis_to_gpu()
            self.train_step_dis(dis_train_dataset)
            self.dis_to_cpu()

    def train(self, train_dataset, valid_dataset, bleu_dataset):
        logging.info("Do GAN train:")

        logging.info("+ train dataset = %d", len(train_dataset))
        logging.info("+ valid dataset = %d", len(valid_dataset))
        logging.info("+ bleu dataset = %d", len(bleu_dataset))

        logging.info("+ g-steps = %d", self.args.gan_g_steps)
        logging.info("+ Teacher forcing = %s", self.args.gan_teach)
        logging.info("+ d-steps = %d", self.args.gan_d_steps)
        logging.info("+ d-sample = %d", self.args.gan_d_sample)
        logging.info("+ d-epochs = %d", self.args.gan_d_epochs)
        logging.info("+ Rollout num = %d", self.args.gan_rollnum)
        logging.info("+ GAN batch size = %d", self.args.gan_batch_size)

        '''
        In theory, the whole dataset should be used to train the discriminator,
        but that's very slow. So a subset is used. The subset won't change
        with d-step.
        '''
        train_subset = SamplingSubset(train_dataset, self.args.gan_d_sample)

        '''
        Change data for pg train every epoch.
        '''
        if is_distributed():
            gan_train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            gan_train_sampler = RandomSampler(train_dataset)

        gan_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.gan_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=gan_train_sampler,
        )
        train_iter = iter(gan_dataloader)

        for self.epoch in trange(self.args.gan_train_epochs, desc="Epoch"):
            # Not need, because one epoch here doesn't exhaust all samples
            # if is_distributed():
            #     gan_train_sampler.set_epoch(self.epoch)
            self.train_epoch_gen(train_iter)
            self.train_epoch_dis(train_subset)
            self.save_models('latest')
            self.eval_epoch(valid_dataset, bleu_dataset)

    def __prepare_eval(self):
        self.best_loss = MinMeter()
        self.best_bleu = MaxMeter()

    def eval(self, valid_dataset, bleu_dataset):
        self.dis_to_cpu()
        self.gen_eval()
        self.gen_to_gpu()
        self.eval_gen_loss(valid_dataset)
        self.eval_gen_bleu(bleu_dataset)
        self.eval_dis_acc(valid_dataset)

    def eval_dis_acc(self, valid_dataset):
        dis_valid_dataset = fakegen(
            self.args,
            valid_dataset,
            self.args.dis_valid_sample,
            self._gen,
            train=False,
        )

        acc_meter = BatchAvgMeter()
        dataloader = DataLoader(
            dis_valid_dataset,
            batch_size=self.args.dis_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.dis_eval()
        self.dis_to_gpu()
        with torch.no_grad():
            with tqdm(dataloader, "Accu 00.0000") as bar:
                for batch in bar:
                    source_ids = self.to_dis_device(batch[0])
                    target_ids = self.to_dis_device(batch[1])
                    labels = batch[2]
                    prob = self.dis(source_ids, target_ids).to("cpu").gt(0.5)
                    right = labels.eq(prob).sum()
                    total = source_ids.size(0)
                    acc = acc_meter.update(right, total)
                    bar.set_description(f"Accu {acc:.4f}")

        logging.info("+ eval acc = %f", acc)
        return acc

    def eval_epoch(self, valid_dataset, bleu_dataset):
        self.dis_to_cpu()
        self.gen_eval()
        self.gen_to_gpu()

        loss = self.eval_gen_loss(valid_dataset)
        if self.best_loss.update(loss) == loss:
            logging.info("+ Best loss !!")
            self.save_models('best_loss', self.best_loss.get())

        dev_bleu = self.eval_gen_bleu(bleu_dataset)
        if self.best_bleu.update(dev_bleu) == dev_bleu:
            logging.info("+ Best bleu !!")
            self.save_models('best_bleu', self.best_bleu.get())

        self.eval_dis_acc(valid_dataset)

    def eval_gen_loss(self, valid_dataset):
        # Eval G with dev dataset
        logging.info("+ Valid dataset = %d", len(valid_dataset))
        logging.info("+ batch size = %d", self.args.eval_batch_size)
        logging.info("+ num workers = %d", self.args.num_workers)

        loss = self.get_loss(valid_dataset)
        logging.info("+ eval loss = %f", loss)
        return loss

    def eval_gen_bleu(self, bleu_dataset):
        logging.info("+ Bleu dataset = %d", len(bleu_dataset))
        logging.info("+ batch size = %d", self.args.eval_batch_size)
        logging.info("+ num workers = %d", self.args.num_workers)

        dev_bleu = self.get_bleu(
            bleu_dataset,
            self.checkpoints["bleu_output"] % (self.epoch, ),
            self.checkpoints["bleu_gold"] % (self.epoch, ),
            self.bleu_index,
            self.bleu_gold
        )
        logging.info("+ bleu-4 = %f", dev_bleu)
        return dev_bleu

    def get_loss(self, valid_dataset):
        loss_meter = BatchAvgMeter()
        gan_valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        for batch in tqdm(gan_valid_dataloader, "validation"):
            source_ids = batch[0].to(self.gen_device)
            source_mask = batch[1].to(self.gen_device)
            target_ids = batch[2].to(self.gen_device)
            target_mask = batch[3].to(self.gen_device)

            with torch.no_grad():
                context, memory_key_padding_mask = self._gen.get_context(
                    source_ids, source_mask
                )
                # [batch_size x source_length x args.hidden_size]

                loss, num, _ = self.gen(
                    context, memory_key_padding_mask, target_ids, target_mask
                )
                loss *= num
                if loss.size():  # is multi-gpu
                    loss = loss.sum()
                    num = num.sum()
            eval_loss = loss_meter.update(loss.item(), num.item())

        return eval_loss

    def get_bleu(self, bleu_dataset, gan_output_file, gan_gold_file, indices, gold):
        # Save best checkpoint for best bleu
        logging.info("Calculate bleu-4:")
        logging.info("+ gan_output_file = %s", gan_output_file)
        logging.info("+ gan_gold_file = %s", gan_gold_file)

        gan_valid_dataloader_bleu = DataLoader(
            bleu_dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        predicts = []
        for batch in tqdm(gan_valid_dataloader_bleu, "bleu"):
            source_ids = batch[0]
            source_mask = batch[1]
            with torch.no_grad():
                preds = self._gen.beam_predict(
                    source_ids.to(self.gen_device),
                    source_mask.to(self.gen_device))
                predicts += tensors_to_text(preds)

        predictions = write_output(
            gan_output_file,
            gan_gold_file,
            indices,
            predicts,
            gold,
        )

        goldMap, predictionMap = bleu.computeMaps(predictions, gan_gold_file)
        dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        return dev_bleu


class Rollout:
    def __init__(self, gen, dis, max_length):
        self.gen = gen
        self.dis = dis
        self.max_length = max_length
        self.gen_device = gen.device
        self.dis_device = dis.device

    def get_reward(
        self,
        source_ids,
        context,
        memory_key_padding_mask,
        pre_target_ids,
        pre_target_mask,
        rollnum=20,
    ):
        """
        Rollout method:
            give none(<s>), predict, get reward
            give pre_target_ids[:1](<s>,a), predict, get reward
            give pre_target_ids[:2](<s>,a,b), predict, get reward
            ...
            give pre_target_ids[:max_length-1], predict, get reward
        Input:
            pre_target_ids: [batch_size x target_length], with bos!!
        Outputs:
            rewards: [batch_size x max_length]
                rewards[i][0] is empty
                rewards[i][j]: the reward of using word Seq[j] as the next of Seq[0..j-1].
        """
        batch_size = context.size(0)

        rewards = torch.zeros(
            self.max_length, batch_size, dtype=torch.float, device=self.dis_device
        )
        self.dis.eval()
        for _ in range(rollnum):  # rollout times, mean() later
            # ignore bos_token
            for init_given_num in range(2, self.max_length):
                if not any(pre_target_mask[:, init_given_num]):
                    break
                target_ids, target_mask = self.gen(
                    context,
                    memory_key_padding_mask,
                    target_ids=pre_target_ids,
                    rollout=True,
                    init_given_num=init_given_num,
                )
                with torch.no_grad():
                    # pred: [0-1] prods
                    pred = self.dis(
                        source_ids.to(self.dis_device),
                        target_ids.to(self.dis_device)
                    )
                # pred = pred.cpu()
                rewards[init_given_num - 1] += pred

            with torch.no_grad():
                pred = self.dis(
                    source_ids.to(self.dis_device),
                    pre_target_ids.to(self.dis_device)
                )
            # pred = pred.cpu()
            # [batch_size]
            rewards[self.max_length - 1] += pred

        # rewards: [batch_size x max_length]
        rewards = rewards.permute([1, 0]).contiguous()
        rewards = rewards * pre_target_mask
        rewards = rewards / (1.0 * rollnum)
        return rewards.to(pre_target_mask.device)
