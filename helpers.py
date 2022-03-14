import logging
import os
import tokenize
import _pickle as cPickle
import random
from typing import List, Literal
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.functional import Tensor
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from tqdm import tqdm, trange
from pandas.io.parquet import to_parquet
from typing import List, Sequence
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    Subset,
    SubsetRandomSampler,
    ConcatDataset,
)
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import bleu
from dataset import CombineDataset, ConstDataset, SelectDataset
from meter import MinMeter, MaxMeter, AvgMeter, BatchAvgMeter
from model import Discriminator, Rollout
import tokenizer
from tokenizer import str_to_ids, tensors_to_text


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


def get_target_mask(target_ids: Tensor):
    """
    bos, token, eos -> 1
    pad -> 0
    """
    mask = torch.ones_like(target_ids, dtype=torch.int64)

    for i, batch in enumerate(target_ids):
        try:
            index_of_eos = (batch == tokenizer.eos_token_id).nonzero()[0]
        except IndexError:
            continue
        mask[i][index_of_eos + 1:] = 0

    target_ids[(1 - mask).bool()] = tokenizer.pad_token_id
    mask = mask.to(target_ids.device)
    return target_ids, mask


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
    if hasattr(os.environ, "LOCAL_RANK") and os.environ["LOCAL_RANK"] != 0:
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


def is_distributed() -> bool:
    return hasattr(os.environ, "LOCAL_RANK")


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def make_fake_dataset2(_gen, dataloader):
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


def fakegen(args, dataset, _gen):
    '''when in distributed, every process will generate and return a part of the mixed data'''
    if is_distributed():
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

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

    def save_models(self, type: Literal['latest|best_loss|best_bleu'], epoch=0, val=0):
        if type == "latest":
            path = self.checkpoints['latest']
        elif type == "best_loss":
            path = self.checkpoints['best_loss'] % (epoch, val)
        else:
            path = self.checkpoints['best_bleu'] % (epoch, val)

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

        if not is_distributed():
            gen_train_sampler = RandomSampler(train_dataset)
        else:
            gen_train_sampler = DistributedSampler(train_dataset)

        gen_train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.gen_batch_size,
            num_workers=self.args.gen_num_workers,
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

        self.save_models('latest')

    def train(self, train_dataset, valid_dataset, bleu_dataset):
        # Start training
        logging.info("Do train generator:")
        logging.info("+ Num examples = %d", len(train_dataset))
        logging.info("+ Batch size = %d", self.args.gen_batch_size)
        logging.info("+ Train epochs = %d", self.args.gen_train_epochs)
        logging.info("+ Learning rate = %e", self.args.gen_learning_rate)
        logging.info("+ Adam epsilon = %e", self.args.gen_adam_epsilon)
        logging.info("+ Distributed workers = %d", self.args.gen_num_workers)

        for epoch in range(self.args.gen_train_epochs):
            self.train_epoch(train_dataset)
            self.eval_epoch(valid_dataset, bleu_dataset, epoch)

    def eval_loss(self, valid_dataset):
        gen_valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.gen_num_workers,
            pin_memory=True,
        )

        # Start Evaling model
        # Save best checkpoint for best ppl
        self.gen_eval()
        avg_meter = BatchAvgMeter()
        for batch in tqdm(gen_valid_dataloader, "eval"):
            source_ids = self.to_gen_device(batch[0])
            source_mask = self.to_gen_device(batch[1])
            target_ids = self.to_gen_device(batch[2])
            target_mask = self.to_gen_device(batch[3])

            with torch.no_grad():
                context, memory_key_padding_mask = self._gen.get_context(
                    source_ids, source_mask
                )
                # [batch_size x source_length x args.hidden_size]

                loss, num, _ = self.gen(
                    context, memory_key_padding_mask, target_ids, target_mask
                )
                loss *= num
                if loss.size():
                    num = num.sum()
                    loss = loss.sum()

            eval_loss = avg_meter.update(loss.item(), num.item())

        return eval_loss

    def eval_epoch(self, valid_dataset, bleu_dataset, epoch):
        loss = self.eval_loss(valid_dataset)
        logging.info(f"+ Eval loss: {loss:.5f}")
        if self.best_loss.update(loss) == loss:
            self.save_models('best_loss', epoch, loss)

        dev_bleu = self.eval_bleu(
            bleu_dataset,
            self.checkpoints["bleu_output"],
            self.checkpoints["bleu_gold"],
            self.bleu_index,
            self.bleu_gold
        )
        logging.info("+ bleu-4 = %f", dev_bleu)
        if self.best_bleu.update(dev_bleu) == dev_bleu:
            self.save_models('best_bleu', epoch, dev_bleu)

    def eval_bleu(self, bleu_dataset, gan_output_file, gan_gold_file, indices, gold):
        # Save best checkpoint for best bleu
        logging.info("Calculate bleu-4:")
        logging.info("+ gan_output_file = %s", gan_output_file)
        logging.info("+ gan_gold_file = %s", gan_gold_file)

        dataloader = DataLoader(
            bleu_dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.gan_num_workers,
            pin_memory=True,
        )
        predicts = []
        for batch in tqdm(dataloader, "bleu"):
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


class DisTrainer():
    def __init__(self, args, _dis, dis):
        self.args = args
        self._dis = _dis
        self.dis = dis
        self.dis_device = _dis.device

        from os.path import join as path_join
        self.checkpoints = {
            "dis_last_path": path_join(args.output_dir, "dis.bin"),
            "dis_best_path": path_join(args.output_dir, "dis_bestloss_%d_%f.bin")
        }
        logging.info("checkpoint paths: %s", self.checkpoints)

        self.__prepare_optimizer()

        logging.info("+ dis_batch_size = %s", args.dis_batch_size)

        self.best_loss = MinMeter()

    def save_model(self, type: Literal['latest|best_loss']):
        if type == 'latest':
            path = self.checkpoints['dis_last_path']
        else:
            path = self.checkpoints['dis_best_path']
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

    def prepare_data(self, train_dataset, valid_dataset):
        train_dataset_sub = Subset(
            train_dataset,
            range(self.args.dis_fakegen_train_sample)
        )
        valid_dataset_sub = Subset(
            valid_dataset,
            range(self.args.dis_fakegen_valid_sample)
        )
        fake_train_dataset = fakegen(self.args, train_dataset_sub, self._gen)
        fake_valid_dataset = fakegen(self.args, valid_dataset_sub, self._gen)

        dis_train_dataset = mix_dataset(
            train_dataset_sub, fake_train_dataset, [0, 2])
        dis_valid_dataset = mix_dataset(
            valid_dataset_sub, fake_valid_dataset, [0, 2])

        return dis_train_dataset, dis_valid_dataset

    def train_epoch(self, dataset):

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.dis_batch_size,
            num_workers=self.args.dis_num_workers,
            pin_memory=True,
        )

        self.dis_train()
        with tqdm(dataloader) as bar:
            for batch in bar:
                source_ids = self.to_dis_device(batch[0])
                target_ids = self.to_dis_device(batch[1])
                labels = self.to_dis_device(batch[2])

                pred = self.dis(source_ids, target_ids)
                loss = F.binary_cross_entropy(pred, labels)
                # mean() to average on multi-gpu
                if loss.size():
                    loss = loss.mean()
                bar.set_description(f"loss {loss.item():.2f}")

                loss.backward()
                self.d_opt.step()
                self.d_opt.zero_grad()

    def train(self, train_dataset, valid_dataset):
        logging.info("Do discriminator train:")
        logging.info("+ generate train sample = %d",
                     self.args.dis_fakegen_train_sample)
        logging.info("+ generate valid sample = %d",
                     self.args.dis_fakegen_valid_sample)
        logging.info("+ Train dataset = %d", 2 *
                     self.args.dis_fakegen_train_sample)
        logging.info("+ Valid dataset = %d", 2 *
                     self.args.dis_fakegen_valid_sample)

        dis_train_dataset, dis_valid_dataset = self.prepare_data(
            train_dataset, valid_dataset)

        for epoch in range(self.args.dis_train_epochs):
            self.train_epoch(dis_train_dataset)
            self.save_model("latest")

            loss = self.eval_epoch(dis_valid_dataset)
            if self.best_loss.update(loss) == loss:
                logging.info("+ Best loss !!")
                self.save_model('best_loss', epoch, loss)

    def eval_epoch(self, dataset):
        avg_loss_meter = AvgMeter()
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.dis_batch_size,
            num_workers=self.args.dis_num_workers,
            pin_memory=True,
        )
        self.dis_eval()
        with torch.no_grad():
            with tqdm(dataloader) as bar:
                for batch in bar:
                    source_ids = self.to_dis_device(batch[0])
                    target_ids = self.to_dis_device(batch[1])
                    labels = self.to_dis_device(batch[2])

                    prob = self.dis(source_ids, target_ids)
                    loss = F.binary_cross_entropy(prob, labels)
                    # mean() to average on multi-gpu
                    if loss.size():
                        loss = loss.mean()
                    loss = loss.item()
                    avg_loss = avg_loss_meter.update(loss)
                    bar.set_description(f"dis eval loss {avg_loss:.2f}")

        logging.info("+ eval loss = %f", avg_loss)
        return avg_loss

    def eval(self):
        pass


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

    def save_models(self, type: Literal['latest|best_loss|best_bleu'], epoch=0, val=0):
        if type == "latest":
            gen_path = self.checkpoints['latest_gen']
            dis_path = self.checkpoints['latest_dis']
        elif type == "best_loss":
            gen_path = self.checkpoints['best_loss_gen'] % (epoch, val)
            dis_path = self.checkpoints['best_loss_dis'] % (epoch, val)
        else:
            gen_path = self.checkpoints['best_bleu_gen'] % (epoch, val)
            dis_path = self.checkpoints['best_bleu_dis'] % (epoch, val)

        save_model(self._gen, gen_path)
        save_model(self._dis, dis_path)

    def __prepare_path(self):
        from os.path import join as path_join

        self.checkpoints = {
            "latest_gen": path_join(self.args.output_dir, "gan_gen.bin"),
            "latest_dis": path_join(self.args.output_dir, "gan_dis.bin"),
            "best_loss_gen": path_join(
                self.args.output_dir, "gan_gen_bestloss_%d_%f.bin"),
            "best_loss_dis": path_join(
                self.args.output_dir, "gan_dis_bestloss_%d_%f.bin"),
            "best_bleu_gen": path_join(
                self.args.output_dir, "gan_gen_bestbleu_%d_%f.bin"),
            "best_bleu_dis": path_join(
                self.args.output_dir, "gan_dis_bestbleu_%d_%f.bin"),
            "bleu_output": path_join(self.args.output_dir, "gan_dev_%d.output"),
            "bleu_gold": path_join(self.args.output_dir, "gan_dev_%d.gold")
        }

        logging.info("Checkpoint paths: %s", self.checkpoints)

    def __prepare_optimizer_gen(self):
        logging.info("+ Generator learning rate = %s",
                     self.args.gen_learning_rate)
        logging.info("+ Generator adam epsilon = %e",
                     self.args.gen_adam_epsilon)
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

        t_total = self.args.gan_g_steps * self.args.gan_train_epochs
        if self.args.gan_teach:
            t_total = t_total * 2

        logging.info("+ Total train steps = %d", t_total)
        logging.info("+ Warmup steps = %d", int(t_total * 0.1))

        self.g_sch = get_linear_schedule_with_warmup(
            self.g_opt, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
        )

    def __prepare_optimizer_dis(self):
        logging.info("+ Discriminator learning rate = %s",
                     self.args.dis_learning_rate)
        logging.info("+ Discriminator adam epsilon = %e",
                     self.args.dis_adam_epsilon)
        self.d_opt = optim.Adam(
            filter(lambda p: p.requires_grad, self.dis.parameters()),
            lr=self.args.dis_learning_rate,
            eps=self.args.dis_adam_epsilon
        )

    def train_step_gen(self, batch):
        source_ids = self.to_gen_device(batch[0])
        source_mask = self.to_gen_device(batch[1])

        """672 MiB"""
        context, memory_key_padding_mask = self._gen.get_context(
            source_ids, source_mask)
        # [batch_size x source_length x args.hidden_size]

        pre_target_ids, _ = self.gen(context, memory_key_padding_mask)
        pre_target_ids, pre_target_mask = get_target_mask(pre_target_ids)
        rewards = self.rollout.get_reward(
            source_ids,
            context,
            memory_key_padding_mask,
            pre_target_ids,
            pre_target_mask,
            rollnum=self.args.gan_rollnum,
        )
        # get pg_loss
        loss = self.gen(context, memory_key_padding_mask,
                        pre_target_ids, rewards=rewards)

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

        context, memory_key_padding_mask = self._gen.get_context(
            source_ids, source_mask)
        # [batch_size x source_length x args.hidden_size]

        rewards = torch.ones_like(target_ids) * target_mask
        tloss = self.gen(context, memory_key_padding_mask,
                         target_ids, rewards=rewards)

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
            num_workers=self.args.gan_d_num_workers,
            pin_memory=True,
        )

        # Train dis for k epochs
        for d_epoch in trange(self.args.gan_d_epochs, desc="d-step epoch"):
            d_loss = AvgMeter()
            with tqdm(dataloader, "loss 0.0000") as bar:
                for batch in bar:
                    source_ids = self.to_dis_device(batch[0])
                    target_ids = self.to_dis_device(batch[1])
                    label = self.to_dis_device(batch[2])

                    pred = self.dis(source_ids, target_ids)
                    loss = F.binary_cross_entropy(pred, label)
                    if loss.size():
                        loss = loss.mean()  # mean() to average on multi-gpu

                    loss.backward()
                    self.d_opt.step()
                    self.d_opt.zero_grad()

                    loss = loss.item()
                    d_loss_avg = d_loss.update(loss)
                    bar.set_description(f"loss {d_loss_avg:.4f}")

            logging.info("d-epoch train loss %f", d_loss_avg)

    def train_epoch_gen(self, dataloader):
        # G-steps
        self.gen_to_gpu()
        self.gen_train()
        self.dis_to_gpu()
        self.dis_eval()
        g_loss = BatchAvgMeter()
        g_tloss = BatchAvgMeter()
        train_iter = iter(dataloader)
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

                    g_step_bar.set_description(
                        f"g-step {g_loss_avg:.4f} {g_tloss_avg:.4f}")
                else:
                    g_step_bar.set_description(f"g-step {g_loss_avg:.4f} -")

        logging.info("g-step train avg loss (gan only): %f", g_loss_avg)
        if self.args.gan_teach:
            logging.info("g-step train avg loss (teach only): %f", g_tloss_avg)
            logging.info("g-step train avg loss: %f",
                         (g_loss_avg + g_tloss_avg) / 2)

    def train_epoch_dis(self, train_dataset):
        # D-steps
        self.gen_eval()
        self.dis_train()
        for d_step in trange(self.args.gan_d_steps, desc="d-step"):
            # (re)generate fake dataset
            self.gen_to_gpu()
            indices = torch.randperm(self.args.gan_d_sample)
            subset = Subset(train_dataset, indices)
            dis_train_dataset = fakegen(self.args, subset, self._gen)
            self.gen_to_cpu()

            self.dis_to_gpu()
            self.train_step_dis(dis_train_dataset)
            self.dis_to_cpu()

    def train_epoch(self, train_dataset, train_dataloader):
        self.train_epoch_gen(train_dataloader)
        self.train_epoch_dis(train_dataset)
        self.save_models('latest')

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

        if is_distributed():
            gan_train_sampler = DistributedSampler(train_dataset)
        else:
            gan_train_sampler = RandomSampler(train_dataset)

        gan_train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.gan_batch_size,
            num_workers=self.args.gan_num_workers,
            pin_memory=True,
            sampler=gan_train_sampler,
        )
        logging.info("+ GAN batch size = %d", self.args.gan_batch_size)
        logging.info("+ GAN num workers = %d", self.args.gan_num_workers)

        for epoch in trange(self.args.gan_train_epochs, desc="Epoch"):
            self.train_epoch(train_dataset, gan_train_dataloader)
            self.eval_epoch(valid_dataset, bleu_dataset, epoch)

    def __prepare_eval(self):
        self.best_loss = MinMeter()
        self.best_bleu = MaxMeter()

    def eval(self, valid_dataset, bleu_dataset):
        self.dis_to_cpu()
        self.gen_eval()
        self.gen_to_gpu()
        self.eval_loss(valid_dataset)
        self.eval_bleu(bleu_dataset)

    def eval_epoch(self, valid_dataset, bleu_dataset, epoch):
        self.dis_to_cpu()
        self.gen_eval()
        self.gen_to_gpu()

        loss = self.eval_loss(valid_dataset)
        if self.best_loss.update(loss) == loss:
            logging.info("+ Best loss !!")
            self.save_models('best_loss', epoch, self.best_loss.get())

        dev_bleu = self.eval_bleu(bleu_dataset)
        if self.best_bleu.update(dev_bleu) == dev_bleu:
            logging.info("+ Best bleu !!")
            self.save_models('best_bleu', epoch, self.best_bleu.get())

    def eval_loss(self, valid_dataset):
        # Eval G with dev dataset
        logging.info("+ Valid dataset = %d", len(valid_dataset))
        logging.info("+ batch size = %d", self.args.eval_batch_size)
        logging.info("+ num workers = %d", self.args.gan_num_workers)

        loss = self.get_loss(valid_dataset)
        logging.info("+ eval loss = %f", loss)
        return loss

    def eval_bleu(self, bleu_dataset):
        logging.info("+ Bleu dataset = %d", len(bleu_dataset))
        logging.info("+ batch size = %d", self.args.eval_batch_size)
        logging.info("+ num workers = %d", self.args.gan_num_workers)

        dev_bleu = self.get_bleu(
            bleu_dataset,
            self.checkpoints["bleu_output"],
            self.checkpoints["bleu_gold"],
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
            num_workers=self.args.gan_num_workers,
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
            num_workers=self.args.gan_num_workers,
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
