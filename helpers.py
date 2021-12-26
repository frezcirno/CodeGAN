import logging
import os
import _pickle as cPickle
import random
from typing import List
import numpy as np
import pandas as pd
import torch
from torch.functional import Tensor, F
from torch.utils.data import TensorDataset
from tqdm import tqdm
from pandas.io.parquet import to_parquet


def str_to_ids(inp, tokenizer):
    tokens = tokenizer.tokenize(inp)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, ids


def to_features(df, tokenizer):
    def make_features(s):
        code_tokens, code_ids = str_to_ids(s.recode, tokenizer)

        doc_tokens, doc_ids = str_to_ids(s.redocstring, tokenizer)

        return {
            "code_tokens": code_tokens,
            "doc_tokens": doc_tokens,
            "code_ids": code_ids,
            "doc_ids": doc_ids,
        }

    return df.apply(make_features, axis=1, result_type="expand")


def padding(ids, padding_to, bos_token_id, eos_token_id, pad_token_id):
    ids = [bos_token_id] + list(ids[: padding_to - 2]) + [eos_token_id]
    padding_length = padding_to - len(ids)
    mask = [1] * len(ids) + [0] * padding_length
    ids += [pad_token_id] * padding_length
    return ids, mask


def to_pad_features(
    df, source_length, target_length, bos_token_id, eos_token_id, pad_token_id
):
    def pad_features(s):
        code_ids, code_mask = padding(
            s.code_ids, source_length, bos_token_id, eos_token_id, pad_token_id
        )
        doc_ids, doc_mask = padding(
            s.doc_ids, target_length, bos_token_id, eos_token_id, pad_token_id
        )

        return {
            "code_ids": code_ids,
            "code_mask": code_mask,
            "doc_ids": doc_ids,
            "doc_mask": doc_mask,
        }

    return df.apply(pad_features, axis=1, result_type="expand")


def get_target_mask(target_ids: Tensor, bos_token_id, eos_token_id, pad_token_id):
    """
    bos, token, eos -> 1
    pad -> 0
    """
    mask = torch.ones_like(target_ids, dtype=torch.int64)

    for i, batch in enumerate(target_ids):
        try:
            index_of_eos = (batch == eos_token_id).nonzero()[0]
        except IndexError:
            continue
        mask[i][index_of_eos + 1 :] = 0

    target_ids[(1 - mask).bool()] = pad_token_id
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


def save_model(model, path):
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


def id_to_text(tkzr, pred_ids: List[int]):
    if tkzr.bos_token_id in pred_ids:
        pred_ids = pred_ids[pred_ids.index(tkzr.bos_token_id) + 1 :]
    if tkzr.eos_token_id in pred_ids:
        pred_ids = pred_ids[: pred_ids.index(tkzr.eos_token_id)]
    if tkzr.pad_token_id in pred_ids:
        pred_ids = pred_ids[: pred_ids.index(tkzr.pad_token_id)]
    text = tkzr.decode(pred_ids, clean_up_tokenization_spaces=False)
    return text


def tensor_to_text(tkzr, pred_ids: Tensor):
    pred_ids = list(pred_ids.cpu().numpy())
    return id_to_text(tkzr, pred_ids)


def tensors_to_text(tkzr, pred_ids: Tensor):
    return [tensor_to_text(tkzr, sample) for sample in pred_ids]


def train_generator(gen, gen_opt, gen_sch, dataloader, epochs):
    device = gen.device_ids[0]

    tr_loss = 0
    nb_tr_steps = 0

    for epoch in range(epochs):
        bar = tqdm(dataloader, total=len(dataloader), ncols=120)
        for batch in bar:
            source_ids = batch[0].to(device)
            source_mask = batch[1].to(device)
            target_ids = batch[2].to(device)
            target_mask = batch[3].to(device)

            loss, _, _ = gen(source_ids, source_mask, target_ids, target_mask)

            # mean() to average on multi-gpu
            if loss.size(0) > 1:
                loss = loss.mean()

            tr_loss += loss.item()
            train_loss = round(tr_loss / (nb_tr_steps + 1), 4)
            bar.set_description("epoch {} loss {}".format(epoch, train_loss))

            loss.backward()

            # Update parameters
            gen_opt.step()
            gen_opt.zero_grad()
            gen_sch.step()

            nb_tr_steps += 1


def train_generator_PG(gen, gen_opt, gen_sch, dis, dataloader, config):
    device = gen.device_ids[0]

    nb_tr_steps = 0

    bar = tqdm(dataloader, total=len(dataloader), ncols=120)
    for batch in bar:
        source_ids = batch[0].to(device)
        source_mask = batch[1].to(device)

        pre_target_ids, _ = gen.module.predict(source_ids, source_mask)
        pre_target_ids, pre_target_mask = get_target_mask(
            pre_target_ids,
            config.bos_token_id,
            config.eos_token_id,
            config.pad_token_id,
        )
        rewards = gen.module.get_reward(
            source_ids, source_mask, pre_target_ids, pre_target_mask, dis
        )

        loss = gen.module.gan_get_loss(
            source_ids, source_mask, pre_target_ids, pre_target_mask, rewards
        )

        loss.backward()

        gen_opt.step()
        gen_opt.zero_grad()
        gen_sch.step()

        nb_tr_steps += 1


def train_discriminator(dis, dis_opt, dataloader, epochs):
    """
    dataloader: no start/end symbol
    """
    device = dis.device_ids[0]

    tr_loss = 0
    nb_tr_steps = 0

    for epoch in range(epochs):
        bar = tqdm(dataloader, total=len(dataloader), ncols=120)
        for batch in bar:
            code_ids = batch[0].to(device)
            code_mask = batch[1].to(device)
            doc_ids = batch[2].to(device)
            doc_mask = batch[3].to(device)
            target = batch[4].to(device)

            out = dis(code_ids, code_mask, doc_ids, doc_mask)

            loss = F.binary_cross_entropy(out, target)

            # mean() to average on multi-gpu
            if loss.size():
                loss = loss.mean()

            tr_loss += loss.item()
            train_loss = round(tr_loss / (nb_tr_steps + 1), 4)
            bar.set_description("epoch {} loss {}".format(epoch, train_loss))

            loss.backward()

            # Update parameters
            dis_opt.step()
            dis_opt.zero_grad()

            nb_tr_steps += 1


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_fake_dataset(_gen, dataloader):
    fake_source_ids = []
    fake_source_mask = []
    fake_target_ids = []

    device = _gen.device_ids[0]

    _gen.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, "making fake dataset"):
            source_ids = batch[0].to(device)
            source_mask = batch[1].to(device)

            g_target_ids = _gen.beam_predict(source_ids, source_mask)

            fake_source_ids.append(source_ids.to("cpu"))
            fake_source_mask.append(source_mask.to("cpu"))
            fake_target_ids.append(g_target_ids.to("cpu"))

    fake_source_ids = torch.concat(fake_source_ids)
    fake_source_mask = torch.concat(fake_source_mask)
    fake_target_ids = torch.concat(fake_target_ids)

    return TensorDataset(fake_source_ids, fake_source_mask, fake_target_ids)


def mix_dataset(real_dataset, fake_dataset):
    """
    Input:
        real_dataset: (source_ids, source_mask, target_ids, target_mask)
        fake_dataset: (source_ids, source_mask, target_ids)
    Output:
        mixed_dataset: (source_ids, source_mask, target_ids, label)
    """
    fake_datas = list(fake_dataset.tensors[:3]) + [torch.zeros(len(fake_dataset))]
    real_datas = list(real_dataset.tensors[:3]) + [torch.ones(len(fake_dataset))]

    all_dataset = (
        torch.concat([fake_col, real_col])
        for fake_col, real_col in zip(fake_datas, real_datas)
    )
    return TensorDataset(*all_dataset)


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

