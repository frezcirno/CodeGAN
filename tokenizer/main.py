import os
import numpy as np
import swifter
import argparse
import torch
import pandas as pd
from typing import List
from torch import Tensor
from torch.utils.data import TensorDataset
from api import token_to_id


def make_dataset(df: pd.DataFrame, src_max_len, tgt_max_len):

    code_ids_mask = df['code'].swifter.apply(lambda s: pd.Series(
        token_to_id(s, src_max_len), index=['code_ids', 'code_mask']))
    doc_ids_mask = df['docstring'].swifter.apply(lambda s: pd.Series(
        token_to_id(s, tgt_max_len), index=['doc_ids', 'doc_mask']))

    feats = pd.concat([code_ids_mask, doc_ids_mask], axis=1)

    train_feats = feats.loc[df.partition == "train"]
    valid_feats = feats.loc[df.partition == "valid"]
    test_feats = feats.loc[df.partition == "test"]

    train_dataset = TensorDataset(
        torch.tensor(np.array(train_feats['code_ids'].to_list())),
        torch.tensor(np.array(train_feats['code_mask'].to_list())),
        torch.tensor(np.array(train_feats['doc_ids'].to_list())),
        torch.tensor(np.array(train_feats['doc_mask'].to_list())),
    )

    valid_dataset = TensorDataset(
        torch.tensor(np.array(valid_feats['code_ids'].to_list())),
        torch.tensor(np.array(valid_feats['code_mask'].to_list())),
        torch.tensor(np.array(valid_feats['doc_ids'].to_list())),
        torch.tensor(np.array(valid_feats['doc_mask'].to_list())),
    )

    test_dataset = TensorDataset(
        torch.tensor(np.array(test_feats['code_ids'].to_list())),
        torch.tensor(np.array(test_feats['code_mask'].to_list())),
        torch.tensor(np.array(test_feats['doc_ids'].to_list())),
        torch.tensor(np.array(test_feats['doc_mask'].to_list())),
    )

    return train_dataset, valid_dataset, test_dataset


parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str)
parser.add_argument("--src_max_len", type=int, default=256)
parser.add_argument("--tgt_max_len", type=int, default=32)
parser.add_argument("-o", "--output", type=str, required=True)

args = parser.parse_args()


df = pd.read_parquet(args.data_path)

datasets = make_dataset(df, args.src_max_len, args.tgt_max_len)

torch.save(datasets, args.output)

print("done.")
