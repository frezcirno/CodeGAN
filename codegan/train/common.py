import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from .utils import Trainer, add_general_arguments, build_parallel, eval_gen_bleu, eval_gen_loss, is_notebook, save_model, to_features, pad_features
from ..utils.cache import cache_result


@cache_result("cache/dataset", format="torch")
def load_dataset(data, src_max_len, tgt_max_len):
    jd = pd.read_parquet(data)

    feats = to_features(jd)

    pad_feats = pad_features(feats, src_max_len, tgt_max_len)

    train_feats = pad_feats.loc[jd.partition == "train"]
    valid_feats = pad_feats.loc[jd.partition == "valid"]
    test_feats = pad_feats.loc[jd.partition == "test"]

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
