# %%
from os.path import join as path_join
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer

from helpers import cache_call

# %%
data_dir = "data_objs"

jd = pd.read_parquet(path_join(data_dir, "jd.parquet"))

tokenizer = RobertaTokenizer.from_pretrained(
    "microsoft/codebert-base", do_lower_case=True
)

# %%


def str_to_ids(tkzr, inp):
    tokens = tkzr.tokenize(inp)
    ids = tkzr.convert_tokens_to_ids(tokens)
    return tokens, ids


def make_features(s, tkzr):
    code_tokens, code_ids = str_to_ids(tkzr, s.recode)

    doc_tokens, doc_ids = str_to_ids(tkzr, s.redocstring)

    return {
        "code_tokens": code_tokens,
        "doc_tokens": doc_tokens,
        "code_ids": code_ids,
        "doc_ids": doc_ids,
    }


def to_features(df):
    return df.apply(make_features, axis=1, args=(tokenizer,), result_type="expand",)


features = cache_call("data_objs/features", to_features)(jd)

# %%
# Code token hist
right = 385
plt.xticks(range(0, right, 32))
features.code_tokens.apply(len).hist(bins=range(0, right, 16))
plt.savefig("bpe_code_token_hist.png")

# %%
# BPE code token length percent-x
with open("bpe_code_token_hist.txt", "w") as f:
    for percent in range(5, 10, 1):
        percent = percent / 10
        print(
            f"p{int(percent*100)} = {features.code_tokens.apply(len).quantile(percent)}",
            file=f,
        )
    for i in range(91, 101, 1):
        percent = i / 100
        print(f"p{i} = {features.code_tokens.apply(len).quantile(percent)}", file=f)


# %%
# Doc token hist
right = 60
plt.xticks(range(0, right, 4))
features.doc_tokens.apply(len).hist(bins=range(0, right, 4))
plt.savefig("bpe_doc_token_hist.png")

# %%
# BPE doc token length percent-x
with open("bpe_doc_token_hist.txt", "w") as f:
    for i in range(50, 91, 10):
        percent = i / 100
        print(f"p{i} = {features.doc_tokens.apply(len).quantile(percent)}", file=f)
    for i in range(91, 101, 1):
        percent = i / 100
        print(f"p{i} = {features.doc_tokens.apply(len).quantile(percent)}", file=f)


# %%
