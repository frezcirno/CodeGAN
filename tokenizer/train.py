import os
import tempfile
import pandas as pd
from typing import List
from torch import Tensor
from tokenizers import AddedToken, Tokenizer, CharBPETokenizer
from tokenizers import AddedToken, Tokenizer
from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
import argparse


def train(df: pd.DataFrame, tokenizer: ByteLevelBPETokenizer):
    codes = df['code'].to_list()
    docstrings = df['docstring'].to_list()

    with tempfile.NamedTemporaryFile("w+") as fcode, tempfile.NamedTemporaryFile("w") as fdoc:
        for code in codes:
            fcode.write(code)
        for doc in docstrings:
            fdoc.write(doc)

        fcode.seek(0)
        fdoc.seek(0)

        tokenizer.train([fcode.name, fdoc.name], vocab_size=50265, special_tokens=['<s>', '<pad>', '</s>'])


parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str)

args = parser.parse_args()


tokenizer = ByteLevelBPETokenizer()

train(pd.read_parquet(args.data_path), tokenizer)

tokenizer.save(os.path.dirname(__file__) + "/trained.json")
tokenizer.save(os.path.dirname(__file__) + "/trained.pretty.json", pretty=True)
