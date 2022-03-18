# /usr/bin/env python3
import sys
import pandas as pd


def die(msg: str):
    print(msg)
    exit(1)


def readlines(f):
    with open(f, "r") as f:
        return f.readlines()


def split(s):
    return s.split('\t')


def readbleu(f):
    lines = readlines(f)
    return {int(idx): cont.strip() for idx, cont in map(split, lines)}


if len(sys.argv) < 3:
    die(f"usage: {sys.argv[0]} OUT1 OUT2")

bleuset = pd.read_parquet("data_objs/jd_bleu.parquet")

out1 = readbleu(sys.argv[1])
out2 = readbleu(sys.argv[2])

for idx in out1.keys():
    if out1[idx] != out2[idx]:
        print("=" * 10)
        print(idx)
        print("> Origin code:")
        print(bleuset['code'][idx])
        print("> Preprocessed code:")
        print(bleuset['recode'][idx])
        print("Res1:", out1[idx])
        print("Res2:", out2[idx])
        print("Gold:", bleuset['redocstring'][idx])
