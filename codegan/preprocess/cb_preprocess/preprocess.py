import os
import pandas as pd

dataset = os.path.dirname(__file__)+'/dataset'


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    lines = []
    for part in ['train', 'valid', 'test']:
        with open(f"{dataset}/java/{part}.txt") as f1:
            lines.extend(f1.readlines())
    lines = [line.strip() for line in lines]
    return df[df['url'].isin(lines)]
