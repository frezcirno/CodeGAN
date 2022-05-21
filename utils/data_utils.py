from pathlib import Path
import pandas as pd


def read_jsonl(data_dir: str):
    file_list = sorted(Path(data_dir).glob("**/*.jsonl"))

    column_list = [
        "repo",
        "path",
        "url",
        "code",
        "code_tokens",
        "docstring",
        "docstring_tokens",
        "language",
        "partition",
    ]

    return pd.concat(
        [pd.read_json(f, orient="records", lines=True)[column_list] for f in file_list],
        sort=False,
        ignore_index=True,
    )
