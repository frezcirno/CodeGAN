import os
from pathlib import Path
import pandas as pd

cache_home = os.path.expanduser(
    os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "codegan")
)


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


def get_cached_file(file: str) -> str:
    os.makedirs(cache_home, exist_ok=True)
    return os.path.join(cache_home, file)
