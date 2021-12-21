# %%
import tqdm
import pandas as pd
from pathlib import Path
from langdetect import detect

# %%
java_files = sorted(Path("/home/zixuantan/CodeSearchNet/csn/java/").glob("**/*.gz"))

column_list = [
    # 'repo',
    # 'path',
    # 'url',
    # 'code',
    "code_tokens",
    # 'docstring',
    "docstring_tokens",
    # 'language',
    "partition",
]


def jsonl_list_to_dataframe(file_list, columns=column_list):
    """Load a list of jsonl.gz files into a pandas DataFrame."""
    return pd.concat(
        [
            pd.read_json(f, orient="records", compression="gzip", lines=True)[columns]
            for f in file_list
        ],
        sort=False,
    )


java_data = jsonl_list_to_dataframe(java_files)


# from codedoc_data import *

java_data = pd.read_pickle("jd.pkl")


def strip_comment(code_token):
    new_code_token = []
    for token in code_token:
        if token.startswith("//") or token.startswith("/*"):
            continue
        new_code_token.append(token)
    return new_code_token


java_data["code_tokens"] = java_data["code_tokens"].map(strip_comment)


def is_getter_or_setter(code_tokens):
    mask = pd.Series(False, index=code_tokens.index)

    for i, code_token in enumerate(code_tokens):
        if len(code_token) < 12:
            continue

        for j in range(len(code_token) - 3):
            if code_token[j + 1] == "(":
                fn_idx = j
                break
        else:
            continue

        fn_name = code_token[fn_idx]
        if len(fn_name) <= 3 or not (
            fn_name.startswith("get") or fn_name.startswith("set")
        ):
            continue

        mask.iloc[i] = True

    return mask


m = is_getter_or_setter(java_data["code_tokens"])
java_data = java_data[~m]


def is_english(doc_tokens):
    mask = pd.Series(True, index=doc_tokens.index)

    for i, doc_token in tqdm.tqdm(enumerate(doc_tokens), mininterval=1):
        try:
            if detect(" ".join(doc_token)) != "en":
                mask.iloc[i] = False
        except Exception as ex:
            # print(doc_token, ex)
            mask.iloc[i] = False

    return mask


m = is_english(java_data["docstring_tokens"])
java_data = java_data[m]

java_data.to_pickle("jd.processed.pkl")
