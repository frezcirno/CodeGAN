# %%
import re
from typing import List
import pandas as pd
from pathlib import Path
import string
import process_utils as utils
import langdetect

# %%
csn = Path("/home/zixuantan/CodeSearchNet/csn/")
pqfile = csn / "java.parquet"

if not pqfile.is_file():
    java_files = sorted((csn / "java" / "final").glob("**/*.jsonl"))

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

    def jsonl_list_to_dataframe(file_list, columns=column_list):
        """Load a list of jsonl.gz files into a pandas DataFrame."""
        return pd.concat(
            [pd.read_json(f, orient="records", lines=True)[columns] for f in file_list],
            sort=False,
            ignore_index=True,
        )

    jd = jsonl_list_to_dataframe(java_files)
    jd.to_parquet(pqfile)


jd = pd.read_parquet(pqfile)

jd.info()


# %%


def remove_comment_token(code_tokens: List[str]):
    return [
        token
        for token in code_tokens
        if not token.startswith("//") and not token.startswith("/*")
    ]


def truncate_docstring(s: List[str]) -> List[str]:
    """ truncate docstrings at the first "@param" or "." """
    res = []
    for t in s:
        if t == "@param":
            break
        res.append(t)
        if t == ".":
            break
    return res


xml_regexp = re.compile(r"<\w+ |< \/ \w+ >")


def split_identifier(code_token: List[str]) -> List[str]:
    return sum((utils.split_code_identifier(token) for token in code_token), [])


def is_string_token(token: str):
    return token.startswith('"')


result = []


def unquote_string_tokenize(token: str):
    stripped = token.strip(""" .-_+*=":;!?$#/\<>()[]{}""")
    res = utils.tokenize_docstring_from_string(stripped)
    result.append((token, stripped, res))
    return res


def unquote_string_token(code_tokens: List[str]) -> List[str]:
    res = []

    for token in code_tokens:
        if is_string_token(token):
            res += unquote_string_tokenize(token)
        else:
            res.append(token)

    return res


def good_data(s: pd.Series):
    s.code_tokens = remove_comment_token(s.code_tokens)

    if len(s.code_tokens) <= 3 or len(s.code_tokens) >= 200:
        return None

    s.code_tokens = split_identifier(s.code_tokens)

    s.code_tokens = unquote_string_token(s.code_tokens)

    recode = " ".join(s.code_tokens)

    if not recode.isascii():
        return None

    s.docstring_tokens = truncate_docstring(s.docstring_tokens)

    if len(s.docstring_tokens) < 5 or len(s.docstring_tokens) > 256:
        return None

    redocstring = " ".join(s.docstring_tokens)

    if not redocstring.isascii():
        return None

    if not redocstring or all((ch not in redocstring for ch in string.ascii_letters)):
        return None

    if "/ *" in redocstring:
        return None

    if "TODO" in redocstring:
        return None

    if "{ @" in redocstring:
        return None

    if "<! --" in redocstring:
        return None

    if "---" in redocstring or "///" in redocstring or "***" in redocstring:
        return None

    if "https : // " in redocstring or "http : // " in redocstring:
        return None

    if re.findall(xml_regexp, redocstring):
        return None

    if redocstring.endswith("{"):
        return None

    return s


jd = jd.apply(good_data, axis=1, result_type="expand").dropna(0, how="any")


pd.DataFrame(result)[2].to_csv("data_objs/1.csv", index=None, header=None)

# %%

jd.info()

jd["recode"] = jd.code_tokens.map(lambda ts: " ".join(ts))
jd["redocstring"] = jd.docstring_tokens.map(lambda ts: " ".join(ts))

jd.recode.to_csv("data_objs/code.csv")
jd.redocstring.to_csv("data_objs/docstring.csv")
jd.to_parquet("data_objs/jd.parquet")

# %%
jd = pd.read_parquet("data_objs/jd.parquet")


# %%
def get_lang(s: str):
    return langdetect.detect(s)


lang = jd.redocstring.apply(get_lang)
