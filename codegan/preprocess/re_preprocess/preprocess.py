import re
from typing import List, Literal
import pandas as pd
import string
from .. import token_utils
import langdetect


BAD_WORDS = ['TODO :', 'XXX :', '/ *', '{ @', '<! --', '---', '///', '***', '~~~', 'http : //', 'https : //']

XML_REGEXP = re.compile(r"<\w+ |< \/ \w+ >")


def get_lang(s: str):
    return langdetect.detect(s)


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


def split_identifiers(code_tokens: List[str]) -> List[str]:
    new_tokens = []
    for token in code_tokens:
        if token_utils.is_identifier(token):
            if token_utils.need_camel_split(token):
                new_tokens.extend(token_utils.split_camel_case(token))
            elif token_utils.need_snake_split(token):
                new_tokens.extend(token_utils.split_snake_case(token))
            else:
                new_tokens.append(token)
        else:
            new_tokens.append(token)
    return new_tokens


def unquote_string_tokenize(res: List[str], token: str) -> List[str]:
    stripped = token.strip(""" .-_+*=":;!?$#/\<>()[]{}""")
    res.extend(token_utils.tokenize_docstring_from_string(stripped))
    return res


def mask_string_token(res: List[str], token: str) -> List[str]:
    res.append('<STR>')
    return res


def process_string_tokens(code_tokens: List[str], method: Literal['unquote|mask'] = 'unquote') -> List[str]:
    process_fn = unquote_string_tokenize if method == 'unquote' else mask_string_token
    res = []

    for token in code_tokens:
        if token_utils.is_string_token(token):
            res = process_fn(res, token)
        else:
            res.append(token)

    return res


def remove_comment_token(code_tokens: List[str]):
    return filter(lambda x: not token_utils.is_comment_token(x), code_tokens)


def process_number_tokens(code_tokens: List[str]) -> List[str]:
    res = []

    for token in code_tokens:
        if token_utils.is_number_token(token):
            res.append('<NUM>')
        else:
            res.append(token)

    return res


def preprocess(
    df: pd.DataFrame,
    split_id=True,
    unquote_str=True,
    mask_str=False,
    mask_num=True,
) -> pd.DataFrame:
    def process_data(s: pd.Series):
        if unquote_str and mask_str:
            raise RuntimeError("unquote_str and mask_str cannot be specified together.")

        s.code_tokens = remove_comment_token(s.code_tokens)

        if split_id:
            s.code_tokens = split_identifiers(s.code_tokens)

        if mask_str:
            s.code_tokens = process_string_tokens(s.code_tokens, 'mask')
        elif unquote_str:
            s.code_tokens = process_string_tokens(s.code_tokens)

        if mask_num:
            s.code_tokens = process_number_tokens(s.code_tokens)

        if len(s.code_tokens) <= 3 or len(s.code_tokens) >= 200:
            return None

        recode = " ".join(s.code_tokens)
        if not recode.isascii():
            return None

        s.docstring_tokens = truncate_docstring(s.docstring_tokens)

        if len(s.docstring_tokens) < 5 or len(s.docstring_tokens) > 256:
            return None

        redocstring = " ".join(s.docstring_tokens)

        """drop the sample if:
        1. docstring is not in english
        2. docstring contains no letters (all symbols)
        3. docstring contains urls, etc.
        4. docstring contains xml tags.
        """
        if not redocstring \
                or not redocstring.isascii() \
                or not any(letter in redocstring for letter in string.ascii_letters) \
                or any(bad_word in redocstring for bad_word in BAD_WORDS) \
                or re.findall(XML_REGEXP, redocstring) \
                or redocstring.endswith("{"):
            return None

        # Reconstructed code and docstring
        s.code = recode
        s.docstring = redocstring

        return s

    return df.apply(process_data, axis=1, result_type="expand").dropna(0, how="any")
