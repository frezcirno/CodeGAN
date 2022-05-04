import re
import multiprocessing as mp
from typing import List, Literal
import pandas as pd
import dask.dataframe as dd
import string
import swifter
import langdetect
from . import token_utils
from tree_sitters.tokenize import parse_ast

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


def parse_code(code: str,
               handle_id: Literal['none|split'] = 'split',
               handle_str: Literal['none|mask|unquote'] = "unquote",
               handle_num: Literal['none|mask'] = "mask",):
    tree = parse_ast(code)

    # retrival
    nodes = []
    nodes_to_expand = [tree.root_node]
    while nodes_to_expand:
        node = nodes_to_expand.pop(0)
        if not node.children and node.text:
            nodes.append(node)
        nodes_to_expand = node.children + nodes_to_expand

    # process
    tokens = []
    for node in nodes:
        text = node.text.decode('utf-8')
        if node.type == 'identifier' or node.type == 'type_identifier':
            if handle_id == 'split':
                segs = token_utils.split_snake_case(text)
                for seg in segs:
                    if token_utils.need_camel_split(seg):
                        subseg = token_utils.split_camel_case(seg)
                        tokens.extend(subseg)
                    else:
                        tokens.append(seg)
            else:
                tokens.append(text)

        elif node.type == 'line_comment' or node.type == 'block_comment':
            # Skip comments
            continue

        elif node.type == 'character_literal':
            tokens.append(text)

        elif node.type == 'string_literal':
            if handle_str == 'mask':
                tokens.append('<STR>')
            elif handle_str == 'unquote':
                stripped = text.strip(""" .-_+*=":;!?$#/\<>()[]{}""")
                tokens.extend(token_utils.tokenize_docstring_from_string(stripped))
            else:
                tokens.append(text)

        elif node.type.endswith('floating_point_literal') or node.type.endswith('integer_literal'):
            if handle_num == 'mask':
                tokens.append("<NUM>")
            else:
                tokens.append(text)

        else:
            tokens.append(text)

    return tokens


def process(
    df: pd.DataFrame,
    handle_id: Literal['none|split'] = 'split',
    handle_str: Literal['none|mask|unquote'] = "unquote",
    handle_num: Literal['none|mask'] = "mask",
) -> pd.DataFrame:

    code_tokens = df['code'].swifter.apply(parse_code, args=(handle_id, handle_str, handle_num))
    docstring_tokens = df['docstring_tokens'].swifter.apply(truncate_docstring)

    df['code_tokens'] = code_tokens
    df['docstring_tokens'] = docstring_tokens

    df_counter = len(df)

    # Rebuild the code and docstring from tokens
    df['code'] = df['code_tokens'].map(' '.join)
    df['docstring'] = df['docstring_tokens'].map(' '.join)

    # drop the sample with non-ascii in code or docstring
    df = df[(df['code'].map(lambda s: s.isascii())) & (df['docstring'].map(lambda s: s.isascii()))]

    new_counter = len(df)
    print(f"drop the sample with non-ascii in code or docstring: {df_counter} -> {new_counter}")
    df_counter = new_counter

    # drop the sample without any letters in code or docstring
    df = df[df['docstring'].map(lambda s: any(letter in s for letter in string.ascii_letters))]

    new_counter = len(df)
    print(f"drop the sample without any letters in code or docstring: {df_counter} -> {new_counter}")
    df_counter = new_counter

    # drop the sample if docstring contains any bad words
    df = df[~df['docstring'].map(lambda s: any(bad_word in s for bad_word in BAD_WORDS))]

    new_counter = len(df)
    print(f"drop the sample if docstring contains any bad words: {df_counter} -> {new_counter}")
    df_counter = new_counter

    # drop the sample if docstring contains any xml tags
    df = df[~df['docstring'].map(lambda s: bool(re.findall(XML_REGEXP, s)))]

    new_counter = len(df)
    print(f"drop the sample if docstring contains any xml tags: {df_counter} -> {new_counter}")
    df_counter = new_counter

    # drop the sample if docstring ends with '{'
    df = df[~df['docstring'].map(lambda s: s.endswith("{"))]

    new_counter = len(df)
    print(f"drop the sample if docstring ends with '{{': {df_counter} -> {new_counter}")
    df_counter = new_counter

    # df.to_parquet("mid.parquet")

    # drop the sample if the code contains too many tokens
    code_tokens_len = df['code_tokens'].map(len)
    df = df[(3 < code_tokens_len) & (code_tokens_len <= 950)]

    new_counter = len(df)
    print(f"drop samples if the code contains too many tokens: {df_counter} -> {new_counter}")
    df_counter = new_counter

    # drop the sample if the docstring contains too many tokens
    docstring_tokens_len = df['docstring_tokens'].map(len)
    df = df[(4 < docstring_tokens_len) & (docstring_tokens_len <= 32)]

    new_counter = len(df)
    print(f"drop samples if the docstring contains too many tokens: {df_counter} -> {new_counter}")
    df_counter = new_counter

    return df
