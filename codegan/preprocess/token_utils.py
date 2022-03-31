import re
from typing import List


DOCSTRING_REGEX_TOKENIZER = re.compile(
    r"""[^,'"`.():=*;<>{}+/\|\s\[\]\-\\]+|\\+|\.+|\(\)|\{\}|\[\]|\(+|\)+|:+|\[+|\]+|\{+|\}+|=+|\*+|;+|<+|>+|\++|-+|\|+|/+"""
)

CAMEL_REGEXP = re.compile(r"([A-Z][a-z0-9]+)")

IDENTIFIER_REGEXP = re.compile(r"^[_a-zA-Z][_a-zA-Z0-9]*$")

NUMBER_REGEXP = re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE]-?\d+)?f?$")


def is_identifier(token: str):
    return bool(re.match(IDENTIFIER_REGEXP, token))


def need_camel_split(token: str):
    return token.isalnum() and (not token.isupper() and not token.islower())


def split_camel_case(token: str):
    return [sub_token for sub_token in re.split(CAMEL_REGEXP, token) if sub_token]


def need_snake_split(s: str):
    return "_" in s.strip('_')


def split_snake_case(s: str):
    return [token for token in s.split("_") if token]


def tokenize_docstring_from_string(docstr: str) -> List[str]:
    return [
        t
        for t in DOCSTRING_REGEX_TOKENIZER.findall(docstr)
        if t is not None and len(t) > 0
    ]


def is_number_token(token: str):
    return bool(re.match(NUMBER_REGEXP, token))


def is_string_token(token: str):
    return token.startswith('"')


def is_comment_token(token: str) -> bool:
    token = token.strip()
    return token.startswith("//") or token.startswith("/*")


def test_is_identifier():
    assert is_identifier("abcAbcABC111") == True
    assert is_identifier("a") == True
    assert is_identifier("1") == False
    assert is_identifier("_") == True
    assert is_identifier("_1") == True


def test_need_camel_split():
    assert need_camel_split("split_camel_case") == False
    assert need_camel_split("helloWorld1TZXTestSdkIsAGoodCase") == True
    assert need_camel_split("hello") == False
    assert need_camel_split("a") == False
    assert need_camel_split("ABC") == False


def test_split_camel_case():
    assert split_camel_case("helloWorld1TZXTestSdkIsAGoodCase") == [
        'hello', 'World1', 'TZX', 'Test', 'Sdk', 'Is', 'A', 'Good', 'Case']


def test_need_snake_split():
    assert need_snake_split("split_camel_case") == True
    assert need_snake_split("helloWorld1TZXTestSdkIsAGoodCase") == False
    assert need_snake_split("hello") == False
    assert need_snake_split("a") == False
    assert need_snake_split("ABC") == False


def test_split_underscore_case():
    assert split_snake_case("split_camel_case") == ['split', 'camel', 'case']
    assert split_snake_case("asdasdds") == ['asdasdds']
    assert split_snake_case("hello") == ['hello']
    assert split_snake_case("a") == ['a']
    assert split_snake_case("ABC") == ['ABC']
