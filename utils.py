import re
from typing import List


DOCSTRING_REGEX_TOKENIZER = re.compile(
    r"""[^,'"`.():=*;<>{}+/\|\s\[\]\-\\]+|\\+|\.+|\(\)|\{\}|\[\]|\(+|\)+|:+|\[+|\]+|\{+|\}+|=+|\*+|;+|<+|>+|\++|-+|\|+|/+"""
)


identifier_regexp = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")


def is_identifier(s: str):
    return bool(re.match(identifier_regexp, s))


# print(is_identifier("abcAbcABC111"))
# print(is_identifier("a"))
# print(is_identifier("1"))
# print(is_identifier("_"))
# print(is_identifier("_1"))


def is_camel_identifier(s: str):
    return s.isalnum()


# print(is_camel_identifier("split_camel_case"))
# print(is_camel_identifier("helloWorld1TZXTestSdkIsAGoodCase"))
# print(is_camel_identifier("hello"))
# print(is_camel_identifier("a"))
# print(is_camel_identifier("ABC"))

camel_regexp = re.compile(r"([A-Z][a-z0-9]+)")


def split_camel_case(s: str):
    return [token for token in re.split(camel_regexp, s) if token]


# print(split_camel_case("split_camel_case"))
# print(split_camel_case("helloWorld1TZXTestSdkIsAGoodCase"))
# print(split_camel_case("hello"))
# print(split_camel_case("a"))
# print(split_camel_case("ABC"))


def is_underscore_identifier(s: str):
    return "_" in s


# print(is_underscore_identifier("split_camel_case"))
# print(is_underscore_identifier("helloWorld1TZXTestSdkIsAGoodCase"))
# print(is_underscore_identifier("hello"))
# print(is_underscore_identifier("a"))
# print(is_underscore_identifier("ABC"))


def split_underscore_case(s: str):
    return [token for token in s.split("_") if token]


# print(split_underscore_case("split_camel_case"))
# print(split_underscore_case("asdasdds"))
# print(split_underscore_case("hello"))
# print(split_underscore_case("a"))
# print(split_underscore_case("ABC"))


def split_code_identifier(token: str) -> List[str]:
    if not is_identifier(token):
        return [token]
    if is_camel_identifier(token):
        return split_camel_case(token)
    if is_underscore_identifier(token):
        return split_underscore_case(token)
    return [token]


def tokenize_docstring_from_string(docstr: str) -> List[str]:
    return [
        t
        for t in DOCSTRING_REGEX_TOKENIZER.findall(docstr)
        if t is not None and len(t) > 0
    ]

