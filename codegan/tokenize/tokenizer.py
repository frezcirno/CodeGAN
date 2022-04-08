import os
from typing import List
from torch import Tensor
from transformers import RobertaTokenizer


tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(
    os.path.dirname(__file__) + "/data",
    do_lower_case=True
)

bos_token_id: int = tokenizer.bos_token_id
eos_token_id: int = tokenizer.eos_token_id
pad_token_id: int = tokenizer.pad_token_id


def tokenize(s):
    return tokenizer.tokenize(s)


def convert_tokens_to_ids(tokens):
    return tokenizer.convert_tokens_to_ids(tokens)


def decode(ids):
    return tokenizer.decode(ids, clean_up_tokenization_spaces=False)


def str_to_ids(inp):
    tokens = tokenize(inp)
    ids = convert_tokens_to_ids(tokens)
    return tokens, ids


def id_to_text(pred_ids: List[int]):
    if bos_token_id in pred_ids:
        pred_ids = pred_ids[pred_ids.index(bos_token_id) + 1:]
    if eos_token_id in pred_ids:
        pred_ids = pred_ids[: pred_ids.index(eos_token_id)]
    if pad_token_id in pred_ids:
        pred_ids = pred_ids[: pred_ids.index(pad_token_id)]
    text = decode(pred_ids)
    return text


def tensor_to_text(pred_ids: Tensor) -> List[str]:
    pred_ids = list(pred_ids.cpu().numpy())
    return id_to_text(pred_ids)


def tensors_to_text(pred_ids: Tensor) -> List[List[str]]:
    return [tensor_to_text(sample) for sample in pred_ids]


if __name__ == '__main__':
    tokens = tokenize("def tensors_to_text(pred_ids: Tensor): return [tensor_to_text(sample) for sample in pred_ids]")
    print(tokens)
