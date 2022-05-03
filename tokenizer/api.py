import os
from typing import List, Sequence, Tuple
from torch import Tensor
from transformers import RobertaTokenizer
from tokenizers import Tokenizer


tokenizer = Tokenizer.from_file(os.path.dirname(__file__) + "/trained.json")

bos_token_id = tokenizer.token_to_id("<s>")
eos_token_id = tokenizer.token_to_id("</s>")
pad_token_id = tokenizer.token_to_id("<pad>")


def token_to_id(code: str, maxlen=None):
    lower_code = code.lower()
    t = tokenizer.encode(lower_code)
    ids = [bos_token_id] + t.ids + [eos_token_id]

    return padding(ids, maxlen) if maxlen else ids


def id_to_token(ids: List[int]):
    if isinstance(ids, Tensor):
        ids = list(ids.cpu().numpy())
    return tokenizer.decode(ids)


def tensors_to_text(pred_ids: Tensor) -> List[List[str]]:
    return [id_to_token(sample) for sample in pred_ids]


def padding(ids: Sequence[int], padding_to: int) -> Tuple[List[int], List[int]]:
    ids = [bos_token_id] + list(ids[: padding_to - 2]) + [eos_token_id]
    id_len = len(ids)
    padding_length = padding_to - id_len

    ids += [pad_token_id] * padding_length
    mask = [1] * id_len + [0] * padding_length
    return ids, mask


if __name__ == '__main__':
    tokens = token_to_id(
        "def tensors_to_text(pred_ids: Tensor): return [tensor_to_text(sample) for sample in pred_ids]")
    print(tokens)
