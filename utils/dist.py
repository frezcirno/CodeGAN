import os
import torch.distributed as dist


def env_get(key: str) -> str:
    return os.environ.get(key)


def is_distributed() -> bool:
    return os.environ.get("LOCAL_RANK") != None


def local_rank() -> int:
    return int(env_get("LOCAL_RANK"))


def rank() -> int:
    return int(env_get("RANK"))


def world_size() -> int:
    return int(env_get("WORLD_SIZE"))


def is_master() -> bool:
    return os.environ.get("LOCAL_RANK") == "0"
