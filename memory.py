import logging
import os
import torch


def check_mem(device: int):
    info = os.popen(
        f'/usr/bin/nvidia-smi -i {device} --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().split(',')
    return (int(el) for el in info)


def __occupy_mem(device):
    total, used = check_mem(device)
    x = torch.rand((256, 1024, int(total * 0.8 - used)), device=device)
    del x
    _, new_used = check_mem(device)
    logging.info("Occupy GPU %d: (%d -> %d)/%d", device,
                 used, new_used, total)


occupied = []


def occupy_mem(device):
    if device in occupied:
        return

    __occupy_mem(device)
    occupied.append(device)
