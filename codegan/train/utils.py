import argparse
from datetime import datetime, timedelta
import logging
import os
import random
from typing import List, Literal, Optional, OrderedDict, Sequence, Tuple
import numpy as np
import pandas as pd
import swifter
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, ConcatDataset, RandomSampler
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import (
    DataLoader,
    SubsetRandomSampler,
    TensorDataset,
    Subset,
    ConcatDataset,
    SequentialSampler,
)
from nlgeval import compute_metrics
from torch.utils.data.distributed import DistributedSampler

from codegan import bleu
# from torch.utils.tensorboard import SummaryWriter
import tokenizer
from ..utils.cache import cache_result
from ..utils.memory import occupy_mem
from ..utils.meter import BatchAvgMeter, MinMeter
from ..utils.dist import is_distributed, is_master, local_rank, rank, world_size
logger = logging.getLogger(__name__)

# writer = SummaryWriter()


def save_model(model, path):
    if is_distributed() and not is_master():
        return

    # Only save the model itself
    model_to_save = getattr(model, "module", model)
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    logger.info(f"saving model to {path}")
    torch.save(model_to_save.state_dict(), path)


def make_gan_dataset(gen, device, dataloader,
                     num_batchs=0, beam_search=True) -> TensorDataset:
    all_source_ids = []
    all_target_ids = []
    all_g_target_ids = []
    batch_size_sum = 0
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, "Making fake dataset", dynamic_ncols=True):
            source_ids = batch[0]
            source_mask = batch[1]
            target_ids = batch[2]

            batch_size = source_ids.size(0)

            all_source_ids.append(source_ids)
            all_target_ids.append(target_ids)

            g_target_ids, _ = gen(source_ids.to(device),
                                  source_mask.to(device),
                                  beam_search=beam_search)
            all_g_target_ids.append(g_target_ids.to('cpu'))

            batch_size_sum += batch_size

            if num_batchs:
                batch_count += 1
                if batch_count >= num_batchs:
                    break

    all_source_ids = torch.cat(all_source_ids + all_source_ids)
    all_target_ids = torch.cat(all_target_ids + all_g_target_ids)
    all_labels = torch.cat([torch.ones(batch_size_sum, dtype=torch.float),
                            torch.zeros(batch_size_sum, dtype=torch.float)])

    return TensorDataset(all_source_ids, all_target_ids, all_labels)


def fakegen2(
    gen, device, dataset: TensorDataset,
    num_batchs, batch_size,
    beam_search=False, num_workers=4
) -> TensorDataset:
    """Generator mixed samples for discriminator training.

    If it's distributed training, every process will generate a unique portion of `dataset`.

    Outputs:
        dataset: [source_ids, target_ids, labels]
    """
    dataloader, sampler = eval_dataloader(dataset, batch_size, num_workers)

    dataset = make_gan_dataset(gen, device, dataloader, num_batchs=num_batchs, beam_search=beam_search)
    if is_distributed():
        tensors = gather_all(dataset.tensors)
        return TensorDataset(*[torch.cat(ts) for ts in zip(*tensors)])
    else:
        return dataset


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta

    def __call__(self, val_loss):
        """ return True if should stop training """

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def write_output(output_file, gold_file, predicts, golds):
    with open(output_file, "w") as f, open(gold_file, "w") as fgold:
        for pred, gold in zip(predicts, golds):
            f.write(pred + "\n")
            fgold.write(gold + "\n")


def evaluate_metrics(
    model,
    device,
    dataset,
    output_file: str,
    gold_file: str,
    batch_size,
    num_workers=4
) -> float:
    dataloader, sampler = eval_dataloader(dataset, batch_size, num_workers)
    predicts = []
    golds = []
    with torch.no_grad():
        with tqdm(dataloader, dynamic_ncols=True) as bar:
            for batch in bar:
                source_ids = batch[0]
                source_mask = batch[1]
                target_ids = batch[2]
                preds, _ = model(source_ids.to(device),
                                 source_mask.to(device), beam_search=True)
                predicts.extend(tokenizer.tensors_to_text(preds))
                golds.extend(tokenizer.tensors_to_text(target_ids))

    if is_distributed():
        predicts = sum(gather_all(predicts), [])
        golds = sum(gather_all(golds), [])

    write_output(output_file, gold_file, predicts, golds)

    metrics = compute_metrics(hypothesis=output_file, references=[gold_file],
                              no_skipthoughts=True, no_glove=True)

    predictions = bleu.write_output(output_file, gold_file, predicts, golds)
    goldMap, predictionMap = bleu.computeMaps(predictions, gold_file)
    metrics['bleu4'] = bleu.bleuFromMaps(goldMap, predictionMap)[0]

    return metrics


def eval_gen_loss(
    gen: nn.Module,
    device: int,
    dataset: TensorDataset,
    batch_size: int,
    num_workers: int = 4
) -> float:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    loss_meter = BatchAvgMeter()
    with tqdm(dataloader, "Eval CEloss 0.0000", dynamic_ncols=True) as bar:
        for batch in bar:
            source_ids = batch[0].to(device)
            source_mask = batch[1].to(device)
            target_ids = batch[2].to(device)
            target_mask = batch[3].to(device)

            with torch.no_grad():
                loss, num = gen(source_ids, source_mask,
                                target_ids, target_mask)
                if loss.size():
                    num = num.sum()
                    loss = loss.sum()

            loss = loss_meter.update(loss.item(), num.item())
            bar.set_description(f"Eval CEloss {loss: .4f}")

    return loss


def eval_dis_loss(
    dis: nn.Module,
    device: int,
    dataset: TensorDataset,
    batch_size: int,
    num_workers: int = 4,
) -> float:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    loss_meter = BatchAvgMeter()
    with torch.no_grad():
        with tqdm(dataloader, "BCEloss 00.0000", dynamic_ncols=True) as bar:
            for batch in bar:
                source_ids = batch[0].to(device)
                target_ids = batch[1].to(device)
                labels = batch[2].to(device)

                prob = dis(source_ids, target_ids)
                loss = F.binary_cross_entropy(prob, labels, reduction='sum')
                # mean() to average on multi-gpu
                if loss.size():
                    loss = loss.mean()

                avg_loss = loss_meter.update(loss.item(), labels.size(0))
                bar.set_description(f"BCEloss {avg_loss:.4f}")

    return avg_loss


def eval_dis_acc(
    model: nn.Module,
    device: int,
    dataset: TensorDataset,
    batch_size: int,
    num_workers: int = 4,
) -> float:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    acc_meter = BatchAvgMeter()
    with torch.no_grad():
        with tqdm(dataloader, "Accu 00.0000", dynamic_ncols=True) as bar:
            for batch in bar:
                source_ids = batch[0].to(device)
                target_ids = batch[1].to(device)
                labels = batch[2]

                prob = model(source_ids, target_ids).to("cpu").gt(0.5)

                right = labels.eq(prob).sum()
                total = source_ids.size(0)

                acc = acc_meter.update(right, total)
                bar.set_description(f"Accu {acc:.4f}")
    return acc


def validate_device_ids(device_ids: List[int]) -> List[int]:
    ngpu = torch.cuda.device_count()

    return [dev_id for dev_id in device_ids if dev_id < ngpu] \
        or (list(range(world_size())) if is_distributed() else [0])


def add_general_arguments(parser: argparse.ArgumentParser):
    # General
    parser.add_argument("--data", type=str, help="path to the dataset", required=True)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--device_ids", nargs="+", type=int, default=[])
    parser.add_argument("--gen_device_ids", nargs="+", type=int, default=[])
    parser.add_argument("--dis_device_ids", nargs="+", type=int, default=[])
    parser.add_argument("--gen_no_ddp", action="store_true")
    parser.add_argument("--dis_no_ddp", action="store_true")
    parser.add_argument("--occupy", action="store_true")


class Trainer(object):
    def __init__(self, args, run_dir, device):
        self.model = None
        self.device = device

        self.do_train = args.do_train
        self.do_eval = args.do_eval
        self.train_epochs = args.train_epochs
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.learning_rate = args.learning_rate
        self.adam_epsilon = args.adam_epsilon
        self.weight_decay = args.weight_decay
        self.load_path = args.load_path

        self.do_train = args.do_train
        self.do_eval = args.do_eval

        self.run_path = run_dir
        self.paths = {}
        self.best_loss = MinMeter()

    @classmethod
    def build_parallel(
        cls,
        model,
        find_unused_parameters,
        device=0,
        dp_device_ids=[]
    ):
        if is_distributed():
            logger.info("Using DistributedDataParallel")
            model = DDP(model,
                        device_ids=[device],
                        find_unused_parameters=find_unused_parameters,
                        broadcast_buffers=False)
        elif len(dp_device_ids) > 1:
            logger.info("Using DataParallel")
            model = DP(model, dp_device_ids)
            model.device = device

        return model

    def to_model_device(self, x: Tensor):
        return x.to(self.device)

    def train_dataloader(self, train_dataset):
        return train_dataloader(train_dataset, self.batch_size, self.num_workers)

    def eval_dataloader(self, dataset):
        return eval_dataloader(dataset, self.eval_batch_size, self.num_workers)

    def run(self, *datasets):
        if self.do_train:
            self.train(*datasets)
        if self.do_eval:
            self.eval(*datasets)

    def train(self, *datasets):
        logger.error("No train operation!")

    def eval(self, *datasets):
        logger.error("No eval operation!")

    def register_path(self, key, path):
        if key in self.paths:
            logger.warn(f"{key} is already in paths")
        self.paths[key] = os.path.join(self.run_path, path)

    def get_path(self, key, *args):
        if args:
            return self.paths[key] % args
        return self.paths[key]

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--do_train", action="store_true")
        parser.add_argument("--do_eval", action="store_true")
        parser.add_argument("--train_epochs", type=int, default=10)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--eval_batch_size", type=int, default=64)
        parser.add_argument("--learning_rate", type=float, default=5e-5)
        parser.add_argument("--adam_epsilon", type=float, default=1e-8)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--load_path", help="The path to the generator model")

    @classmethod
    def modify_weights(cls, weights: Tensor) -> Tensor:
        """Adopt old weights"""
        return weights

    @classmethod
    def load_model(cls, raw_model, load_path, map_location):
        logger.info(f"Load model from {load_path}")
        weights: OrderedDict = torch.load(load_path,
                                          map_location=torch.device(map_location))
        weights = cls.modify_weights(weights)
        raw_model.load_state_dict(weights)
        return raw_model


def init_run_dir(bookmark: Optional[str] = None) -> str:
    from datetime import datetime

    timestr = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if bookmark:
        timestr += "_" + bookmark
    run_dir = os.path.join('ckpts', timestr)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def setup_logging(run_dir: str):
    log_fname = "train"
    if is_distributed():
        log_fname += f"_{local_rank()}"
    log_fname += ".log"
    fh = logging.FileHandler(os.path.join(run_dir, log_fname), "w", "utf-8")
    sh = logging.StreamHandler()
    fmt = "[%(asctime)s] [%(module)s] [%(levelname)s] %(message)s"
    if is_distributed():
        fmt = f"[%(asctime)s] [Rank_{rank()}] [%(module)s] [%(levelname)s] %(message)s"
    logging.basicConfig(
        format=fmt,
        datefmt="%Y-%m-%d_%H:%M:%S",
        level=logging.DEBUG,
        handlers=[fh, sh]
    )


def setup_gpu(device_ids: List[int], occupy=False) -> Tuple[int, List[int]]:
    if not torch.cuda.is_available():
        logger.error("cuda is not available")
        exit(-1)

    logger.info("GPU count = %d", torch.cuda.device_count())
    if is_distributed():
        logger.info(f"Distributed training: rank {rank()} world_size {world_size()} local_rank {local_rank()}")

        _device_ids = validate_device_ids(device_ids)
        local_device_id = _device_ids[local_rank()]

        torch.cuda.set_device(local_device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{local_device_id}"

        _device_ids = [local_device_id]
        _device = local_device_id
        if occupy:
            occupy_mem(local_device_id)

        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # NCCL_BLOCKING_WAIT should be set
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=timedelta(hours=12))

    else:
        _device_ids = validate_device_ids(device_ids)
        _device = _device_ids[0]

        if occupy:
            for device in _device_ids:
                occupy_mem(device)

    return _device, _device_ids


def train_dataloader(train_dataset, batch_size, num_workers):
    if is_distributed():
        sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    else:
        sampler = RandomSampler(train_dataset)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    return dataloader, sampler


def eval_dataloader(dataset, batch_size, num_workers):
    if is_distributed():
        sampler = DistributedSampler(dataset, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    return dataloader, sampler


def gather_all(object) -> List:
    all_objects = [None for _ in range(dist.get_world_size())]
    if isinstance(object, Tensor):
        dist.all_gather(all_objects, object)
    else:
        dist.all_gather_object(all_objects, object)
    return all_objects
