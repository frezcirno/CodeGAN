import logging
import os
from typing import Literal
import torch
import numpy as np
import pandas as pd
import _pickle as cPickle
from pandas.io.parquet import to_parquet
from .dist import is_master

logger = logging.getLogger(__name__)


def cache_call(path, func, format="parquet"):
    def pickle_dump(obj, path):
        with open(path, "wb") as f:
            cPickle.dump(obj, f)

    def pickle_load(path):
        with open(path, "rb") as f:
            return cPickle.load(f)

    FORMAT = {
        "parquet": (".parquet", to_parquet, pd.read_parquet),
        "numpy": (".npy", lambda obj, path: np.save(path, obj), np.load),
        "torch": (".bin", torch.save, torch.load),
        "pickle": (".pkl", pickle_dump, pickle_load),
    }

    postfix, saver, loader = FORMAT[format]

    path += postfix

    def new_func(*args, **kwargs):
        if not os.path.exists(path):
            res = func(*args, **kwargs)
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            saver(res, path)
            return res
        return loader(path)

    return new_func


def remember_result(func):
    def new_func():
        scope = func
        if hasattr(scope, "res"):
            return scope.res
        scope.res = res = func()
        return res
    return new_func


def cache_result(path: str, format: Literal['parquet|numpy|torch|pickle'] = "parquet"):
    def pickle_dump(obj, path):
        with open(path, "wb") as f:
            cPickle.dump(obj, f)

    def pickle_load(path):
        with open(path, "rb") as f:
            return cPickle.load(f)

    FORMAT = {
        "parquet": (".parquet", to_parquet, pd.read_parquet),
        "numpy": (".npy", lambda obj, path: np.save(path, obj), np.load),
        "torch": (".bin", torch.save, torch.load),
        "pickle": (".pkl", pickle_dump, pickle_load),
    }

    postfix, saver, loader = FORMAT[format]

    path = str(path)
    if not path.endswith(postfix):
        path += postfix

    def decorate(func):
        def new_func(*args, **kwargs):
            if os.path.exists(path):
                logger.info(f"load data from cache {path}")
                return loader(path)
            res = func(*args, **kwargs)
            if not is_master():
                return res
            try:
                if os.path.dirname(path):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                saver(res, path)
            except Exception as ex:
                logger.error(f"cache_result failed: {ex}")
            return res
        return new_func
    return decorate
