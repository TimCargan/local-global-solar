import os
from typing import Union

import numpy as np
import tensorflow as tf
from absl import logging
from jax.tree_util import tree_map

from zephyrus.data_pipelines.transformers.eumetsat_transform import EU_Images


class PSB:
    """
    Prefretch Shuffle Batch
    A standard final layer transformation
    """

    def __init__(self, batch_size: int = None, cache: bool = False, drop_remainder: bool = True, pad: bool = False,
                 shuffle_buff: int = None, reshuffle_each_iteration: bool = True, prefetch_buff: Union[int, None] = tf.data.AUTOTUNE):
        self.batch_size = batch_size
        self.shuffle_buff = shuffle_buff
        self.reshuffle_each_iteration = reshuffle_each_iteration
        self.drop_remainder = drop_remainder
        self.cache = cache
        self.prefetch_buff = prefetch_buff
        self.pad = pad


    def __call__(self, ds: tf.data.Dataset):
        if self.cache:
            ds = ds.cache()
        if self.shuffle_buff:
            ds = ds.shuffle(self.shuffle_buff, reshuffle_each_iteration=self.reshuffle_each_iteration)
        if self.batch_size and self.pad:
            self.drop_remainder = True
            zero = tree_map(lambda x: np.zeros(shape=x.shape, dtype=x.dtype.as_numpy_dtype), ds.element_spec)
            pad = tf.data.Dataset.from_tensors(zero).repeat(self.batch_size)
            logging.info("Padded and set drop remainder to True")
            ds = ds.concatenate(pad)
        if self.batch_size:
            cores = os.environ.get("SLURM_CPUS_PER_TASK", "0")
            cores = max(int(cores) - 6 , 8)  # Use 2 less incase the threadpool is smaller. Might want to tune this a bit
            ds = ds.batch(self.batch_size, drop_remainder=self.drop_remainder, num_parallel_calls=cores, deterministic=False)
        if self.prefetch_buff is not None:
            ds = ds.prefetch(self.prefetch_buff)
        return ds

    def __dict__(self):
        rep = {"batch_size": self.batch_size,
                "shuffle_buff": self.shuffle_buff,
                "drop_remainder": self.drop_remainder
                }
        return rep


class Pass:
    def __call__(self, ds: tf.data.Dataset):
        return ds

    @staticmethod
    def __dict__():
        return {}


class KNExtract:
    def _kn_extract(self, x, y):
        nx = {f: v for f, v in x.items()}
        nx["irradiance"] = x["kn_irradiance"]
        return nx, y

    def __call__(self, ds: tf.data.Dataset):
        ds = ds.map(self._kn_extract, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        return ds

    @staticmethod
    def __dict__():
        return {}


class ParallelUnbatch:
    def __call__(self, ds:tf.data.Dataset):
        def _to_ds(xs, ys):
            return tf.data.Dataset.from_tensor_slices((xs, ys))
        ds = ds.interleave(_to_ds, num_parallel_calls=tf.data.AUTOTUNE, block_length=8, deterministic=False)
        return ds