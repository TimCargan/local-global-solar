import math
from typing import NamedTuple

import jax
import tensorflow as tf

ZScore = NamedTuple("ZScore", [("name", str), ("mean", float), ("std", float)])

log = ['ghi']
zscore = [ZScore("clouds", 60, 30), ZScore("rh", 82, 13), ZScore("pres", 1000, 15.5),
          ZScore("precip", 0.1, 0.33), ZScore("temp", 283, 5.5)]
clip_min = 3.0
clip_max = 9.0

class Normalizer:
    # Cols to transform

    def __call__(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        Args:
            ds:
        Returns:
        """
        # Not sure if its better to do two applies or apply(F(G(X)))
        return ds.map(self._norm, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    @staticmethod
    def _zscore(v, mean, std):
        return (v - mean) / std

    @staticmethod
    def _log(v):
        v = tf.cast(v, dtype=tf.float32)
        return tf.math.log1p(tf.math.maximum(v, 0))

    @staticmethod
    def _un_log(v):
        v = tf.cast(v, dtype=tf.float32)
        return tf.math.expm1(v)

    @staticmethod
    def _norm_irrad(v, clip=True):
        v = tf.cast(v, dtype=tf.float32)
        v = tf.math.maximum(v, 0) # Scale any negative values to 0 for safety
        v = tf.math.log1p(v) # Log1p scale irrad, falls in the range of 0 - ~8.5
        # Clip all values less than 3.0  ~ 20, and set a max of 9
        v = tf.clip_by_value(v, clip_value_min=clip_min, clip_value_max=clip_max) if clip else v
        v = v - 4.0 # Set irrad to be about [-3, +3]
        return v

    @staticmethod
    def _unnorm_irrad(v, clip=True):
        v = tf.cast(v, dtype=tf.float32) # Needed so expm1 is safe
        v = v + 4.0 # Move values back to min
        # Keep value in clip range, all values less than 3.0  ~ 20, and set a max of 9

        v = tf.clip_by_value(v, clip_value_min=clip_min, clip_value_max=clip_max) if clip else v
        v = tf.math.expm1(v)
        return v

    @classmethod
    def _norm(self, x, y):
        key = sorted(x.keys())
        x = {f: x[f] for f in key}

        # Normalize values
        x["elev_angle"] = tf.math.sin(x["elev_angle"] * (math.pi / 180))

        for n in log:
            x[n] = self._log(x[n])

        for n, m, s in zscore:
            x[n] = self._zscore(x[n], m, s)

        # Cast to plants to ints so XLA works
        x["plant"] = tf.strings.to_number(x["plant"], out_type=tf.int64)

        x["irradiance"] = self._norm_irrad(x["irradiance"], clip=True)
        x["irradiance_in"] = self._norm_irrad(x["irradiance_in"], clip=True)
        x["irradiance_in_kn"] = self._norm_irrad(x["irradiance_in_kn"], clip=True)

        y = jax.tree_util.tree_map(lambda x: self._norm_irrad(x, clip=True), y)
        return x, y



