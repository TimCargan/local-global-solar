from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from zephyrus.utils.hyperparameters import HyperParameters_Extend


@dataclass
class FeatureExtract:
    """
    Extract the correct feature width
    """
    hp: HyperParameters_Extend
    extra_steps: int = 2
    y_key: str = "pred"

    def __call__(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return self.slice(ds)

    def slice(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = ds.map(lambda x, y: self._slice((x, y)), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        return ds

    def v_slice(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = ds.map(lambda x, y: tf.vectorized_map(self._slice, (x, y)), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        return ds

    def _slice(self, xy):
        x, y = xy[0], xy[1]
        key = sorted(x.keys())
        nx = {f: x[f][:self.hp.get("WARMUP_STEPS") + self.hp.get("OUTPUT_STEPS") + self.extra_steps] for f in key}
        nx["Latitude"] = x["Latitude"][0]
        nx["Longitude"] = x["Longitude"][0]

        nx["out_ts"] = x["ts"][self.hp.get("OUTPUT_STEPS_START") : self.hp.get("OUTPUT_STEPS_END")]
        nx["plant"] = x["plant"][self.hp.get("OUTPUT_STEPS_START") : self.hp.get("OUTPUT_STEPS_END")]

        nx["irradiance_in"] = tf.cast(x["irradiance"], tf.float32)[:self.hp.get("WARMUP_STEPS")]
        nx["irradiance_in_kn"] = x.get("kn_irradiance", (nx["irradiance_in"] * np.NAN))
        nx["irradiance_in_kn"] = tf.cast(nx["irradiance_in_kn"], tf.float32)[:self.hp.get("WARMUP_STEPS")]

        # Set irrad_in to -1 if ex irrad
        if self.hp.get("EX_IRRAD"):
            nx["irradiance_in"] = tf.zeros_like(nx["irradiance_in"]) - 1.0

        y = y[self.hp.get("OUTPUT_STEPS_START"): self.hp.get("OUTPUT_STEPS_END")]
        y = {self.y_key: y}
        return (nx, y)



