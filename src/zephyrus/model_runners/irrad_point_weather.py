import math
import tensorflow as tf
from absl import flags, logging
from dataclasses import dataclass, field

from zephyrus.data_pipelines.transformers import EU_Images, PSB
from zephyrus.data_pipelines.transformers.feature_extract import FeatureExtract
from zephyrus.data_pipelines.transformers.normalize import Normalizer
from zephyrus.utils.hyperparameters import HyperParameters_Extend as HP
from zephyrus.utils.runner import DataLoader

FLAGS = flags.FLAGS

flags.DEFINE_float("last_hour_sub_prob", default=0.0, help="The probability of substituting irrad_in[t] for irrad_in[t-1] ")
flags.DEFINE_float("kn_sub_prob", default=0.0, help="The probability of substituting irrad_in[t] for irrad_kn[t] ")

@dataclass
class IrradPointWeather(DataLoader):

    train_psb: PSB = field(default_factory=lambda : PSB(batch_size=FLAGS.batch_size, cache=True, prefetch_buff=2,
                         shuffle_buff=2 ** 14, drop_remainder=False))

    test_psb: PSB = field(default_factory=lambda : PSB(batch_size=FLAGS.batch_size, cache=True, prefetch_buff=2,
                          drop_remainder=False))


    def __post_init__(self):
        hp = HP()
        self.hp = hp
        # super(IrradPointWeather, self).__init__(**kwargs)

        self.hp.Fixed("NUM_EPOCHS", FLAGS.num_epochs)
        self.hp.Config("BATCH_SIZE", FLAGS.batch_size)
        # self.hp.Config("LEARNING_RATE", learning_rate)
        self.hp.Config("LIMIT_EVAL", False)

        self.hp.Config("CACHE_INPUT", True)
        self.hp.Config("INCLUDE_WEATHER", True)
        self.hp.Config("EX_IRRAD", False)
        self.hp.Config("IMG_SIZE", 16)
        self.hp.Config("WARMUP_STEPS", FLAGS.warmup_steps)
        self.hp.Config("OUTPUT_STEPS", FLAGS.output_steps)
        self.hp.Config("PRED_OFFSET", 0)
        self.hp.Config("OUTPUT_STEPS_START", hp.get("PRED_OFFSET"))  # Overlap step 0 with the warmup
        self.hp.Config("OUTPUT_STEPS_END", hp.get("PRED_OFFSET") + hp.get("WARMUP_STEPS") + hp.get("OUTPUT_STEPS"))


    def feature_extract(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        logging.log_first_n(logging.INFO, self.hp.__dict__(), 1)
        logging.debug(f"Cardinality before opps: {ds.cardinality()}")
        ds = ds.apply(FeatureExtract(hp=self.hp))
        ds = ds.apply(Normalizer())


        # Sort keys
        ds = ds.map(lambda x, *y: ({f: x[f] for f in sorted(x.keys()) if f not in ["kn_irradiance", "DOY", "Hour"]}, *y),
                    num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        # Helper functions to simulate messy data if flags are set
        def substitute(original, sub, sub_prob):
            """Randomly merge two arrays of the same shape.

            Args:
                original: The original array
                sub: Values to substitute in
                sub_prob: the probability of a new value being used

            Returns:
                An array of the same shape as original
            """
            rnd = tf.random.categorical(tf.math.log([[1 - sub_prob, sub_prob]]), num_samples=original.shape[0])[0]
            new_array = tf.where(rnd != 1, original, sub)
            return new_array

        def ffil(x):
            """Forward fill nan in the array.
            Args:
                x: Array to forward fill

            Returns:
                Array of values with strings of nans replaced by the preceding value
            """
            # Find non-NaN values
            mask = ~tf.math.is_nan(x)
            # Take non-NaN values and precede them with a NaN
            values = tf.concat([[math.nan], tf.boolean_mask(x, mask)], axis=0)
            # Use cumsum over mask to find the index of the non-NaN value to pick
            idx = tf.cumsum(tf.cast(mask, tf.int64))
            # Gather values
            x = tf.gather(values, idx)
            return x

        def carry_forward(x, *y):
            """Apply Carry forward, doesn't apply to ts_0.

            Args:
                x: A tensor to randomly apply ffil to

            Returns:
                Tensor of same shape as x with random subs
            """
            def _carry_forward(x):
                norm = x[1:]
                sub = tf.zeros_like(norm) / 0  # array of Nan
                sub_prob = FLAGS.last_hour_sub_prob
                new_tail = substitute(norm, sub, sub_prob)
                x = tf.concat([x[:1], new_tail], axis=0)
                x = ffil(x)
                return x
            """Apply carry forward function to irrad in"""
            x["irradiance_in"] = _carry_forward(x["irradiance_in"])
            return x, *y

        def kn_sub(x, *y):
            """Apply carry forward function to irrad in"""
            sub_prob = FLAGS.kn_sub_prob
            x["irradiance_in"] = substitute(x["irradiance_in"], x["irradiance_in_kn"], sub_prob)
            return x, *y

        if FLAGS.kn_sub_prob > 0:
            """ IF the prob is set, apply the transform"""
            ds = ds.map(kn_sub, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        if FLAGS.last_hour_sub_prob > 0:
            """ IF the prob is set, apply the transform"""
            ds = ds.map(carry_forward, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        if self.vectorize:
            ds = ds.apply(EU_Images.group)
            ds = ds.map(EU_Images.sort_group, num_parallel_calls=8, deterministic=False)
            ds = ds.map(EU_Images.padd_sorted_group, num_parallel_calls=8, deterministic=False)

        return ds
