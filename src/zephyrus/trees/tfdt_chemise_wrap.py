from dataclasses import dataclass, field

import jax.tree_util
import numpy as np
import tensorflow as tf
from absl import flags
from absl import logging as absl_logging

flags.DEFINE_integer("tree_batch_size", default=1024, help="Batch size for trees to speed them up")


FLAGS = flags.FLAGS


@dataclass
class TFDTChemise:
    threads: int
    output_steps: int
    input_feats: list[str]


    tree_settings = {"num_trees": 20, "min_examples": 2, "max_depth": 32, "verbose": 0}

    # needed to conform to the API
    callbacks: list = field(default=list)
    state = None

    def make_model(self):
        # Defer import to here
        import tensorflow_decision_forests as tfdf
        # Defer import for compat with TF
        absl_logging.info(f"Using {self.threads} threads")
        trees = {}
        for m in range(self.output_steps):
            absl_logging.info(f"Making model {m}")
            model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION, num_threads=self.threads, check_dataset=False,
                                                 **self.tree_settings)
            model.compile()
            trees[m] = model

        self._m = trees
        return

    def feat_extract(self, x, y, *z):
        nx = {k: x[k] for k in self.input_feats}
        ny = y["pred"][FLAGS.warmup_steps:]
        return (nx, ny)

    def fit(self, d: tf.data.Dataset, **kwargs):
        absl_logging.info(f"Starting to fit using {self.threads} threads")
        try:
            d = d.map(self.feat_extract)
            for output_step, t in self._m.items():
                absl_logging.info(f"Fitting step {output_step}")
                sd = d.map(lambda x, y: (x, y[output_step]))
                sd = sd.batch(FLAGS.tree_batch_size, drop_remainder=False)
                t.fit(sd)

        except Exception as e:
           absl_logging.error("Fit Failed")
           absl_logging.error(e)
           raise e
        else:
            absl_logging.info("fit done")
        return

    def predict_batch(self, x):
        outputs = []
        for output_step, t in self._m.items():
            pred = t(x)
            outputs.append(pred)
        concated = tf.concat(outputs, axis=-1)
        return concated

    def map_model(self, data: tf.data.Dataset):
        for xyz in data.as_numpy_iterator():
            y_hat = self.predict_batch(xyz[0]).numpy()
            # Pad the negative steps with Nans, so it's the same shape as the DNNs
            nan_array = np.empty((y_hat.shape[0], FLAGS.warmup_steps))
            nan_array.fill(np.NAN)
            y_hat = np.concatenate([nan_array, y_hat], axis=-1)
            output = *xyz, y_hat
            p_batch = jax.tree_util.tree_map(lambda x: np.expand_dims(x, 0), output)
            yield p_batch
