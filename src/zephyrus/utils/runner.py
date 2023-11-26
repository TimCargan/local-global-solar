import functools
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Callable

import pandas as pd
import tensorflow as tf
from absl import flags, logging

from zephyrus.data_pipelines import read_glob
from zephyrus.data_pipelines.transformers import PSB, KNExtract
from zephyrus.utils import FromDict
from zephyrus.utils.hyperparameters import HyperParameters_Extend
from zephyrus.utils.standard_logger import build_logger

flags.DEFINE_string('name', default=None, help="Runner Name, included in logs")
flags.DEFINE_string('model_dir', default=None, help="Location to check for checkpoints, if in eval only mode")
flags.DEFINE_integer("verbose", default=1, help="Keras Log verbosity, see Keras docs")
flags.DEFINE_bool('min_date', default=False, help="filter from before 2015")
flags.DEFINE_bool('eval_train', default=False, help="Save predictions from  training data")
flags.DEFINE_bool('eval_only', default=False,
                  help="Only Eval model, helpful if we want to rerun eval, maybe with other data")
flags.DEFINE_bool('kn_irrad', default=False, help="irrad x as nearest plant not in fold y")

FLAGS = flags.FLAGS

TrainTest = Tuple[tf.data.Dataset, tf.data.Dataset]
TrainTestL = Callable[[], TrainTest]


@dataclass
class BaseRunner(ABC, FromDict):
    runner: str = ""
    hp: HyperParameters_Extend = None
    hyper_opt: bool = False
    output_dir: str = ""
    model_dir: str = None
    name: str = ""
    drop_remainder: bool = False
    eval_only: bool = False
    eval_train: bool = False,
    verbose: int = 1

    def __post_init__(self):
        # Logging
        name = __name__ if self.name is None else self.name
        if self.verbose != 1:
            tf.keras.utils.disable_interactive_logging()
        self.logger = build_logger(name)

    @abstractmethod
    def make_model(self, data_spec: tf.data.DatasetSpec) -> tf.keras.Model:
        pass

    @abstractmethod
    def fit_model(self, m: tf.keras.Model, d: tf.data.Dataset, t: tf.data.Dataset, output_dir):
        return m.fit(d)

    @abstractmethod
    def eval_model(self, model, test, output_dir) -> pd.DataFrame:
        """
        Use the given test data to evaluate the model and return its predictions in a pd.Dataframe
        """
        pass

    def eval_summary(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        pass

    def settings_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k[0] != "_" and k != "logger"}

    def save_settings(self, output_folder: str) -> None:
        logging.info(f"Save Settings -- {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
        out_file = os.path.join(output_folder, "_settings.json")
        with open(out_file, "w") as f:
            settings_dict = self.settings_dict()
            settings = json.dumps(settings_dict, skipkeys=True, indent=4, sort_keys=True, default=lambda o: o.__dict__)
            f.write(settings)

    def save_df(self, df: pd.DataFrame, output_folder: str, filename: str = "res.pqt",
                out=pd.DataFrame.to_parquet) -> None:
        logging.info(f"Save -- {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
        out_file = os.path.join(output_folder, filename)
        logging.info(f"Writing -- {out_file}")
        out(df, out_file)

    def run(self, dataset_f: TrainTestL, output_dir: str, filename: str = "res"):
        """
        Run a standard train eval experiment with the gien data saving outputs to the folder
        :param dataset_f:
        :param output_dir:
        :param filename:
        :return:
        """
        train, test = dataset_f()
        ckpt_dir = os.path.join(output_dir, f"_ckpt_{filename}")

        logging.info(f"Building Model -- {ckpt_dir}")
        model = self.make_model(train.element_spec)
        self.save_settings(ckpt_dir)

        if not self.eval_only:
            logging.info(f"Fitting -- {ckpt_dir}")
            self.fit_model(model, train, test, ckpt_dir)

        if self.eval_only and self.model_dir is not None:
            ckpt_dir = os.path.join(self.model_dir, f"_ckpt_{filename}")

        if self.eval_train:
            logging.info(f"Eval Train -- {output_dir}")
            res = self.eval_model(model, train, None)
            self.save_df(res, output_dir, filename=f"_{filename}_train_res.pqt")
            summary_dict = self.eval_summary(res)
            for s_name, summary in summary_dict.items():
                self.save_df(summary, output_dir,
                             filename=f"_{filename}_{s_name}_train_res.csv", out=pd.DataFrame.to_csv)

        logging.info(f"Deleting training data")
        del train

        logging.info(f"Eval Test -- Model Location: {ckpt_dir}")
        res = self.eval_model(model, test, ckpt_dir)
        self.save_df(res, output_dir, filename=f"{filename}.pqt")
        summary_dict = self.eval_summary(res)
        for s_name, summary in summary_dict.items():
            self.save_df(summary, output_dir, filename=f"{filename}_{s_name}_summary.csv", out=pd.DataFrame.to_csv)






@dataclass
class DataLoader(ABC, FromDict):
    kn_irrad: bool = False
    min_date: bool = False

    vectorize: bool = True

    min_date_filter_ts: int = 1420070400  # 2015-01-01 00:00 UTC
    test_train_split_ts: int = 1556668800  # 2019-05-01 00:00 UTC

    train_psb: PSB = field(default_factory=lambda : PSB(batch_size=FLAGS.batch_size, cache=False, prefetch_buff=2,
                         shuffle_buff=2 ** 14, drop_remainder=False, pad=FLAGS.pad_psb))

    test_psb: PSB = field(default_factory=lambda : PSB(batch_size=FLAGS.batch_size, cache=False, prefetch_buff=2,
                          drop_remainder=False, pad=FLAGS.pad_psb))

    _opts = tf.data.Options()
    _opts.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    _opts.deterministic = False

    @abstractmethod
    def feature_extract(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds

    def make_test_train_data_set(self, path: str, filters: List = None, train_filters=None,
                                 test_filters=None, mode_code=0) -> TrainTest:
        # Set default empy lists
        if test_filters is None:
            test_filters = []
        if train_filters is None:
            train_filters = []
        if filters is None:
            filters = []

        def add_mode(code: int):
            def f(x, *y):
                x["mode"] = tf.where(x["plant"] == 0, 0, code)
                return x, *y
            return f

        # Add min date filter if needed
        if self.min_date:
            filters.append(lambda ds: ds.filter(lambda x, *_: x["ts"][0] > self.min_date_filter_ts))

        # Apply transformations
        train = read_glob(os.path.join(path, "train"))
        train = train.with_options(self._opts)
        train = functools.reduce(tf.data.Dataset.apply, filters + train_filters, train)
        train = train.apply(self.feature_extract).map(add_mode(mode_code)).apply(self.train_psb)

        test = read_glob(os.path.join(path, "test"))
        test = test.with_options(self._opts)
        test = functools.reduce(tf.data.Dataset.apply, filters + test_filters, test)
        if self.kn_irrad:
            logging.info("Applying KN Extract to test")
            test = test.apply(KNExtract())
        test = test.apply(self.feature_extract).map(add_mode(mode_code)).apply(self.test_psb)

        return train, test
