import os
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import unquote

import tensorflow as tf
from absl import flags, logging

from zephyrus.data_pipelines.transformers import Pass
from zephyrus.utils.runner import BaseRunner, TrainTest, TrainTestL, DataLoader
from zephyrus.utils.translator import get_path

flags.DEFINE_string('output_dir', default="./res", help="Folder to save results")
flags.DEFINE_bool('per_plant', default=False, help="Run per plant or all at once")
flags.DEFINE_integer('plant_index', default=None, help="Run for a given, if plant per_plant ignored")
flags.DEFINE_bool('gpp', default=False, help="Run Global++")
flags.DEFINE_bool('cv', default=False, help="Split data into plant folds")
flags.DEFINE_bool('run_all_modes', default=False, help="Run all exper modes")
flags.DEFINE_integer('fold_index', default=None, help="Run on a given fold")
FLAGS = flags.FLAGS

"""" Consts """
PLANTS = ["17314", "1005", "56963", "862", "918", "56424", "384", "534", "643", "1395", "212", "1190",
          "458", "1467", "55827", "440", "235", "471", "1007", "1161"]

FOLDS = [[1190, 1395, 1005, 17314],
         [458, 471, 918, 212],
         [1467, 56424, 862, 1161],
         [384, 643, 1007, 56963],
         [534, 55827, 235, 440]]

MODE_CODE = {"pass": 0, "local": 1, "global": 2, "cv": 3, "kn": 4, "global++": 5}

# Data File
time_range = "2015-2021"
FOLD_DATASET_FILE = f"point_newirrad_weather_({time_range})_FOLD_TTSplit2019-05-01.tfdata.gz"
PLANT_DATASET_FILE = f"point_newirrad_weather_({time_range})PP_TTSplit2019-05-01.tfdata.gz"
GLOBAL_DATASET_FILE = f"point_newirrad_weather_({time_range})GS_TTSplit2019-05-01.tfdata.gz"
GLOBALPP_DATASET_FILE = f"point_newirrad_weather_({time_range})TTSplit2019-05-01.tfdata.gz"


def plant_glob(plant: str = "*"):
    return os.path.join(get_path("data"), PLANT_DATASET_FILE, f"plant={plant}")


def fold_glob(fold: str = "*"):
    return os.path.join(get_path("data"), FOLD_DATASET_FILE, f"fold={fold}")


@dataclass
class ExperimentRunner:
    experiment: BaseRunner
    data_loader: DataLoader

    @property
    def output_dir(self) -> str:
        out_dir = FLAGS.output_dir
        res_dir = get_path("results")
        out_dir = os.path.join(res_dir, out_dir)
        return out_dir

    @property
    def sp(self) -> str:
        sp = os.path.join(get_path("data"), GLOBAL_DATASET_FILE)
        if FLAGS.gpp:
            sp = os.path.join(get_path("data"), GLOBALPP_DATASET_FILE)
        return sp


    def _run_fold(self, fold, dsl: TrainTestL, output_dir_suffix: str = None):
        logging.info(f"Running for fold: {fold}")
        output_dir = self.output_dir
        if output_dir_suffix:
            output_dir = os.path.join(output_dir, output_dir_suffix)
        logging.info(f"Saving results to: {output_dir}")
        self.experiment.run(dsl, output_dir, filename=f"fold_{fold}")

    def _run_on_dict(self, list_dict: dict[Any, TrainTestL], output_dir_suffix: str = None):
        for el in list_dict.items():
            fold, dsl = el
            self._run_fold(fold, dsl, output_dir_suffix)

    def _read_fn_build(self, path: str, mode_code:int) -> Callable[[], TrainTest]:
        # Log so we know what files we might be touching
        logging.info(f"Defer reading data from {path}")

        def _f():
            logging.info(f"Reading data from {path}")
            return self.data_loader.make_test_train_data_set(path=path, mode_code=mode_code)

        return _f

    def make_fold_dict_filter(self) -> dict[str, TrainTestL]:
        # Extract PSB for use in the helper function
        test_psb = self.data_loader.test_psb
        train_psb = self.data_loader.train_psb

        def _filter(train_test: TrainTest, fold_plants):
            logging.info(f"Defer filtering {fold_plants}")

            def _f():
                logging.info(f"Filtering plants {fold_plants}")
                train, test = train_test
                _fold_plants = tf.constant(fold_plants, dtype=tf.int64)
                ftrain = train.filter(lambda x, y: tf.reduce_any(_fold_plants != x["plant"][0])).apply(train_psb)
                ftest = test.filter(lambda x, y: tf.reduce_any(_fold_plants == x["plant"][0])).apply(test_psb)
                return ftrain, ftest

            return _f

        # Read data using a new PSB with no test train PSBs
        path = os.path.join(get_path("data"), GLOBAL_DATASET_FILE)
        data_loader = self.data_loader.replace(test_psb=Pass(), train_psb=Pass())
        train_test = data_loader.make_test_train_data_set(path=path, mode_code=MODE_CODE["cv"])
        logging.info(f"Making fold datasets by filtering data from {path}")

        dataset = {}
        for i in range(5):
            fold_plants = FOLDS[i]
            dataset[str(i)] = _filter(train_test, fold_plants)
        return dataset

    def make_fold_index_data_set(self, fold_index: int) -> TrainTestL:
        """ Hard coded for 5 fold"""
        folds = list(range(5))
        in_fold = folds[:fold_index] + folds[fold_index + 1:]
        train_path = fold_glob(fold=str(in_fold))
        test_path = fold_glob(fold=str(fold_index))

        test_f = self._read_fn_build(test_path, mode_code=MODE_CODE["cv"])
        train_f = self._read_fn_build(train_path, mode_code=MODE_CODE["cv"])

        # Wrap in another func, so we can defer the reads.
        # For now, we just reread each fold. in theory this could be fixed  but is a fairly low performance overhead
        def _read():
            train, _ = train_f()
            _, test = test_f()
            return train, test

        return _read

    def make_folds_data_set(self, num_folds: int = 5) -> dict[str, TrainTestL]:
        """
        Make a dict of functions that when called will read in the data needed to each fold
        We have to use a named function as lambdas get funny about referencing variables
        Returns: Dict of [plantID, callable -> (Train,Test)]
        """
        datasets = {}
        for f in range(num_folds):
            datasets[str(f)] = self.make_fold_index_data_set(f)
        return datasets

    def make_plants_data_set(self) -> dict[str, TrainTestL]:
        """
        Make a dict of functions that when called will read in the data needed to train the given plant
        We have to use a named function as lambdas get funny about referencing variables
        Returns: Dict of [plantID, callable -> (Train,Test)]
        """
        datasets = {}
        for i, p in enumerate(PLANTS):
            datasets[p] = self.make_plant_index_data_set(i, mode_code=MODE_CODE["local"])
        return datasets

    def make_plant_index_data_set(self, plant_index: int, mode_code: int) -> TrainTestL:
        plant_str = PLANTS[plant_index]
        plant_str = unquote(plant_str)
        path = plant_glob(plant=plant_str)
        return self._read_fn_build(path, mode_code=mode_code)

    def _run_all_modes(self):
        logging.info("Running all Modes")
        assert self.kn_irrad == False, "Cant run KN irrad on all"
        assert self.eval_only == False, "Cant run eval_only on all"

        # Local
        self.test_psb.cache = True
        self.train_psb.cache = True
        logging.info(f"Fitting per plants")
        plant_dataset = self.make_plants_data_set()
        self._run_on_dict(plant_dataset, output_dir_suffix="mode=Local")
        self.test_psb.cache = False
        self.train_psb.cache = False

        # Global
        logging.info(f"Fitting Global")
        path = os.path.join(get_path("data"), self.global_dataset_file)
        dsl = self._read_fn_build(path=path)
        self._run_fold("global", dsl, output_dir_suffix="mode=Global")

        # Global++
        logging.info(f"Fitting Global++")
        path = os.path.join(get_path("data"), self.globalPP_dataset_file)
        dsl = self._read_fn_build(path=path)
        self._run_fold("globalPP", dsl, output_dir_suffix="mode=Global++")

        # CV - Can run with KN_Irrad
        logging.info(f"Fitting per fold (cv)")
        self.kn_irrad = False
        fold_datasets = self.make_fold_dict_filter()
        self._run_on_dict(fold_datasets, output_dir_suffix="mode=CV")

        # KN - Use CV output for faster training
        logging.info(f"Fitting per fold (KN)")
        self.kn_irrad = self.eval_only = True
        logging.debug(f"Set kn_irrad and eval_only True")
        FLAGS.model_dir = os.path.join(get_path("results"), FLAGS.output_dir, "mode=CV")
        logging.debug(f"Set model_dir: %s", FLAGS.model_dir)
        fold_datasets = self.make_fold_dict_filter()
        self._run_on_dict(fold_datasets, output_dir_suffix="mode=KN")
        self.kn_irrad = self.eval_only = False
        logging.debug(f"Set kn_irrad and eval_only False")

    def run(self):
        if FLAGS.run_all_modes:
            self._run_all_modes()
        # Local
        elif FLAGS.per_plant:
            logging.info("Setting up plants datasets")
            logging.info(f"Reading data from {PLANT_DATASET_FILE}")
            plant_dataset = self.make_plants_data_set()
            logging.info(f"Fitting per plants")
            self._run_on_dict(plant_dataset)
        # CV - Can run with KN_Irrad
        elif FLAGS.cv:
            logging.info(f"Fitting per fold (cv)")
            fold_datasets = self.make_fold_dict_filter()  # self.make_folds_data_set()
            self._run_on_dict(fold_datasets)
        # Local - with index, used in sbatch to parralise jobs
        elif FLAGS.plant_index is not None:
            logging.info(f"Plant index set: {FLAGS.plant_index}")
            dsl = self.make_plant_index_data_set(FLAGS.plant_index, mode_code=MODE_CODE["local"])
            self._run_fold(FLAGS.plant_index, dsl)
        # Local - with index, used in sbatch to parralise jobs
        elif FLAGS.fold_index is not None:
            logging.info(f"Fold index set: {FLAGS.fold_index}")
            dsl = self.make_fold_index_data_set(FLAGS.fold_index)
            self._run_fold(FLAGS.fold_index, dsl)
        # Global
        else:
            all_datasets = lambda: self.data_loader.make_test_train_data_set(path=self.sp, mode_code=MODE_CODE["global"])
            logging.info(f"Fitting on all")
            self.experiment.run(all_datasets, self.output_dir)
