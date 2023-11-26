import dataclasses
import datetime
import os
import pathlib
import shutil
from dataclasses import dataclass
from typing import Any

import jax
import mlflow
import numpy as np
import tensorflow as tf
from absl import flags, logging

from hemera.push_notify import push_job_run
from zephyrus.data_pipelines.transformers import PSB, Pass
from zephyrus.utils import FromDict
from zephyrus.utils.runner import BaseRunner, DataLoader, TrainTestL
from zephyrus.utils.translator import get_path

flags.DEFINE_string('output_dir', default="./res", help="Folder to save results")
flags.DEFINE_bool('per_plant', default=False, help="Run per plant or all at once")
flags.DEFINE_integer('plant_index', default=None, help="Run for a given, if plant per_plant ignored")
flags.DEFINE_bool('gpp', default=False, help="Run Global++")
flags.DEFINE_bool('cv', default=False, help="Split data into plant folds")
flags.DEFINE_bool('run_all_modes', default=False, help="Run all exper modes")
flags.DEFINE_integer('fold_index', default=None, help="Run on a given fold")
flags.DEFINE_integer('threads', default=16, help="Number of threads per model")

flags.DEFINE_boolean("vec_split_lg", default=False, help="Train local and glob separately")
flags.DEFINE_boolean("vec_run_local", default=True, help="If vec split, run local models")
flags.DEFINE_boolean("vec_run_gcv", default=True, help="If vec split, run Global and CV")
flags.DEFINE_integer("vec_shuff_buf_scale", default=8, help="Amount to scale the shuffle buffer pre batch in vec load")

flags.DEFINE_integer("local_vec_size", default=20, help="Number of local plants to use")
flags.register_validator('local_vec_size',
                         lambda value: 0 < value <= 20,
                         message="--local_vec_size can't be more than 20")
flags.DEFINE_integer("local_vec_start_idx", default=None, help="If set, only in plants the slice [idx: idx+local_vec_size] are run.")
flags.register_validator('local_vec_start_idx',
                         lambda value: not value or 0 <= value < 20,
                         message="--local_vec_start_idx must be between 0 and 20")

flags.DEFINE_bool('vec_cache_train', default=True, help="Cache Train Data")
flags.DEFINE_bool('vec_cache_val', default=True, help="Cache Validation Data")

FLAGS = flags.FLAGS

"""" Consts """
PLANTS = ["17314", "1005", "56963", "862", "918", "56424", "384", "534", "643", "1395", "212", "1190",
          "458", "1467", "55827", "440", "235", "471", "1007", "1161"]

MODE_CODE = {"pass": 0, "local": 1, "global": 2, "cv": 3, "kn": 4, "global++": 5}

FOLDS = [[1190, 1395, 1005, 17314],
         [458, 471, 918, 212],
         [1467, 56424, 862, 1161],
         [384, 643, 1007, 56963],
         [534, 55827, 235, 440]]

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


def robust_mlflow_log_art(path):
    """
    There have been issues with the big checkpoints going over the network.
    This can fall back to just writing to disk

    Args:
        path: Path to log atrifcat
        """
    try:
        mlflow.log_artifacts(path)
    except mlflow.exceptions.MlflowException as e:
        logging.warning("Failed to save all artifacts")
        run = mlflow.active_run()
        mlflow_path = pathlib.Path(get_path("mlflow")) / run.info.experiment_id / run.info.run_id / "artifacts"
        logging.info(f"Attempting to write to {mlflow_path}")
        shutil.copytree(path, mlflow_path, dirs_exist_ok=True)


@dataclass
class VecExperimentRunner(FromDict):
    experiment: BaseRunner
    data_loader: DataLoader

    _tmp_dir: str = None

    def __post_init__(self):
        self._pass_data_loader = dataclasses.replace(self.data_loader, test_psb=Pass(), train_psb=Pass())

    @property
    def output_dir(self) -> str:
        # out_dir = FLAGS.output_dir
        # res_dir = get_path("results")
        # out_dir = os.path.join(res_dir, out_dir)
        if self._tmp_dir is None:
            slurm_id = os.environ.get("SLURM_JOB_ID", "0")
            step_id = os.environ.get("SLURM_STEP_ID", "0")
            path = pathlib.Path(".") / f"{slurm_id}.{step_id}"
            path.mkdir(parents=True, exist_ok=True)
            self._tmp_dir = str(path.resolve())
        out_dir = self._tmp_dir
        return out_dir

    def _cleanup_temp(self):
        """Remove the temp dir and reset it in case of future calls. Should only be called at the end of runs"""
        # Clean up temp file after run
        shutil.rmtree(self._tmp_dir)
        self._tmp_dir = None

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

    def stack_plant_ds(self) -> TrainTestL:
        # Extract PSB for use in the helper function
        test_psb = self.data_loader.test_psb
        train_psb = self.data_loader.train_psb

        # Set this so batch size is populate in ds el spec
        test_psb.drop_remainder = train_psb.drop_remainder = True
        train_psb.cache = test_psb.cache = False
        train_psb.batch_size = train_psb.batch_size * 6
        test_psb.batch_size = test_psb.batch_size * 6
        train_psb.shuffle_buff = None
        test_psb.shuffle_buff = None

        def shuffle_batch_vec():
            """Shuffle and batch vector data.
            Returns:

            """
            global_train, global_test = self._pass_data_loader.make_test_train_data_set(path=self.sp)

            if FLAGS.local_vec_size != 20:
                # Slice out plant dims here, otherwise vmap gets upset since dims dont match
                def slice_el(*x):
                    start = FLAGS.local_vec_start_idx
                    end = FLAGS.local_vec_start_idx + FLAGS.local_vec_size
                    return jax.tree_util.tree_map(lambda l: l[start:end], x)

                global_train = global_train.map(slice_el, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
                global_test = global_test.map(slice_el, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

            # Pad so all batches can be full
            zero = jax.tree_util.tree_map(lambda x: np.zeros(shape=x.shape, dtype=x.dtype.as_numpy_dtype),
                                          global_train.element_spec)
            pad = tf.data.Dataset.from_tensors((*zero,)).repeat(train_psb.batch_size)

            train = global_train.shuffle(train_psb.batch_size * FLAGS.vec_shuff_buf_scale).concatenate(pad).apply(train_psb)
            # train = train.apply(plant_filter(True))
            train = train.apply(PSB(cache=FLAGS.vec_cache_train, shuffle_buff=2 ** 4))

            val = global_test.concatenate(pad).apply(train_psb)
            # val = val.apply(plant_filter(False))
            val = val.apply(PSB(cache=False))

            return (train, val)

        return shuffle_batch_vec

    def run(self):
        logging.info("Setting up plants datasets")
        logging.info(f"Reading data from {PLANT_DATASET_FILE}")
        plant_dataset = self.stack_plant_ds()
        logging.info(f"Fitting per plants")
        if FLAGS.vec_split_lg:
            if FLAGS.vec_run_local:
                # Local runs
                FLAGS.inc_globcv = False
                FLAGS.inc_local = True

                # default is to run on all plants using a slice of 20 (so only one step).
                # Can run smaller slice slices (e.g 5) when VRAM issues are present on local but not GCV
                # Can also run on a subset of plants if local_vec_start_idx is set
                start_idxs = range(0, 20, FLAGS.local_vec_size) if FLAGS.local_vec_start_idx is None else [FLAGS.local_vec_start_idx]
                for idx in start_idxs:
                    FLAGS.local_vec_start_idx = idx
                    name = f"local_{idx}-{idx+FLAGS.local_vec_size}"
                    with mlflow.start_run(run_name=name, nested=True):
                        st = datetime.datetime.now()
                        self._run_fold(name, plant_dataset)
                        push_job_run(name, runtime=st - datetime.datetime.now())

                # TODO: Log the artifacts (ckpts etc to the parent run, might be a bad idea)
                robust_mlflow_log_art(self.output_dir)

            if FLAGS.vec_run_gcv:
                # Global runs
                FLAGS.local_vec_size = 20  # Set this so we always run with all AOIs
                FLAGS.inc_globcv = True
                FLAGS.inc_local = False
                with mlflow.start_run(run_name="GCV", nested=True):
                    st = datetime.datetime.now()
                    self._run_fold("gcv", plant_dataset)
                    push_job_run("gvc", runtime=datetime.datetime.now() - st)

                # TODO: Log the artifacts (ckpts etc to the parent run, might be a bad idea)
                robust_mlflow_log_art(self.output_dir)
        else:
            self._run_fold("all", plant_dataset)
            robust_mlflow_log_art(self.output_dir)

        # TODO: make this more robust
        self._cleanup_temp()