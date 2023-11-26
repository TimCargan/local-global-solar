import datetime
from abc import ABC

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import flags, logging
from einops import *

import chemise.callbacks.loggers
from chemise.callbacks import Checkpointer, Line, Profile, ProgressBar
from chemise.traning import BasicTrainer
from chemise.utils import list_dict_to_dict_list
from zephyrus.data_pipelines.transformers.normalize import Normalizer
from zephyrus.utils.runner import BaseRunner

FLAGS = flags.FLAGS


class ConRunner(BaseRunner, ABC):
    def fit_model(self, m: BasicTrainer, d, val_data, output_dir):
        self.logger.debug("Add chemise callbacks")
        ckpter = Checkpointer(output_dir, overwrite=True,
                              intra_train_freq_time=datetime.timedelta(minutes=10),
                              epoch_keep_period=4,
                              auto_restore=FLAGS.auto_restore)
        graph = Line(title="Loss")
        prog_bar = ProgressBar(update_metric_freq=3)
        mlflow_track = chemise.callbacks.loggers.Mlflow(update_metric_freq=FLAGS.mlflow_metric_freq)
        m.callbacks = [ckpter, graph, mlflow_track]
        if FLAGS.profile_step:
            prof = Profile(profile_dir=output_dir,
                           steps=(FLAGS.profile_step, FLAGS.profile_step + FLAGS.profile_step_count))
            m.callbacks.append(prof)

        if FLAGS.debug:
            # first = d.take(1).as_numpy_iterator().next()
            # x, y = first[:2]
            # res = m.step(first)
            # m.train_step(first)
            pass

        self.logger.info("Fitting Model")
        if not FLAGS.run_val:
            val_data = None
        m.fit(d, val_data=val_data, num_epochs=FLAGS.num_epochs)
        return m

    def calc_steps(self):
        # TODO, this is tied into the data used so be careful here. Might be good to unify this
        # Calculate output Step IDs
        out_step_index = np.arange(0, FLAGS.warmup_steps + FLAGS.output_steps)
        out_step_index = out_step_index - (FLAGS.warmup_steps - 1)
        return out_step_index

    keep_cols = ["plant", "out_ts", "mode", "ghi"]

    def eval_model(self, m: BasicTrainer, test: tf.data.Dataset, ckpt_dir) -> pd.DataFrame:
        if ckpt_dir:
            logging.info("Restoring Model")
            try:
                m = Checkpointer.restore(m, ckpt_dir)
            except ValueError as e:
                """
                Try loading with no restore kwargs, some older checkpoints have this issue.
                """
                logging.warning("Failed to restore model, trying fallback with no kwargs")
                m = Checkpointer.restore(m, ckpt_dir, use_restore_kwargs=False)

        res = {}
        out_step_index = self.calc_steps()

        @jax.pmap
        def res_reshape(b):
            """
            We assume all but the last dim are batch dims and so can be folded together without issue
            :param b:
            :return:
            """
            x, y = b[:2]  # [(b | p, b), e]
            y_hat = b[-1]  # [(b | p, b), e]

            y_hat = rearrange(y_hat, "b ... e -> (b ...) e")  # [(b | p, b), e]
            steps = repeat(out_step_index, "s -> b s", **parse_shape(y_hat, 'b _'))

            shape = jnp.shape(y_hat)
            e = shape[-1]

            # Batch results dict and reshape batch dims into one
            b_dict = dict(**{"y": y["pred"]}, **{k: x[k][..., :e] for k in self.keep_cols})
            b_dict = {k: rearrange(v, "b ... e -> (b ...) e") for k, v in b_dict.items()}

            # Add y_hat and step count
            b_dict["step"] = steps
            b_dict["y_hat"] = y_hat
            return b_dict

        logging.info("Running eval loop")
        res = [res_reshape(r) for r in m.map_model(test)]
        res = list_dict_to_dict_list(res)

        logging.info("Reshape and save eval")
        # Bring flatten gpu and batch dims,  and data into host memory
        res = {k: [asnumpy(x) for x in v] for k, v in res.items()}
        res = {k: [np.reshape(x, (-1)) for x in v] for k, v in res.items()}
        res = {k: np.concatenate(v) for k, v in res.items()}

        eval_mse = np.mean(((res["y_hat"] - res["y"]) ** 2))
        logging.info(f"Eval MSE: {eval_mse}")

        # Convert to Dataframe
        res = pd.DataFrame(res)
        res = res[res["mode"] != 0]  # Remove any padding
        return res

    def eval_summary(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # Add Time cols
        df['date'] = pd.to_datetime(df['out_ts'], unit='s')
        df["month"] = df["date"].dt.month

        df['date_m1'] = df["date"] - pd.Timedelta(hours=1)

        # Un Norm irrad
        df["y"] = Normalizer._unnorm_irrad(df["y"])
        df["y_hat"] = Normalizer._unnorm_irrad(df["y_hat"])
        df["ghi"] = Normalizer._un_log(df["ghi"])

        # Calculate Metrics
        df_serror = self.s_error(df)
        df_me = self.mean_errors(df)

        logging.info(df_serror.to_string())
        return {"mean_errors": df_me, "skill": df_serror}

    def mean_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        df["se"] = (df["y"] - df["y_hat"]) ** 2  # Square Error
        df["ae"] = np.abs(df["y"] - df["y_hat"])  # Abs Error
        # Filter out night values
        df = df[(df["y"] >= 20) & (df["ghi"] > 1)]
        df = df.groupby(["plant", "mode", "step"], as_index=False).agg(mse=("se", np.mean), mae=("ae", np.mean), ymean=("y", np.mean))
        df["rmse"] = np.sqrt(df["mse"])
        df["nrmse"] = df["rmse"] / df["ymean"]

        # Reshape to make nice tables
        metrics_of_intrest = ["rmse", "mae", "nrmse"]
        df = df[df["step"] >= -2].pivot(index=["mode", "plant"], columns=["step"], values=metrics_of_intrest)
        return df

    def s_error(self, df: pd.DataFrame) -> pd.DataFrame:
        # K_t = Irrad_t / Clearsky_t
        # V = sqrt(mean(∆K(t)**2)) where ∆K(t) = K_t - K_t-1
        # U = sqrt(mean((error / clearsky)**2)) ->
        #                us = (error / clearsky)**2

        # Sanity check for gaps in the time ranges
        df['date_m1'] = df["date"] - pd.Timedelta(hours=1)
        df["date_s1"] = df.sort_values(by=['date'], ascending=True).groupby(["plant", "mode", "step"])["date"].shift(1)

        df["K"] = np.where(df["ghi"] != 0, df["y"] / df["ghi"], np.NAN)
        # df["K"] = np.nan_to_num(posinf=np.NAN, neginf=np.NAN)
        df["K_s1"] = df.sort_values(by=['date'], ascending=True).groupby(["plant", "mode", "step"])["K"].shift(1)
        df["∆K^2"] = np.where(df['date_m1'] == df["date_s1"], (df["K"] - df["K_s1"]) ** 2, np.NAN)

        df["us"] = ((df["y_hat"] - df["y"]) / df["ghi"]) ** 2
        df["us"] = np.where(df["ghi"] != 0, df["us"], np.NAN)

        df = df[(df["y"] >= 20) & (df["ghi"] > 1)]

        def root_mean(c):
            return np.sqrt(np.nanmean(c))

        df = df.groupby(["plant", "mode", "step", "month"], as_index=False).agg(U=("us", root_mean), V=("∆K^2", root_mean))

        df["s"] = 1 - (df["U"] / df["V"])
        df = df.groupby(["plant", "mode", "step"], as_index=False).agg(S=("s", np.mean))

        # Pivot
        df = df[df["step"] >= -2].pivot(index=["mode", "plant"], columns=["step"], values=["S"])
        return df
