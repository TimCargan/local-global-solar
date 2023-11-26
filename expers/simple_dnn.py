import jax
import jax.numpy as jnp
import tensorflow as tf
from absl import app, flags, logging
from jaxtyping import Array
from typing import Dict

from chemise.traning.vector_trainer import VectorTrainer
from hemera.mlflow_utils import ml_flow_track
from zephyrus.core.jax_bits import loss, metrics
from zephyrus.model_runners.con_runner import ConRunner
from zephyrus.model_runners.irrad_point_weather import IrradPointWeather
from zephyrus.model_runners.jax_model_runner import JaxVecModelRunner
from zephyrus.utils.vec_experiment_runner import VecExperimentRunner
from zephyrus.utils.zeph_vec_unbatch import on_dev_shape

tf.config.set_visible_devices([], "GPU")
from flax import linen as nn

from chemise.layers import MLP
from chemise.traning import BasicTrainer

flags.DEFINE_integer("mlp_depth", default=3, help="Depth of MLP")
flags.DEFINE_integer("mlp_width", default=128, help="With of MLP")
flags.DEFINE_boolean("weather", default=True, help="Include weather as an input")
flags.DEFINE_boolean("irrad", default=True, help="Include irrad as an input")
flags.DEFINE_boolean("embed", default=False, help="Include embeded lat lon")

FLAGS = flags.FLAGS


class Model(nn.Module):
    weather_feats = ['clouds', 'pres', 'precip', 'temp', 'rh', 'ghi', 'wind_x', 'wind_y']
    time_loc_feats = ["hour_sin", "hour_cos", "year_sin", "year_cos", "elev_angle", "azimuth_sin", "azimuth_cos"]
    irrad_key = 'irradiance_in'

    output_width: int = 30
    warmup_steps: int = 24
    hidden_size: int = 128
    mlp_width: int = 128
    mlp_depth: int = 3

    @nn.compact
    def __call__(self, _x: Dict[str, Array], train=True):
        # Load input data
        weather_in = [_x[f][:, :self.output_width] for f in self.weather_feats]
        weather_stack = jnp.stack(weather_in, axis=-1)  # [batch, time_steps, feats]

        time_loc_in = [_x[f][:, :self.output_width] for f in self.time_loc_feats]
        time_loc_stack = jnp.stack(time_loc_in, axis=-1)  # [batch, time_steps, feats]

        batch_size, warmup_steps = _x[self.irrad_key].shape[0], _x[self.irrad_key].shape[1]
        irad = _x[self.irrad_key]  # [batch, warmup_steps]

        # Build x
        x = time_loc_stack  # [B, ts, F]
        if FLAGS.weather:
            x = jnp.concatenate([x, weather_stack], axis=-1)
        x = jnp.reshape(x, (batch_size, -1))  # [B, ts * f_width]

        if FLAGS.irrad:
            x = jnp.concatenate([x, irad], axis=-1)  # [B, ts * f_width + e_width + irrad]

        if FLAGS.embed:
            embed = self.embed_lat_lon(_x)  # [B, e_width]
            x = jnp.concatenate([x, embed], axis=-1)  # [B, ts * f_width + e_width]

        # Main part of the DNN
        x = MLP(depth=self.mlp_depth, width=self.mlp_width)(x)
        pred = nn.Dense(self.output_width)(x)

        # pred = jnp.squeeze(pred, axis=2)  # remove final dim
        return pred

    def embed_lat_lon(self, _x):
        ### Lat Lon Embedding
        em_keys = [('Latitude', 10, -50), ('Longitude', 10, 7)]
        e_feats = []
        for n, size, min in em_keys:
            key = _x[f"{n}"].astype(int) + min
            key = jnp.clip(key, a_min=0, a_max=10)  # clip out any bad values, otherwise we get lots of NaNs
            e_feat = nn.Embed(size, 4, name=f"{n}_embed")(key)
            e_feat = jnp.reshape(e_feat, (-1, 4))
            e_feats.append(e_feat)
        e_feats = jnp.concatenate(e_feats, axis=-1)  # [[batch, 4], [batch, 4]] -> [batch, 8]
        return e_feats


class DNN_Runner(JaxVecModelRunner, ConRunner):
    trainer: BasicTrainer = VectorTrainer(state=None, loss_fn=loss, metrics_fn=metrics, rng_keys=["lstm_cell"], on_dev_shape=on_dev_shape)

    def make_jax_model(self) -> nn.Module:
        out_steps = FLAGS.warmup_steps + FLAGS.output_steps
        return Model(output_width=out_steps, mlp_width=FLAGS.mlp_width, mlp_depth=FLAGS.mlp_depth)


EXPR_NAME = "DNN"
VERSION = "0.0.1"
@ml_flow_track(exper_name=EXPR_NAME, version=VERSION)
def main(argv):
    logging.info(f"JAX visible devices: {jax.devices()}")
    exper = DNN_Runner.from_dict(FLAGS.flag_values_dict())
    data = IrradPointWeather.from_dict(FLAGS.flag_values_dict())
    VecExperimentRunner(experiment=exper, data_loader=data).run()


if __name__ == "__main__":
    # Parse Args
    app.run(main)
