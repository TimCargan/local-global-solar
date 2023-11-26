import jax
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from absl import app, flags, logging
import jax.numpy as jnp
from flax import linen as nn
from chemise.layers import MLP
from chemise.traning import BasicTrainer
from chemise.layers.lstm import FullAutoLSTM, SimpleLSTM
from chemise.traning.vector_trainer import VectorTrainer
from hemera.mlflow_utils import ml_flow_track
from zephyrus.core.jax_bits import loss, metrics
from zephyrus.model_runners.con_runner import ConRunner
from zephyrus.model_runners.irrad_point_weather import IrradPointWeather
from zephyrus.model_runners.jax_model_runner import JaxVecModelRunner
from zephyrus.utils.vec_experiment_runner import VecExperimentRunner
from zephyrus.utils.zeph_vec_unbatch import on_dev_shape


flags.DEFINE_bool("weather_mlp", default=True, help="Have an MLP for the weather")
flags.DEFINE_integer("mlp_depth", default=3, help="Depth of MLP")
flags.DEFINE_integer("mlp_width", default=128, help="With of MLP")
flags.DEFINE_boolean("irrad", default=True, help="Include irrad as an input")
flags.DEFINE_boolean("weather", default=True, help="Include weather as an input")

FLAGS = flags.FLAGS


class Model(nn.Module):
    weather_feats = ['clouds', 'pres', 'precip', 'temp', 'rh', 'ghi', 'wind_x', 'wind_y']
    time_loc_feats = ["hour_sin", "hour_cos", "year_sin", "year_cos", "elev_angle", "azimuth_sin", "azimuth_cos"]
    irrad_key = 'irradiance_in'
    fs = time_loc_feats
    fs = fs + weather_feats
    output_width: int = 30
    warmup_steps: int = 24
    hidden_size: int = 128
    mlp_width: int = 128
    mlp_depth: int = 3

    @nn.compact
    def __call__(self, _x, train: bool = True):
        # Load input data
        weather_in = [_x[f][:, :self.output_width] for f in self.weather_feats]
        weather_stack = jnp.stack(weather_in, axis=-1)  # [batch, time_steps, feats]

        time_loc_in = [_x[f][:, :self.output_width] for f in self.time_loc_feats]
        time_loc_stack = jnp.stack(time_loc_in, axis=-1)  # [batch, time_steps, feats]

        batch_size, warmup_steps = _x[self.irrad_key].shape[0], _x[self.irrad_key].shape[1]

        irad = _x[self.irrad_key]  # [batch, warmup_steps]

        """ For the images, we use the MLC, could have an MLP here"""
        x = time_loc_stack

        if FLAGS.weather:
            x = weather_stack  # [B, ts, F]
            x = MLP(depth=3, width=32)(x)
            axis = (-1, -2)
            x = nn.LayerNorm(reduction_axes=axis, feature_axes=axis)(x)
            x = jnp.concatenate([x, time_loc_stack], axis=-1)  # [B, ts, width]

        # Set up lstm
        cell = nn.OptimizedLSTMCell(features=self.hidden_size)
        rng = self.make_rng('lstm_cell')
        initial_state = cell.initialize_carry(rng, x[:, 0].shape)

        # Define output layer
        output_layer = nn.Sequential([MLP(depth=self.mlp_depth - 1, width=self.mlp_width), nn.Dense(1)])

        if FLAGS.irrad:
            warmup_input = jnp.concatenate([x[:, :warmup_steps], jnp.expand_dims(irad, axis=-1)], axis=-1)
            auto_input = x[:, warmup_steps:]
            pred = FullAutoLSTM(cell=cell, output_layer=output_layer)(initial_state, warmup_input, auto_input)
        else:
            carry, warm_lstm = SimpleLSTM(cell=cell)(initial_state, x)
            pred = output_layer(warm_lstm)

        pred = jnp.squeeze(pred, axis=2)  # remove final dim
        return pred


class LSTM_Runner(JaxVecModelRunner, ConRunner):
    trainer: BasicTrainer = VectorTrainer(state=None, loss_fn=loss, metrics_fn=metrics, rng_keys=["lstm_cell"], on_dev_shape=on_dev_shape)

    def make_jax_model(self) -> nn.Module:
        out_steps = FLAGS.warmup_steps + FLAGS.output_steps
        return Model(output_width=out_steps, mlp_width=FLAGS.mlp_width, mlp_depth=FLAGS.mlp_depth)


EXPR_NAME = "LSTM"
VERSION = "0.0.1"


@ml_flow_track(exper_name=EXPR_NAME, version=VERSION)
def main(argv):
    logging.info(f"JAX visible devices: {jax.devices()}")
    exper = LSTM_Runner.from_dict(FLAGS.flag_values_dict())
    data = IrradPointWeather.from_dict(FLAGS.flag_values_dict())
    VecExperimentRunner(experiment=exper, data_loader=data).run()


if __name__ == "__main__":
    # Parse Args
    app.run(main)
