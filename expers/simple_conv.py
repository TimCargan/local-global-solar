import jax
import jax.numpy as jnp
import tensorflow as tf
from absl import app, flags, logging
from typing import Sequence

from chemise.layers.lstm import FullAutoLSTM, SimpleLSTM
from chemise.traning.vector_trainer import VectorTrainer
from hemera.mlflow_utils import ml_flow_track
from zephyrus.core.jax_bits import loss, metrics
from zephyrus.model_runners.con_runner import ConRunner
from zephyrus.model_runners.jax_model_runner import JaxVecModelRunner
from zephyrus.utils.vec_experiment_runner import VecExperimentRunner
from zephyrus.utils.zeph_vec_unbatch import on_dev_shape

tf.config.set_visible_devices([], "GPU")
from flax import linen as nn
from jaxtyping import Array, Num

from zephyrus.data_pipelines.utils import unpack_jax
# from zephyrus.utils import experiment_runner
from zephyrus.model_runners.irrad_with_images import WithImages

from chemise.misc import nd_tile
from chemise.layers import MLP, MLC
from chemise.traning import BasicTrainer

flags.DEFINE_integer("mlp_depth", default=3, help="Depth of MLP")
flags.DEFINE_integer("mlp_width", default=128, help="With of MLP")
flags.DEFINE_integer("weather_offset", default=1, help="How much to offset weather")
flags.DEFINE_boolean("irrad", default=True, help="Include irrad as an input")
flags.DEFINE_boolean("multi_head", default=True, help="Run multihead")

FLAGS = flags.FLAGS


class Square(nn.Module):
    size: Sequence[int]

    @nn.compact
    def __call__(self, x: Num[Array, "... N"], /) -> Num[Array, "... H W N"]:
        return nd_tile(x, self.size)


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
        time_loc_in = [_x[f][:, :self.output_width] for f in self.time_loc_feats]  # Reshape the inputs out of the dict
        time_loc_stack = jnp.stack(time_loc_in, axis=-1)  # [batch, time_steps, feats]

        irad = _x[self.irrad_key]  # [batch, warmup_steps]

        batch_size, warmup_steps = _x[self.irrad_key].shape[0], _x[self.irrad_key].shape[1]

        # e_feats = self.embed_lat_lon(_x)
        images = unpack_jax(_x["img_stack"])
        # Normalize, no issues of the resize adding in extra noise
        images = images.astype(jnp.float32)  # Cast to floats
        images = images / 255.0  # keep it between 0 and 1

        x = images

        norm_axis = (-1, -2, -3, -4)
        if FLAGS.multi_head:
            # IMG ORDER = ["HRV", "VIS006", "VIS008", "IR_016",
            #              "IR_039", "WV_062", "WV_073", "IR_087",
            #              "IR_097", "IR_108", "IR_120", "IR_134"]
            # Group into bands as defined by https://www.eumetsat.int/media/45126 (page 20)
            c_groups = [[0, 1, 2, 3], [5, 6], [8, 9, 10, 11]]
            # c_groups = [[i] for i in range(12)]
            res = []
            for c in c_groups:
                feats = len(c) * 4
                mlc = MLC(depth=2, features=feats, kernel_size=(3, 3, 3), pool_size=(1, 1, 1))
                chan = x[..., c]
                chan = mlc(chan)
                chan = nn.LayerNorm(reduction_axes=norm_axis, feature_axes=norm_axis)(chan)
                res.append(chan)  # [B, T, H, W, C]
            x = jnp.concatenate(res, axis=-1)
            x = MLC(depth=3, features=16, kernel_size=(3, 3, 3), pool_size=(1, 2, 2))(x)
        else:
            mlc = MLC(depth=3, features=16, kernel_size=(3, 3, 3), pool_size=(1, 2, 2))
            x = mlc(x)  # [B, T, H, W, C]

        x = MLC(depth=1, features=16, kernel_size=(3, 3, 3), pool_size=(FLAGS.img_update_feq, 1, 1))(x)  # [B, T, H, W, C]
        x = nn.LayerNorm(reduction_axes=norm_axis, feature_axes=norm_axis)(x)

        x = x.reshape((*x.shape[:-3], -1))  # [B, ts, F]
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


class ConvRunner(JaxVecModelRunner, ConRunner):
    trainer: BasicTrainer = VectorTrainer(state=None, loss_fn=loss, metrics_fn=metrics, rng_keys=["lstm_cell"], on_dev_shape=on_dev_shape)

    def make_jax_model(self) -> nn.Module:
        out_steps = FLAGS.warmup_steps + FLAGS.output_steps
        return Model(output_width=out_steps, mlp_width=FLAGS.mlp_width, mlp_depth=FLAGS.mlp_depth)


EXPR_NAME = "CNN"
VERSION = "0.0.1"


@ml_flow_track(exper_name=EXPR_NAME, version=VERSION)
def main(argv):
    logging.info(f"JAX visible devices: {jax.devices()}")
    exper = ConvRunner.from_dict(FLAGS.flag_values_dict())
    data = WithImages.from_dict(FLAGS.flag_values_dict())
    VecExperimentRunner(experiment=exper, data_loader=data).run()


if __name__ == "__main__":
    # Parse Args
    app.run(main)
