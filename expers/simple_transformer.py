import optax
import tensorflow as tf

from hemera.mlflow_utils import ml_flow_track

tf.config.set_visible_devices([], "GPU")

import jax
import jax.numpy as jnp
from absl import app, flags, logging
from typing import Sequence

from chemise.traning.vector_trainer import VectorTrainer
from zephyrus.core.jax_bits import loss, metrics
from zephyrus.model_runners.con_runner import ConRunner
from zephyrus.model_runners.jax_model_runner import JaxVecModelRunner
from zephyrus.utils.vec_experiment_runner import VecExperimentRunner
from zephyrus.utils.zeph_vec_unbatch import on_dev_shape

from flax import linen as nn
from jaxtyping import Array, Num

from zephyrus.data_pipelines.utils import unpack_jax
from zephyrus.model_runners.irrad_with_images import WithImages
from einops import rearrange, pack, repeat

from chemise.misc import nd_tile
from chemise.layers import MLC
from chemise.layers.transformers import TransformerEncoder, Decoder, AddPositionEmbs
from chemise.traning import BasicTrainer

flags.DEFINE_integer("mlp_depth", default=3, help="Depth of MLP")
flags.DEFINE_integer("mlp_width", default=128, help="With of MLP")
flags.DEFINE_integer("weather_offset", default=1, help="How much to offset weather")
flags.DEFINE_boolean("weather", default=False, help="Include weather as an input")
flags.DEFINE_boolean("irrad", default=True, help="Include irrad as an input")
flags.DEFINE_boolean("meta_as_tokens", default=False, help="Include weather as an input")
flags.DEFINE_boolean("multi_head", default=True, help="Run multihead")
flags.DEFINE_integer("grad_accum_steps", default=1, help="Accumulate multiple steps of grads")
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

    num_heads: int = 4  # Number of heads to use in the Multi-Head Attention blocks
    num_layers: int = 4  # Number of encoder blocks to use
    model_dim: int = 256  # Hidden dimensionality to use inside the Transformer
    dropout_prob: float = 0.00  # Dropout to apply inside the model
    input_dropout_prob: float = 0.0  # Dropout to apply on the input features

    dtype = jnp.float32

    enc_dec_dim: int = 8
    topology: bool = True
    metadata: bool = True
    seq_len: int = 16

    noise_level: float = 0.04
    cnn_pre_embed: bool = True

    @nn.compact
    def __call__(self, _x, train: bool = True):
        # Load input data
        time_loc_in = [_x[f][:, :self.output_width] for f in self.time_loc_feats]  # Reshape the inputs out of the dict
        time_loc_stack = jnp.stack(time_loc_in, axis=-1)  # [batch, time_steps, feats]

        # Load input data
        weather_in = [_x[f][:, :self.output_width] for f in self.weather_feats]
        weather_stack = jnp.stack(weather_in, axis=-1)  # [batch, time_steps, feats]

        if FLAGS.weather:
            meta_stack = jnp.concatenate([time_loc_stack, weather_stack], axis=-1)
        else:
            meta_stack = time_loc_stack

        irad = _x[self.irrad_key]  # [batch, warmup_steps]

        batch_size, warmup_steps = _x[self.irrad_key].shape[0], _x[self.irrad_key].shape[1]

        # Load images
        images = unpack_jax(_x["img_stack"])
        # Normalize, no issues of the resize adding in extra noise
        images = images.astype(jnp.float32)  # Cast to floats
        images = images / 255.0  # keep it between 0 and 1

        batch_shape = images.shape
        B, in_TS, H, W, C = batch_shape
        norm_axis = (-1, -2, -3, -4)

        # IMG ORDER = ["HRV", "VIS006", "VIS008", "IR_016",
        #              "IR_039", "WV_062", "WV_073", "IR_087",
        #              "IR_097", "IR_108", "IR_120", "IR_134"]
        # Group into bands as defined by https://www.eumetsat.int/media/45126 (page 20)
        c_groups = [[0, 1, 2, 3], [5, 6], [8, 9, 10, 11]]
        res = []
        for c in c_groups:
            feats = len(c) * 4
            mlc = MLC(depth=2, features=feats, kernel_size=(3, 3, 3), pool_size=(1, 1, 1))
            chan = images[..., c]
            chan = mlc(chan)
            chan = nn.LayerNorm(reduction_axes=norm_axis, feature_axes=norm_axis)(chan)
            res.append(chan)  # [B, T, H, W, C]
        embedded_img = jnp.concatenate(res, axis=-1)

        enc_shrink = 4
        # WH, WW = int((H - enc_shrink) // math.sqrt(self.seq_len)), int((W - enc_shrink) // math.sqrt(self.seq_len))
        WH = WW = 4
        img_embed = nn.Conv(features=self.model_dim, kernel_size=(1, WH, WW), strides=(1, WH, WW),
                            padding="VALID", use_bias=True, dtype=self.dtype)

        embedded_img = img_embed(embedded_img)

        warmup_img = rearrange(embedded_img[:, :warmup_steps], "b ts h w c -> b (ts h w) c")  # Batch, Seq, Dim
        decode_img = rearrange(embedded_img[:, warmup_steps:], "b ts h w c -> b (ts h w) c")  # Batch, Seq, Dim

        """
         ENCODER Block
         """
        # Load the metadata and project into a token - add a 1 dim at axis 1 so we can concat into the seq
        meta, _ = pack([meta_stack[:, :warmup_steps], irad], "b ts *")
        meta = nn.Dense(self.model_dim)(meta)  # Expand vector dim out to match model dim of transformer
        meta = nn.LayerNorm()(meta)

        # Combine into a sequence and Add learned position info
        x = jnp.concatenate([meta, warmup_img], axis=1)  # [BS, 1 + seq_len + sq_len * TS, model_dim]
        x = AddPositionEmbs(posemb_init=nn.initializers.normal(stddev=0.02), name='posembed_input',
                            dims=self.model_dim, dtype=self.dtype)(x)
        x = nn.Dropout(self.input_dropout_prob)(x, deterministic=not train)
        # Appy Encoder Transformer
        encoded = TransformerEncoder(
            input_dim=self.model_dim,
            dim_feedforward=self.model_dim * 2,
            dropout_prob=self.dropout_prob,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dtype=self.dtype
        )(x, mask=None, train=train)

        """
        DECODER Block
        """
        out_TS = in_TS // FLAGS.img_update_feq

        if FLAGS.meta_as_tokens:
            # Use metadata as input tokens
            meta = meta_stack[:, warmup_steps:]
            meta = nn.Dense(self.model_dim)(meta)  # Expand vector dim out to match model dim of transformer
            meta = nn.LayerNorm()(meta)
            tokens = meta
        else:
            # Use a learend value as tokens
            tokens = self.param("out", nn.initializers.normal(stddev=0.02), (out_TS, self.model_dim), self.dtype)
            tokens = repeat(tokens, "time_steps dim -> batch time_steps dim", batch=B)

        decode_input = jnp.concatenate([tokens, decode_img], axis=1)  # [B, OUT_TS + IN_IMG, model_dim]
        decode_input = AddPositionEmbs(posemb_init=nn.initializers.normal(stddev=0.02), name='dec_pos',
                                       dims=self.model_dim, dtype=self.dtype)(decode_input)
        decode_input = nn.Dropout(self.input_dropout_prob)(decode_input, deterministic=not train)

        x = Decoder(
            input_dim=self.model_dim,
            dim_feedforward=self.model_dim * 2,
            dropout_prob=self.dropout_prob,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dtype=self.dtype
        )(decode_input, encoded, train=train)

        x = x[:, :out_TS]
        x = nn.LayerNorm()(x)
        pred = nn.Dense(1, dtype=self.dtype)(x)
        pred = jnp.squeeze(pred, axis=2)  # remove final dim
        return pred


class ConvRunner(JaxVecModelRunner, ConRunner):
    trainer: BasicTrainer = VectorTrainer(state=None, loss_fn=loss, metrics_fn=metrics, rng_keys=["lstm_cell"], on_dev_shape=on_dev_shape)

    def make_jax_model(self) -> nn.Module:
        out_steps = FLAGS.warmup_steps + FLAGS.output_steps
        return Model(output_width=out_steps, mlp_width=FLAGS.mlp_width, mlp_depth=FLAGS.mlp_depth)

    def make_optim(self) -> optax.GradientTransformation:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-7,
            peak_value=FLAGS.learning_rate * 3,
            warmup_steps=200,
            decay_steps=10_000,
            end_value=FLAGS.learning_rate / 10
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
            optax.adamw(lr_schedule)
        )
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=FLAGS.grad_accum_steps)
        return optimizer


EXPR_NAME = "Irrad-Transformer"
VERSION = "0.0.1"


@ml_flow_track(exper_name=EXPR_NAME, version=VERSION)
def main(argv):
    logging.info(f"JAX visible devices: {jax.devices()}")
    exper = ConvRunner.from_dict(FLAGS.flag_values_dict())
    data = WithImages.from_dict(FLAGS.flag_values_dict())
    runner = VecExperimentRunner(experiment=exper, data_loader=data)
    runner.run()


if __name__ == "__main__":
    # Parse Args
    app.run(main)
