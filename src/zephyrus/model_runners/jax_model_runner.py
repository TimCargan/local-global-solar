import flax.linen as nn
import jax
import optax
import tensorflow as tf
from abc import ABC, abstractmethod
from absl import flags, logging
from functools import partial

from chemise.traning import BasicTrainer, MpTrainState
from chemise.traning.vector_trainer import VectorTrainer
from chemise.utils import datasetspec_to_zero

flags.DEFINE_float("learning_rate", default=3e-3, help="Learning rate for the optimizer")

FLAGS = flags.FLAGS


class JaxModelRunner(ABC):
    @property
    @abstractmethod
    def trainer(self) -> BasicTrainer:
        pass

    @abstractmethod
    def make_jax_model(self) -> nn.Module:
        pass

    def make_optim(self) -> optax.GradientTransformation:
        return optax.adam(FLAGS.learning_rate)

    state_init: bool = False

    def init_state(self):
        self._jax_model = self.make_jax_model()
        self._jax_model_init = jax.jit(self._jax_model.init)
        self._jax_opt = self.make_optim()
        self._jax_state = MpTrainState(step=0, params=None, opt_state=None, apply_fn=self._jax_model.apply,
                                     tx=self._jax_opt)
        self.state_init = True

    # @partial(jax.jit, static_argnums=(0, 1))
    def _make_state(self, jax_model, zeros):
        """ Reset vs recreate state so we avoid retracing everything"""
        rng = jax.random.PRNGKey(0)
        _rng, lstm_rng, _init_rng = jax.random.split(rng, num=3)
        rngs = {"params": _rng, 'lstm_cell': lstm_rng}
        params = jax_model.init(rngs, zeros[0])['params']
        opt_state = self.make_optim()
        jax_state = MpTrainState.create(apply_fn=jax_model.apply, params=params, tx=opt_state)
        return jax_state

    def make_state(self, zeros):
        """ Reset vs recreate state so we avoid retracing everything"""
        logging.info("Reset state objects")
        if not self.state_init:
            logging.info("First Run, creating state objects")
            self.init_state()

        state = self._make_state(self._jax_model, zeros)
        self._jax_state = self._jax_state.replace(step=state.step, params=state.params, opt_state=state.opt_state)
        return self._jax_state

    def make_model(self, data_spec: tf.data.DatasetSpec):
        zeros = datasetspec_to_zero(data_spec, batch_size=FLAGS.batch_size)
        new_state = self.make_state(zeros)
        self.trainer.reset()
        self.trainer.state = new_state
        return self.trainer


class JaxVecModelRunner(JaxModelRunner, ABC):
    """
    Wrap Jax model build with a vmap
    """
    @property
    @abstractmethod
    def trainer(self) -> VectorTrainer:
        pass

    def make_model(self, data_spec: tf.data.DatasetSpec):
        vec_size = int(FLAGS.inc_local) * 20 + FLAGS.inc_globcv * 6
        data_spec = jax.tree_util.tree_map(lambda x: tf.TensorSpec((x.shape[0], vec_size, *x.shape[2:]), x.dtype), data_spec)
        return super(JaxVecModelRunner, self).make_model(data_spec)

    @partial(jax.vmap, in_axes=(None, None, 1), out_axes=(0))
    def _make_state(self, m, z):
        return super(JaxVecModelRunner, self)._make_state(m, z)
