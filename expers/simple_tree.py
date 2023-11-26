
import tensorflow as tf
from absl import app, flags

from zephyrus.data_pipelines.transformers import Pass
from zephyrus.model_runners.con_runner import ConRunner
from zephyrus.model_runners.irrad_point_weather import IrradPointWeather
from zephyrus.trees.tfdt_chemise_wrap import TFDTChemise
from zephyrus.utils.experiment_runner import ExperimentRunner

tf.config.set_visible_devices([], "GPU")

flags.DEFINE_integer("threads", default=3, help="Number of threads to use for the trees")
flags.DEFINE_boolean("weather", default=True, help="Include weather as an input")
flags.DEFINE_boolean("irrad", default=True, help="Include irrad as an input")

FLAGS = flags.FLAGS


class Tree_Runner(ConRunner):

    def make_model(self, data_spec):
        weather_feats = ['clouds', 'pres', 'precip', 'temp', 'rh', 'ghi', 'wind_x', 'wind_y']
        time_loc_feats = ["hour_sin", "hour_cos", "year_sin", "year_cos", "elev_angle", "azimuth_sin", "azimuth_cos"]
        irrad_key = ['irradiance_in']

        inputs = time_loc_feats
        inputs = inputs + weather_feats if FLAGS.weather else inputs
        inputs = inputs + irrad_key if FLAGS.irrad else inputs

        model = TFDTChemise(output_steps=FLAGS.output_steps, threads=FLAGS.threads, input_feats=inputs)
        model.make_model()
        return model


def main(argv):
    exper = Tree_Runner.from_dict(FLAGS.flag_values_dict())
    env = FLAGS.flag_values_dict()
    env["vectorize"] = False
    data = IrradPointWeather.from_dict(env)
    data.train_psb = Pass()

    # data.test_psb.batch_size = FLAGS.tree_batch_size
    # data.test_psb.drop_remainder = False

    ExperimentRunner(experiment=exper, data_loader=data).run()


if __name__ == "__main__":
    # Parse Args
    app.run(main)
