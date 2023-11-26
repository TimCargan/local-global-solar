from dataclasses import field

import tensorflow as tf
from absl import flags, logging

from zephyrus.data_pipelines.parsers.EUMETSAT import EuMetSat
from zephyrus.data_pipelines.transformers import EU_Images, PSB
from zephyrus.model_runners.irrad_point_weather import IrradPointWeather
from zephyrus.utils.runner import DataLoader

flags.DEFINE_integer("img_crop_size", default=64, help="Size of image after center crop")
flags.DEFINE_integer("img_update_feq", default=1, help="Number of images per hour")
flags.DEFINE_integer("img_min_offset", default=0, help="Min of hour offset, one of 0, 15, 30, 45")

flags.DEFINE_bool("img_ys", default=False, help="Include image stack in the Y dict")


flags.register_validator('img_update_feq',
                         lambda value: value in [1, 2, 4],
                         message='--img_update_feq must one of 1, 2, 4')

flags.register_validator('img_min_offset',
                         lambda value: value in [0, 15, 30, 45],
                         message='--img_update_feq must one of 1, 2, 4')

FLAGS = flags.FLAGS

_img_reader = None

def get_img_reader():
    global _img_reader
    if _img_reader is None:
        _img_reader = EuMetSat(slice_size=FLAGS.img_crop_size)
    return _img_reader


class WithImages(DataLoader):
    train_psb: PSB = field(default_factory=lambda : PSB(batch_size=FLAGS.batch_size, cache=False, prefetch_buff=2,
                         shuffle_buff=2 ** 14, drop_remainder=False, pad=FLAGS.pad_psb))

    test_psb: PSB = field(default_factory=lambda : PSB(batch_size=FLAGS.batch_size, cache=False, prefetch_buff=2,
                          drop_remainder=False, pad=FLAGS.pad_psb))

    def __init__(self, num_epochs: int = 20, batch_size: int = 256, learning_rate: float = 3e-4,
                 warmup_steps: int = 12, output_steps: int = 6, img_crop_size: int = 64,
                 no_group_data: bool = False, limit_eval: bool = False, cache_input: bool = True, **kwargs):

        super(WithImages, self).__init__(**kwargs)

        img_crop_size = FLAGS.img_crop_size
        self.irrad_loader = IrradPointWeather(vectorize=False)
        self.hp = self.irrad_loader.hp
        self.hp.Config("NO_GROUP_DATA", False)
        self.hp.Config("CACHE_INPUT", True)
        self.hp.Config("IMG_SIZE", img_crop_size)

        ## Image Reader config
        self._im_reader = EuMetSat(slice_size=img_crop_size)
        logging.info(
            f"Missing 36 of {len(self._im_reader.date_dict)} ({self._im_reader.min_date} - {self._im_reader.max_date})")
        self._add_img = EU_Images(self._im_reader, crop_size=img_crop_size)


    def feature_extract(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        This will take in the raw irrad data
        call  irrad shape (self.irrad_loader)
        then use the image loader to get the images (self._add_img, it in the fuction)

        Vectrized the data (group all plants in the same ts togher) but don't pad it
        do a v_read to vectorize the image load

        pad the data

        Args:
            ds:

        Returns:

        """
        it = self._add_img
        logging.info(f"Cardinality before opps: {ds.cardinality()}")
        ds = self.irrad_loader.feature_extract(ds)

        ds = ds.apply(it.filter)

        if not self.hp.get("NO_GROUP_DATA"):
            ds = ds.apply(it.group)
            ds = ds.map(it.sort_group)

        ds = ds.cache()
        # ds = ds.apply(it.extra_shape)
        ds = ds.apply(it.v_read)

        def _img_xys(xs, ys):
            nxs = {f: v for f, v in xs.items()}
            nys = {f: v for f, v in ys.items()}
            nxs["img_stack"] = xs["img_stack"]
            if FLAGS.img_ys:
                nys["img_stack"] = tf.image.central_crop(xs["img_stack"], 0.125)
            return nxs, nys
        def _to_ds(*xys):
            return tf.data.Dataset.from_tensor_slices((*xys,))


        if self.vectorize:
            ds = ds.map(_img_xys, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            ds = ds.map(it.padd_sorted_group, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        else:
            ds = ds.interleave(_to_ds, num_parallel_calls=tf.data.AUTOTUNE, block_length=8, deterministic=False)
            ds = ds.map(_img_xys, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        # if self.hp.get("CACHE_INPUT") and not self.hp.get("NO_GROUP_DATA"):
        #     ds = ds.cache()
        #
        # if not FLAGS.debug:
        #     ds = ds.shuffle(2 ** 8)

        return ds
