import os.path
from datetime import datetime
from multiprocessing.pool import ThreadPool

import pandas as pd
import tensorflow as tf
from absl import flags
from absl import logging as l

from zephyrus.data_pipelines.utils import pack_tf
from zephyrus.utils.translator import get_path

flags.DEFINE_integer("img_lookup_initial_num_buckets", default=2 ** 16, help="Default size for image lookup buffer")
flags.DEFINE_string("img_lookup_min_date", default="2015-01-01 00:00", help="Min data for image files")
flags.DEFINE_string("img_lookup_max_date", default="2020-12-01 00:00", help="Max data for image files")
flags.DEFINE_integer("img_lookup_numchannels", default=12, help="How many of the channels to load, can be 4, 8 or 12",
                     lower_bound=4, upper_bound=12)
flags.DEFINE_bool("img_lookup_skipfill", default=False, help="Skip filling img lookup")
flags.DEFINE_bool("img_center_crop", default=True, help="Center crop images over plant")
flags.DEFINE_integer("img_crop_fuzz", default=0, help="Fuzz where image get cropped by a random amount +-")

flags.register_validator("img_lookup_numchannels", lambda x: x % 4 == 0, message='Must be a multiple of 4')

FLAGS = flags.FLAGS

RAW_IMG_SIZE = (500, 500, 1)

KNOWN_BAD_TS = [0, 1425211200, 1425297600, 1425384000, 1425385800, 1430208000, 1430209800, 1430222400, 1430224200,
                1435096800, 1435708800, 1435710600, 1445421600, 1447558200, 1447560000, 1447561800, 1447563600,
                1447565400, 1447567200, 1447569000, 1447570800, 1447572600, 1447574400, 1447662600, 1456574400,
                1456660800, 1456747200, 1457433000, 1465394400, 1475919000, 1476291600, 1476534600, 1476536400,
                1476538200, 1476540000, 1476541800, 1476545400, 1476549000, 1476552600, 1476556200, 1476559800,
                1476563400, 1476567000, 1476570600, 1476574200, 1476577800, 1476581400, 1476585000, 1476588600,
                1476592200, 1476595800, 1476599400, 1476603000, 1476606600, 1476610200, 1476613800, 1476617400,
                1491600600, 1492900200, 1510038000, 1510039800, 1510041600, 1525636800, 1529487000, 1530624600,
                1531265400, 1531267200, 1531269000, 1531270800, 1531272600, 1531274400, 1531276200, 1531278000,
                1548171000, 1548172800, 1548763200, 1558706400, 1560495600, 1570530600, 1573480800, 1573482600,
                1583325000, 1587220200, 1602331200, 1602417600, 1609459200]

# Channels order from the smallest wavelength to the biggest ex HRV is at index 0
img_layers = ["HRV", "VIS006", "VIS008", "IR_016",
              "IR_039", "WV_062", "WV_073", "IR_087",
              "IR_097", "IR_108", "IR_120", "IR_134"]

# Sorted alphabetically, leggacy ordering
img_layers_alfa = ["HRV", "IR_016", "IR_039", "IR_087",
                   "IR_097", "IR_108", "IR_120", "IR_134",
                   "VIS006", "VIS008", "WV_062", "WV_073"]

# Standard Global min max xy values
G = {"x_min": 148,
     "x_max": 381,
     "y_min": 153,
     "y_max": 414}

# ALL plant min max xy values
GPP = {"x_max": 392,  # Known max plant x value
       "x_min": 128,  # Known min plant x value
       "y_max": 414,  # Known max plant y value
       "y_min": 78}  # Known min plant y value


def dict_to_lookup(py_dict: dict, key_dtype=tf.int64, value_dtype=tf.int64):
    tf_lookup = tf.lookup.experimental.DenseHashTable(key_dtype=key_dtype, value_dtype=value_dtype,
                                                      default_value=-1, empty_key=-1, deleted_key=-2,
                                                      checkpoint=True)
    ts_int = tf.constant(list(py_dict.keys()), dtype=key_dtype)
    ts_path = tf.constant(list(py_dict.values()), dtype=value_dtype)
    tf_lookup.insert(ts_int, ts_path)
    return tf_lookup


def crop_cals(slice_size=64):
    SLICE_SIZE = slice_size  # 128
    ## This is to save mem, only hold enough image data to slice for plants

    geo_info = GPP if FLAGS.gpp or FLAGS.run_all_modes else G
    l.info(f"Set Geo info to: {geo_info}")
    x_min, x_max, y_min, y_max = geo_info["x_min"], geo_info["x_max"], geo_info["y_min"], geo_info["y_max"]
    x_diff = x_max - x_min
    y_diff = y_max - y_min
    IMG_SIZE = (y_diff + SLICE_SIZE, x_diff + SLICE_SIZE)
    IMG_SIZE = tuple([min(RAW_IMG_SIZE[i], IMG_SIZE[i]) for i in range(2)])
    RESIZE = (max(0, y_min - SLICE_SIZE // 2), max(0, x_min - SLICE_SIZE // 2), *IMG_SIZE)
    return IMG_SIZE, RESIZE


class EuMetSat:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print('Creating the object')
            cls._instance = super(EuMetSat, cls).__new__(cls)
            cls.init(cls, *args, **kwargs)
        return cls._instance

    def init(self, slice_size: int = 64, **kwargs):
        img_base_dir = os.path.join(get_path("data"), "EUMETSAT", "UK-EXT")
        l.info(f"Loading Images ({img_base_dir=})")
        self.img_base_path = img_base_dir

        self.geo_info = GPP if FLAGS.gpp or FLAGS.run_all_modes else G
        if FLAGS.img_crop_fuzz > 0:
            self.geo_info["x_min"] = self.geo_info["x_min"] - FLAGS.img_crop_fuzz
            self.geo_info["y_min"] = self.geo_info["y_min"] - FLAGS.img_crop_fuzz
            self.geo_info["x_max"] = self.geo_info["x_max"] + FLAGS.img_crop_fuzz
            self.geo_info["y_max"] = self.geo_info["y_max"] + FLAGS.img_crop_fuzz
        # This is such a bad way to do this
        self.min_date = datetime.fromisoformat(FLAGS.img_lookup_min_date)
        self.max_date = datetime.fromisoformat(FLAGS.img_lookup_max_date)
        date_rage = pd.date_range(self.min_date, self.max_date, freq="15min")
        self.date_dict = {int(ts.timestamp()): ts.strftime("/year=%Y/month=%m/day=%d/time=%H_%M") for ts in date_rage}

        # # Time Lookups
        self.YEARS = dict_to_lookup({int(ts.timestamp()): int(ts.year) for ts in date_rage})
        self.MONTHS = dict_to_lookup({int(ts.timestamp()): int(ts.month) for ts in date_rage})
        self.DAYS = dict_to_lookup({int(ts.timestamp()): int(ts.day) for ts in date_rage})
        self.HOURS = dict_to_lookup({int(ts.timestamp()): int(ts.hour) for ts in date_rage})
        self.MINUTES = dict_to_lookup({int(ts.timestamp()): int(ts.minute) for ts in date_rage})

        self.min_date_tf = tf.constant(int(self.min_date.timestamp()), tf.int64)
        self.max_date_tf = tf.constant(int(self.max_date.timestamp()), tf.int64)

        self.IMG_SIZE, self.RESIZE = crop_cals(slice_size)
        self.IMG_CHANNELS = (FLAGS.img_lookup_numchannels // 4)

        empty_img = tf.reshape(tf.zeros((*self.IMG_SIZE, self.IMG_CHANNELS), dtype=tf.int32),
                               [-1]) - 1  # negative 1 as default
        self.lookup = tf.lookup.experimental.DenseHashTable(key_dtype=tf.int64, value_dtype=tf.int32,
                                                            default_value=empty_img, empty_key=-1, deleted_key=-2,
                                                            checkpoint=True,
                                                            initial_num_buckets=FLAGS.img_lookup_initial_num_buckets)

        self.in_lookup = tf.lookup.experimental.DenseHashTable(key_dtype=tf.int64, value_dtype=tf.bool,
                                                               default_value=FLAGS.img_lookup_skipfill, empty_key=-1,
                                                               deleted_key=-2,
                                                               checkpoint=True,
                                                               initial_num_buckets=FLAGS.img_lookup_initial_num_buckets)

    def ts_to_path(self, ts):
        def pad(i):
            """
            Pad numbers less than 10 with a leading zero
            :param i: int to pad
            :return: int as a string with a padded zero
            """
            s = tf.strings.as_string(i)
            if i < 10:
                s = "0" + s
            return s

        path = "/year=" + pad(self.YEARS[ts])
        path += "/month=" + pad(self.MONTHS[ts])
        path += "/day=" + pad(self.DAYS[ts])
        path += "/time=" + pad(self.HOURS[ts]) + "_" + pad(self.MINUTES[ts])
        return path

    def exclude_list(self):
        """
        Returns: Set of timestamps that fall within the min_max date range but don't exist on disk
        """

        def _exists(kv):
            k, v = kv
            t_path = self.img_base_path + v
            for l in self.img_layers:
                path = t_path + f"/format={l}/img.png"
                if not os.path.exists(path):
                    return k
            return 0

        with ThreadPool(8) as p:
            ex = p.map(_exists, self.date_dict.items())
        return list(sorted(set(ex)))

    def has_img(self, ts):
        ls_max = ts < self.max_date_tf
        gt_min = ts > self.min_date_tf
        not_in_kb = tf.reduce_all(KNOWN_BAD_TS != ts)
        return tf.reduce_all([ls_max, gt_min, not_in_kb])

    def init_lookup(self):
        @tf.function
        def reader(d):
            for ts in d:
                if (ts % (3600 * 24 * 30)) == 0:
                    path = self.ts_to_path(ts)
                    tf.print("...Loading,  reading ", path, output_stream=l.info, end="")

        times = tf.constant(list(sorted(self.date_dict.keys())), dtype=tf.int64)
        d = tf.data.Dataset.from_tensor_slices(times)
        d = d.filter(self.has_img)
        d = d.map(self._add_imgs_to_lookup, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        l.info("Reading images into lookup")
        reader(d)
        l.info("Images loaded")

    @tf.function
    def _add_imgs_to_lookup(self, ts):
        ts_path = self.ts_to_path(ts)
        path = self.img_base_path + ts_path

        def _read_decode_img(l):
            f_path = path + f"/format={l}/img.png"
            img = tf.io.read_file(f_path)
            img = tf.io.decode_png(img, channels=1)
            img = tf.reshape(img, RAW_IMG_SIZE)
            img = tf.image.crop_to_bounding_box(img, *self.RESIZE)
            img = tf.image.resize(img, [*self.IMG_SIZE])
            return img

        if self.has_img(ts):
            imgs_raw = [_read_decode_img(l) for l in img_layers[:FLAGS.img_lookup_numchannels]]
            imgs_stack = tf.stack(imgs_raw, axis=-1)
            imgs_pack_u32 = pack_tf(imgs_stack)
            imgs_pack = tf.cast(imgs_pack_u32, tf.int32)
            imgs_flat = tf.reshape(imgs_pack, [-1])
            self.lookup.insert(ts, imgs_flat)

        self.in_lookup.insert(ts, True)
        return ts

    def dynamic_read(self, ts):
        keys = tf.reshape(ts, (-1,))
        is_loaded = self.in_lookup.lookup(keys, name="lookup_loaded")
        to_load = tf.logical_not(is_loaded)
        to_load = tf.reshape(to_load, tf.shape(keys))
        if tf.reduce_any(to_load):
            filter_keys = tf.boolean_mask(keys, to_load)
            tf.map_fn(self._add_imgs_to_lookup, filter_keys)

        imgs_packed = self.read_imgs_lookup(ts)
        return imgs_packed

    def read_imgs_lookup(self, ts):
        if tf.shape(ts).shape.dims[0] == 0:
            steps = 1
        else:
            steps = tf.shape(ts)[0]

        imgs_packed = self.lookup.lookup(ts, name="lookup_weather_imagess1")
        img = tf.reshape(imgs_packed, (steps, *self.IMG_SIZE, self.IMG_CHANNELS))  # padded
        return img
