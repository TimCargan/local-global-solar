import jax.tree_util
import tensorflow as tf
from absl import flags

from zephyrus.data_pipelines.parsers.EUMETSAT import EuMetSat

FLAGS = flags.FLAGS

MIN = 60
HOUR = 60 * MIN

old_plants = {b'rosedew': (250, 369), b'newnham': (234, 407), b"far dane's": (332, 295), b'moor': (268, 314),
              b'caegarw': (243, 363), b'asfordby a': (325, 316), b'somersal solar farm': (299, 311),
              b'lains farm': (305, 376), b'bake solar farm': (224, 407), b'grange farm': (354, 317),
              b'magazine': (306, 324), b'kelly green': (213, 402), b'ashby': (309, 317), b'nailstone': (312, 320),
              b'combermere farm': (276, 308), b'box road': (283, 356), b'crumlin': (260, 358),
              b'asfordby b': (325, 316), b'roberts wall solar farm': (213, 358), b'kirton': (350, 310),
              b'moss electrical': (359, 367), b'caldecote': (346, 338)}

new_plants = {b'23': (267, 78), b'32': (262, 97), b'40': (213, 108), b'44': (222, 104), b'48': (237, 106),
              b'52': (187, 120),
              b'54': (167, 107), b'66': (196, 130), b'67': (208, 125), b'79': (236, 122), b'103': (206, 160),
              b'105': (214, 158),
              b'113': (240, 145), b'117': (245, 149), b'132': (248, 129), b'137': (255, 126), b'145': (252, 158),
              b'150': (269, 150),
              b'160': (287, 146), b'161': (288, 145), b'163': (289, 148), b'177': (286, 159), b'181': (262, 174),
              b'212': (243, 179),
              b'235': (268, 177), b'236': (268, 175), b'253': (258, 193), b'268': (282, 203), b'307': (305, 231),
              b'315': (305, 214),
              b'326': (306, 239), b'332': (311, 234), b'342': (311, 264), b'16725': (360, 304), b'346': (316, 267),
              b'358': (333, 255),
              b'370': (339, 274), b'373': (347, 265), b'381': (336, 295), b'384': (337, 300), b'386': (338, 306),
              b'393': (347, 304),
              b'395': (357, 312), b'399': (352, 286), b'405': (357, 289), b'409': (369, 321), b'413': (372, 328),
              b'421': (385, 309),
              b'24996': (270, 343), b'429': (392, 317), b'433': (402, 319), b'435': (369, 336), b'440': (381, 341),
              b'447': (378, 348),
              b'455': (355, 336), b'456': (345, 330), b'458': (335, 345), b'461': (339, 337), b'465': (340, 342),
              b'471': (342, 353),
              b'487': (365, 356), b'498': (377, 363), b'513': (298, 276), b'516': (300, 276), b'523': (307, 276),
              b'533': (317, 275),
              b'534': (314, 274), b'535': (319, 275), b'542': (304, 317), b'554': (316, 314), b'556': (316, 307),
              b'562': (323, 289),
              b'576': (333, 318), b'583': (339, 322), b'586': (296, 328), b'595': (313, 332), b'596': (305, 338),
              b'605': (306, 355),
              b'607': (311, 346), b'613': (320, 360), b'622': (286, 307), b'643': (274, 315), b'651': (281, 316),
              b'657': (292, 340),
              b'669': (268, 336), b'671': (276, 349), b'673': (275, 341), b'675': (276, 367), b'676': (277, 364),
              b'692': (303, 351),
              b'708': (339, 366), b'709': (340, 363), b'719': (338, 372), b'721': (343, 366), b'723': (344, 366),
              b'724': (346, 381),
              b'726': (350, 372), b'743': (392, 379), b'744': (366, 373), b'765': (386, 370), b'775': (392, 371),
              b'779': (325, 391),
              b'795': (344, 390), b'811': (362, 388), b'825': (320, 361), b'836': (328, 363), b'838': (329, 369),
              b'842': (298, 393),
              b'846': (306, 394), b'847': (306, 378), b'858': (317, 392), b'862': (325, 375), b'868': (327, 377),
              b'869': (330, 373),
              b'876': (314, 400), b'886': (294, 365), b'888': (299, 376), b'889': (301, 378), b'17278': (344, 333),
              b'908': (185, 213),
              b'918': (192, 174), b'17314': (307, 257), b'982': (238, 197), b'987': (243, 206), b'1005': (218, 212),
              b'1007': (218, 210),
              b'1023': (258, 218), b'1033': (235, 238), b'1035': (236, 233), b'1039': (207, 236), b'1055': (246, 249),
              b'1060': (260, 245),
              b'1070': (265, 233), b'1073': (270, 243), b'1074': (281, 247), b'1076': (277, 228), b'1078': (257, 264),
              b'1083': (273, 249),
              b'1085': (280, 242), b'1086': (283, 242), b'1090': (263, 277), b'1096': (267, 286), b'1105': (271, 268),
              b'1112': (271, 274),
              b'1115': (278, 283), b'1125': (288, 284), b'1132': (278, 304), b'1135': (285, 294), b'1137': (249, 297),
              b'1144': (265, 300),
              b'1145': (219, 297), b'1161': (213, 315), b'1171': (237, 304), b'1180': (247, 311), b'1190': (251, 317),
              b'1198': (218, 340),
              b'1205': (234, 329), b'1209': (236, 332), b'1215': (204, 357), b'1223': (215, 346), b'1226': (224, 357),
              b'1238': (251, 337),
              b'1255': (235, 362), b'1272': (259, 365), b'1285': (246, 381), b'1302': (275, 384), b'1319': (280, 403),
              b'1336': (231, 409),
              b'1346': (230, 381), b'1352': (238, 393), b'1367': (250, 407), b'1378': (252, 394), b'1383': (257, 389),
              b'1386': (167, 426),
              b'1393': (198, 419), b'1395': (196, 414), b'1415': (215, 403), b'1429': (146, 228), b'1448': (162, 235),
              b'1450': (169, 243),
              b'1467': (171, 223), b'1488': (171, 266), b'1490': (173, 250), b'1502': (181, 259), b'1504': (173, 257),
              b'1517': (186, 247),
              b'1523': (183, 243), b'1529': (188, 243), b'1534': (161, 260), b'1543': (130, 242), b'1568': (128, 254),
              b'16596': (312, 261),
              b'16611': (246, 343), b'16630': (364, 322), b'18903': (135, 140), b'18974': (150, 173),
              b'19144': (349, 364), b'19172': (182, 143),
              b'19187': (303, 327), b'19188': (366, 350), b'19204': (326, 292), b'19206': (251, 369),
              b'19211': (150, 123), b'19260': (254, 195),
              b'17176': (329, 358), b'61986': (387, 328), b'62041': (252, 394), b'17224': (306, 397),
              b'4911': (390, 320), b'17309': (262, 288),
              b'17336': (225, 176), b'17344': (327, 247), b'30103': (240, 145), b'30270': (185, 137),
              b'30437': (130, 263), b'30523': (297, 229),
              b'30620': (346, 379), b'30690': (294, 302), b'55511': (289, 294), b'25728': (402, 319),
              b'25729': (392, 371), b'55827': (253, 153),
              b'23417': (169, 204), b'56424': (296, 328), b'16581': (290, 126), b'16588': (362, 366),
              b'24102': (307, 329), b'16589': (285, 219),
              b'24125': (219, 195), b'56963': (148, 224), b'57063': (129, 256), b'57199': (282, 293)}


def stack_els(ls):
    tree = jax.tree_util.tree_structure(ls[0])
    flat = [jax.tree_util.tree_flatten(d)[0] for d in ls]
    flat_n = [[n[l] for n in flat] for l in range(len(flat[0]))]
    stacked = [tf.stack(x, axis=0) for x in flat_n]
    return jax.tree_util.tree_unflatten(tree, stacked)

class EU_Images:
    # Precomputed conversion from lat long to PX location in the form (col/row)

    plants = new_plants
    X_INDEX, Y_INDEX = 0, 1
    NUM_PLANTS = 24

    def __init__(self, im_reader: EuMetSat, crop_size=128):
        self.im_reader = im_reader
        self.crop_size = crop_size
        plants_pyx = self.plants
        self.plant_px = tf.lookup.StaticHashTable(self._make_kv(plants_pyx, self.X_INDEX), 0)
        # self.plant_py = tf.lookup.StaticHashTable(self._make_kv(plants_pyx, self.Y_INDEX), 0)
        self.plant_py = self.dict_to_lookup({int(k): v[self.Y_INDEX] for k, v in self.plants.items()},
                                            value_dtype=tf.int32)

        PLANTS = ["17314", "1005", "56963", "862", "918", "56424", "384", "534", "643", "1395", "212", "1190",
                  "458", "1467", "55827", "440", "235", "471", "1007", "1161"]
        self.plant_2_plant = self.dict_to_lookup({int(p): int(p) for p in PLANTS}, value_dtype=tf.int32)

        geo_info = im_reader.geo_info  # TODO: Dont hard code this
        self.X_MIN, self.X_MAX, self.Y_MIN, self.Y_MAX = geo_info["x_min"], geo_info["x_max"], geo_info["y_min"], \
                                                         geo_info["y_max"]

    def dict_to_lookup(self, py_dict: dict, key_dtype=tf.int64, value_dtype=tf.int64):
        tf_lookup = tf.lookup.experimental.DenseHashTable(key_dtype=key_dtype, value_dtype=value_dtype,
                                                          default_value=-1, empty_key=-1, deleted_key=-2)
        ts_int = tf.constant(list(py_dict.keys()), dtype=key_dtype)
        ts_path = tf.constant(list(py_dict.values()), dtype=value_dtype)
        tf_lookup.insert(ts_int, ts_path)
        return tf_lookup

    @staticmethod
    def _make_kv(kv: dict, i: int) -> tf.lookup.KeyValueTensorInitializer:
        string_keys = tf.constant(list(kv.keys()))
        keys = tf.strings.to_number(string_keys, out_type=tf.int64, name="cast_plant_name")
        return tf.lookup.KeyValueTensorInitializer(
            keys=keys,
            values=[xy[i] for xy in kv.values()])

    def pad_img_dict(self, imgs):
        c_imgs = {}
        for l in imgs:
            c_imgs[l] = self.pad_img(imgs[l])
        return c_imgs

    def pad_img(self, img):
        ty = tx = self.crop_size
        pad = (0, (tx // 2), self.im_reader.IMG_SIZE[0], self.im_reader.IMG_SIZE[1] + (tx // 2))
        p_img = tf.image.pad_to_bounding_box(img, *pad)
        return p_img

    @tf.function
    def center_crop_dict(self, imgs, plant):
        c_imgs = {}
        py = self.plant_py.lookup(plant, name="lookup_latlon_y")
        px = self.plant_px.lookup(plant, name="lookup_latlon_x")
        for l in imgs:
            c_imgs[l] = self._center_crop_img(imgs[l], py, px)
        return c_imgs

    @tf.function
    def center_crop_img(self, img, plant):
        py = self.plant_py.lookup(plant, name="lookup_latlon_y")
        px = self.plant_px.lookup(plant, name="lookup_latlon_x")
        return self._center_crop_img(img, py, px)

    @tf.function
    def _center_crop_img(self, img, py, px):
        ty = tx = self.crop_size
        # Adjust to cropped image in mem
        py = py - (self.Y_MIN - (ty // 2))
        px = px - (self.X_MIN - (tx // 2))

        fuzz = FLAGS.img_crop_fuzz
        if fuzz > 0:
            py = py + tf.random.uniform([1], maxval=fuzz, minval=-fuzz, dtype=tf.int32)[0]
            px = px + tf.random.uniform([1], maxval=fuzz, minval=-fuzz, dtype=tf.int32)[0]

        crop = (py - (ty // 2), px - (tx // 2), ty, tx)
        c_img = tf.image.crop_to_bounding_box(img, *crop)

        # crop = (py - (ty // 2), px - (tx // 2), py + (ty // 2), px + (tx // 2))
        # c_img = img[..., crop[0]:crop[2], crop[1]:crop[3], :]
        # c_img = tf.reshape(c_img, (*img.shape[:-3], ty, tx, c_img.shape[-1]))

        return c_img

    @staticmethod
    def sort_group(*xy):
        PLANTS = ["17314", "1005", "56963", "862", "918", "56424", "384", "534", "643", "1395", "212", "1190",
                  "458", "1467", "55827", "440", "235", "471", "1007", "1161"]
        plant_idx = xy[0]["plant"][:, 0]
        idx = tf.argsort(plant_idx)
        return jax.tree_util.tree_map(lambda x: tf.gather(x, idx), xy)

        #
        # slice_order = [] #[tf.argmax(plant_idx == int(p)) for p in PLANTS if tf.reduce_any(plant_idx == int(p))]
        # for p in PLANTS:
        #     comp = plant_idx == int(p)
        #     idx = tf.argmax(comp)
        #     if tf.reduce_any(comp):
        #         slice_order.append(idx)
        #
        # sliced = [jax.tree_util.tree_map(lambda x: x[i], xy) for i in slice_order]
        # return stack_els(sliced)

    @staticmethod
    def padd_sorted_group(*xy):
        """ Assuming shape [Plant, Els]"""
        PLANTS = ["17314", "1005", "56963", "862", "918", "56424", "384", "534", "643", "1395", "212", "1190",
                  "458", "1467", "55827", "440", "235", "471", "1007", "1161"]

        PLANTS = sorted(map(int, PLANTS))
        zero = jax.tree_util.tree_map(lambda x: tf.zeros_like(x[0]), xy)

        sort = xy
        plants = sort[0]["plant"]
        max_sidx = tf.shape(plants)[0]
        s_idx = 0
        lookup = []
        for p in PLANTS:
            pred = tf.reduce_all(plants[s_idx] == int(p))
            ext, inc = tf.cond(pred,
                                lambda: (jax.tree_util.tree_map(lambda x: x[s_idx], sort), True),
                                lambda : (zero, False)
                               )
            lookup.append((*ext, 0, inc))
            if inc and s_idx < max_sidx - 1:  # Check max size to avoid oob index error on last opp
                s_idx = s_idx + 1

        merged = stack_els(lookup)
        return *merged[:-1], tf.expand_dims(merged[-1], axis=-1)

    @staticmethod
    def group(ds: tf.data.Dataset) -> tf.data.Dataset:
        # group by time 0 and read in corresponding image set
        def _read(key, ds: tf.data.Dataset):
            return ds.batch(20)

        def _key_func(x, y):
            return tf.reshape(x["ts"], [-1])[0]

        ds = ds.group_by_window(key_func=_key_func, window_size=20, reduce_func=_read)
        return ds

    def filter(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = ds.filter(lambda x, *_: self.im_reader.has_img(tf.reshape(x["ts"], [-1])[0] - HOUR))  # -1 hour
        ds = ds.filter(lambda x, *_: self.im_reader.has_img(tf.reshape(x["ts"], [-1])[0]))
        ds = ds.filter(lambda x, *_: self.im_reader.has_img(tf.reshape(x["ts"], [-1])[0] + HOUR))  # +1 hour
        return ds

    def extra_shape(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = ds.map(self._extra_shape, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        return ds

    def _location_square(self, plant):
        plant = plant[..., 0]

        def _n(x):
            x = tf.round(tf.cast(x, tf.float32) / 500.0 * 32)
            return tf.cast(x, tf.int32)

        py = self.plant_py.lookup(plant, name="lookup_latlon_y")
        py = tf.reshape(py, tf.shape(plant))

        px = self.plant_px.lookup(plant, name="lookup_latlon_x")
        px = tf.reshape(px, tf.shape(plant))

        px = tf.expand_dims(tf.one_hot(_n(px), 32), axis=-2)
        py = tf.expand_dims(tf.one_hot(_n(py), 32), axis=-1)
        pxy = tf.expand_dims(px * py, axis=-1)
        return pxy

    def _extra_shape(self, x, *y):
        key = sorted(x.keys())
        nx = {f: x[f] for f in key}

        # Lat Lon adjust for use in embedding
        em_keys = [('Latitude', 10, -50), ('Longitude', 10, 7)]
        for n, size, min in em_keys:
            # Use hacky version of embedding as standard breaks multi-gpu
            e_add = x[n] + min
            e_add = tf.cast(e_add, tf.uint8, name=f"{n}_cast")
            nx[f"{n}_hackEMB"] = tf.one_hot(e_add, size)

        # Square Location
        nx["plant_em"] = self._location_square(nx["plant"])
        return (nx, *y)

    def v_read(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(self._v_read, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    def _v_read(self, xs, ys):
        key = sorted(xs.keys())
        nxs = {f: xs[f] for f in key}
        # This will un-batch the first axis of the x and y so it goes back to a dataset per plant
        # add sample weight of # tf.math.log1p(y) + 1e-4
        steps = FLAGS.warmup_steps + FLAGS.output_steps
        key_r_start = xs["ts"][0, 0]

        start_time = key_r_start + (FLAGS.img_min_offset * MIN)
        step_size = HOUR // FLAGS.img_update_feq
        keys = start_time + tf.range(0, steps * HOUR, step_size, dtype=tf.int64)

        imgs = self.im_reader.dynamic_read(keys)
        # nxs["keys"] = keys  # tf.tile(tf.expand_dims(keys, axis=0), (tf.shape(xs["ts"])[0], 1))

        def _v_slice(p):
            if FLAGS.img_center_crop:
                return self.center_crop_img(imgs, tf.reshape(p, [-1])[0])
            return imgs

        sliced = tf.vectorized_map(_v_slice, xs["plant"])

        # nxs["raw_img"] = imgs
        nxs["img_stack"] = sliced
        return nxs, ys
