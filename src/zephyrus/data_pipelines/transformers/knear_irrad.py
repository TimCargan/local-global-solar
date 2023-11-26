import math

import tensorflow as tf
from zephyrus.data_pipelines.transformers.plant_fold_transform import PlantFold
from haversine import haversine

HOUR = 60 * 60


class KN_Irrad:
    # Precomputed conversion from lat long to PX location in the form (col/row)
    plants = PlantFold.plants

    NUM_PLANTS = 16

    def __init__(self, cache=False, num_folds = 5):
        self.cache = cache
        self.fold_dict = {k: v % num_folds for v, k in enumerate(self.plants.keys())}
        self.plant_near_plant = {p: self.near(p) for p in self.plants}

        self.near_lookup = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=list(self.plant_near_plant.keys()),
                values=list(self.plant_near_plant.values()))
            , "IDK")

    def near(self, plant):
        m_dist = 100000000000000000000000000
        ploc = self.plants[plant]
        pfold = self.fold_dict[plant]
        for p in self.plants:
            if p == plant or self.fold_dict[p] == pfold:
                continue
            elif (t_dist := self.dist(ploc, self.plants[p])) < m_dist:
                m_dist = t_dist
                near = p
        return near

    def dist(self, p1, p2):
        return haversine(p1, p2)
        #return math.sqrt(sum([(p1[i] - p2[i]) ** 2 for i in range(2)]))

    def __call__(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        if self.cache:
            ds = ds.cache()

        # group by time 0 and read in corresponding image set
        @tf.function
        def _read(key, ds: tf.data.Dataset):
            return ds.batch(self.NUM_PLANTS, drop_remainder=True)

        ds = ds.group_by_window(key_func=lambda x, y: x["ts"][0], window_size=self.NUM_PLANTS, reduce_func=_read)

        # Expload out per plant and center crop the image over the plant
        @tf.function
        def _expose(xs, ys):
            # This will un-batch the first axis of the x and y so it goes back to a dataset per plant
            ds = tf.data.Dataset.from_tensor_slices((xs, ys))

            # lookup plant index in xs, this is slow and bad but and hacky but works so just use it

            def idx(plant):
                plants = xs["plant"][:,0]
                i = -1
                for ix in range(self.NUM_PLANTS):
                    if plants[ix] == plant:
                        i = ix
                return i

            def add_near_irrad(x, y):
                near_plant = self.near_lookup.lookup(x["plant"][0])
                near_plant_i = idx(near_plant)
                new_irrad = xs["irradiance"][near_plant_i,:]
                x['_old_y_irrad'] = x['irradiance']
                x['irradiance'] = new_irrad
                return (x, y)

            return ds.map(add_near_irrad, num_parallel_calls=tf.data.AUTOTUNE)

        return ds.interleave(_expose, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE)


