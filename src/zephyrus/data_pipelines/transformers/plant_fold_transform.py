from abc import abstractmethod
from collections.abc import Sequence

import tensorflow as tf

old_plants = {
    b'rosedew': (51.39693832397461, -3.4709300994873047),
    b'newnham': (50.402259826660156, -4.039949893951416),
    b"far dane's": (53.32658004760742, -0.704289972782135),
    b'moor': (52.81214141845703, -2.8645100593566895),
    b'caegarw': (51.537139892578125, -3.7144598960876465),
    b'asfordby a': (52.778011322021484, -0.9376500248908997),
    b'somersal solar farm': (52.90476989746094, -1.8042000532150269),
    b'lains farm': (51.2010612487793, -1.616760015487671),
    b'bake solar farm': (50.40196990966797, -4.356460094451904),
    b'grange farm': (52.750648498535156, 0.0565200001001358),
    b'magazine': (52.570621490478516, -1.5824899673461914),
    b'kelly green': (50.54756164550781, -4.752530097961426),
    b'ashby': (52.75457000732422, -1.492859959602356),
    b'nailstone': (52.66413116455078, -1.3697400093078613),
    b'combermere farm': (52.98828887939453, -2.601870059967041),
    b'box road': (51.718570709228516, -2.361959934234619),
    b'crumlin': (51.67448043823242, -3.1556200981140137),
    b'asfordby b': (52.78329849243164, -0.9357799887657166),
    b'roberts wall solar farm': (51.67116165161133, -4.746419906616211),
    b'kirton': (52.93244171142578, -0.09311000257730484),
    b'moss electrical': (51.45643997192383, 0.20656999945640564),
    b'caldecote': (52.201839447021484, -0.21804000437259674)
}

class PlantFold(Sequence):
    # Plant Lat Lons
    plants = {
        #b'rosedew': (51.39693832397461, -3.4709300994873047),
        b'newnham': (50.402259826660156, -4.039949893951416),
        b"far dane's": (53.32658004760742, -0.704289972782135),
        b'moor': (52.81214141845703, -2.8645100593566895),
        #b'caegarw': (51.537139892578125, -3.7144598960876465),
        b'asfordby a': (52.778011322021484, -0.9376500248908997),
        #b'somersal solar farm': (52.90476989746094, -1.8042000532150269),
        b'lains farm': (51.2010612487793, -1.616760015487671),
        b'bake solar farm': (50.40196990966797, -4.356460094451904),
        b'grange farm': (52.750648498535156, 0.0565200001001358),
        b'magazine': (52.570621490478516, -1.5824899673461914),
        #b'kelly green': (50.54756164550781, -4.752530097961426),
        b'ashby': (52.75457000732422, -1.492859959602356),
        b'nailstone': (52.66413116455078, -1.3697400093078613),
        #b'combermere farm': (52.98828887939453, -2.601870059967041),
        b'box road': (51.718570709228516, -2.361959934234619),
        #b'crumlin': (51.67448043823242, -3.1556200981140137),
        b'asfordby b': (52.78329849243164, -0.9357799887657166),
        b'roberts wall solar farm': (51.67116165161133, -4.746419906616211),
        b'kirton': (52.93244171142578, -0.09311000257730484),
        #b'moss electrical': (51.45643997192383, 0.20656999945640564),
        b'caldecote': (52.201839447021484, -0.21804000437259674),
        b'clapham': (52.16442, -0.48715)
    }

    def __init__(self, fold_index, num_folds=5, determ=True):
        self.fold_index = fold_index
        self.num_folds = num_folds
        self.fold_dict = {k: v % num_folds for v, k in enumerate(self.plants.keys())}

        self.fold_lookup = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=list(self.fold_dict.keys()),
                values=list(self.fold_dict.values()))
            , num_folds+1)

    def in_fold(self, index):
        def f(ds:tf.data.Dataset):
            return ds.filter(lambda x,y: self.fold_lookup.lookup(x["plant"][0]) == index)
        return f

    def not_in_fold(self, index):
        def f(ds:tf.data.Dataset):
            return ds.filter(lambda x,y: self.fold_lookup.lookup(x["plant"][0]) != index)
        return f

    def __getitem__(self, i: int):
        return self.in_fold(i)

    def __len__(self):
        return self.num_folds

    def __call__(self, ds: tf.data.Dataset):
        return ds

    @property
    def __dict__(self):
        rep = {}
        return rep
