import glob
import tensorflow as tf
from functools import reduce
from typing import List


def read_glob(path: str) -> tf.data.Dataset:
    ds_path = sorted(glob.glob(path))
    cl = len(ds_path)
    # Don't interleave if 2 or fewer reads, this is nice as it wil keep the cardinality etc
    if cl > 1:
        # Read the first file just to get el spec
        es = tf.data.Dataset.load(ds_path[0], compression="GZIP").element_spec
        paths = tf.data.Dataset.from_tensor_slices(ds_path)
        ds = paths.interleave(lambda p: tf.data.experimental.load(p, compression="GZIP", element_spec=es),
                          cycle_length=cl, block_length=2**10)
    else:
        ds = tf.data.Dataset.load(ds_path[0], compression="GZIP")
    return ds


def apply_list(ds: tf.data.Dataset, fns: List):
    """
    Apply ordered list of transformations to a dataset
    Args:
        ds: Dataset to apply functions too
        fns: list of transformation functions [DS -> DS]

    Returns: Dataset with all transformations applied

    """
    return reduce(tf.data.Dataset.apply, fns, ds)


# Plant Lat Lons
plants = {
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
