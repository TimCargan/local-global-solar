import absl.app
import numpy as np
from absl import flags
from einops import rearrange

import zephyrus.utils.zeph_vec_unbatch as ub

flags.DEFINE_integer("batch_size", default=128, help="Batch size to train for (eval is sometimes 8*bs)")

absl.app._register_and_parse_flags_with_usage()
FLAGS = flags.FLAGS


def test_fold_extract():
    FLAGS.batch_size = 1  # Set flag
    plants = np.arange(20) + 1
    plants = rearrange(plants, "x -> 1 x 1 1")
    x = ({"plant": plants}, plants)
    filter = np.array([[1], [4], [8], [16]])

    # Ensure that filter works in test mode
    res = ub.fold_extract(x, None, filter, False, 0)
    assert np.all(np.unique(res[1]) == np.unique(filter))

    # Test that train filter works
    res = ub.fold_extract(x, None, filter, True, 0)
    assert set(list(np.unique(res[1]))) == {0, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19}

    # Test for filters that are smaller than exp size
    filter = np.array([[16]])
    res = ub.fold_extract(x, None, filter, False, 0)
    assert np.all(np.unique(res[1]) == np.array([0, 16]))

