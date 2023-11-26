import jax
import jax.numpy as jnp
from absl import flags
from einops import rearrange
from functools import partial
from jax.tree_util import tree_map

from chemise.traning.basic_trainer import Batch

flags.DEFINE_boolean("inc_local", default=True, help="Include Local Vector")
flags.DEFINE_boolean("inc_globcv", default=True, help="Include Local Vector")
flags.DEFINE_boolean("new_fold_extract", default=True, help="Use the old fold extract method")
flags.DEFINE_boolean("on_dev_unbatch", default=False, help="Ammortize data load costs and unbatch on device,"
                                                           " only works with vector runners and is hacky")

FLAGS = flags.FLAGS

FOLDS = [[1190, 1395, 1005, 17314],
         [458, 471, 918, 212],
         [1467, 56424, 862, 1161],
         [384, 643, 1007, 56963],
         [534, 55827, 235, 440]]


# @partial(jax.pmap, in_axes=(0, None), axis_name="batch")

@partial(jax.pmap)
@partial(jax.jit)
def make_vec_list(data):
    """
    Take a set of dataset superbatches of various sizes
    Pad them all to be the same size and concat along the vector axis
    Explode out the super batch into a list of batches
    ie [superbatch, batch, vector, ...] -> [[batch, vector, ...] * superbatch]

    Args:
        data: A tuple of size n pytrees with leafs of shape [superbatch, batch, vector, ...]
                superbatch and vector do not have to be the same size

    Returns:
        A list of len superbatch
    """
    # Calculate the max batch size of all elements
    batches = [el[-1].shape[0] for el in data]
    max_batches = max(batches)

    # Pad all datasets to be the same superbatch size
    def pad(x):
        batch_pad = (max_batches - x.shape[0] % max_batches) % max_batches
        p = jnp.pad(x, [(0, batch_pad), *[(0, 0)] * len(x.shape[1:])])
        return p
    padded = [tree_map(pad, el) for el in data]

    # Merge the datasets along the vector axis
    merged = jax.tree_util.tree_map(lambda *v: jnp.concatenate(v, axis=2), *padded)

    # Split out into a list of len superbatch
    leaves, treedef = jax.tree_util.tree_flatten(merged)
    output = [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]
    return output


@partial(jax.pmap, in_axes=(0,0,None), donate_argnums=[0])
def extract(x, rmask, c):
    """Shuffle the batch and split out the superbatch

    Args:
        x:
        rmask:
        c:

    Returns:

    """
    key = jax.random.fold_in(jax.random.PRNGKey(0), c)
    r = jax.random.permutation(key, rmask.shape[0])
    global_shuffled = tree_map(jax.jit(lambda n: n[r]), x)
    global_batched = tree_map(jax.jit(lambda n: rearrange(n, "(b s) ... -> b s ...", s=FLAGS.batch_size)), global_shuffled)
    return global_batched


MODE_CODE = {"pass": 0, "local": 1, "global": 2, "cv": 3, "kn": 4, "global++": 5}
def add_mode(xys, mode):
    xs = xys[0]
    xs["mode"] = jnp.where(xys[0]["plant"] == 0, 0, mode)
    return (xs, *xys[1:])

@jax.pmap
def add_kn(xys):
    def _add_kn(xs):
        xs = {k: v for k, v in xs.items()}
        xs["irradiance_in"] = xs["irradiance_in_kn"]
        return xs

    kn_extract = (_add_kn(xys[0]), *xys[1:])
    kn_extract = add_mode(kn_extract, MODE_CODE["kn"])
    cv_kn = tree_map(lambda *x: jnp.concatenate(x, axis=0), xys, kn_extract)
    return cv_kn


@partial(jax.pmap, in_axes=(0, 0, None, None, None), static_broadcasted_argnums=(3))
def fold_extract(x, rmask, fold_idx, train, c):
    """Extract and shuffle data for a given fold."""
    # Pad with zero at idx 0, this is to deal with the end of batch cases.
    # Since argwhere needs a size, if fewer than exp_size els are returned, 0 is returned
    # 0 pad indicates a noop and so is safe, otherwise
    x = tree_map(lambda el: jnp.pad(el, [(1, 0), *[(0, 0)] * len(el.shape[1:])]), x)

    # Find mask of plants for fold
    plants = x[0]["plant"][:, 0, 0]
    fold_mask = (plants == fold_idx)
    fold_mask = jnp.any(fold_mask, axis=0)
    # If training is true, set foldmask to be all plants not in the fold
    fold_mask = jax.lax.cond(train, lambda v: jnp.logical_not(v), lambda v: v, fold_mask)

    # Find the indexes to extract
    exp_size = 16 if train else 4  # Fix sizes as we only use 20 plants
    exp_size = int((exp_size * (x[0]["plant"].shape[0] -1)) / 20)
    fold_index = jnp.argwhere(fold_mask, size=exp_size)[:, 0]

    # Extract the fold indexes
    x = tree_map(lambda n: n[fold_index], x)

    # Shuffle data
    key = jax.random.fold_in(jax.random.PRNGKey(0), c)
    r = jax.random.permutation(key, fold_index.shape[0])
    global_shuffled = tree_map(jax.jit(lambda n: n[r]), x)
    global_shuffled = add_mode(global_shuffled, MODE_CODE["cv"])

    # Split out the superbatch
    global_batched = tree_map(jax.jit(lambda n: rearrange(n, "(b s) ... -> b s ...", s=FLAGS.batch_size)), global_shuffled)
    return global_batched


@jax.pmap
def extract_tree(xys):
    local = tree_map(lambda l: rearrange(l, "(b s) plant ... -> b s plant ...", s=FLAGS.batch_size), xys)
    local = add_mode(local, MODE_CODE["local"])
    glob = tree_map(lambda x: rearrange(x, "b plant ... -> (plant b) 1 ..."), xys)
    glob = add_mode(glob, MODE_CODE["global"])
    return local, glob

def unbatch(xys: Batch, c: int, train: bool):
    local_vec, global_explode = extract_tree(xys)
    ret = ()
    if FLAGS.inc_local:
        ret = (local_vec,)

    if FLAGS.inc_globcv:
        # Global shape
        zero_mask = global_explode[-1][:, :, 0, 0]
        g_v = extract(global_explode, zero_mask, c)
        # CV and KN extract
        cvs = []
        for f in FOLDS:
            pf = jnp.reshape(jnp.array(f), (4, 1))
            fold_mask = fold_extract(global_explode, zero_mask, pf, train, c)
            if not train:
                # If not training add KN
                fold_mask = add_kn(fold_mask)
            cvs.append(fold_mask)
        ret = (g_v, *cvs, *ret)
    return ret


def on_dev_shape(xys: Batch, c: int, train: bool):
    """To help keep the GPU fed we can load super batches and then reshape them on the device

    The input data comes in as
    BIG_BATCH, PLANT (TS ...) all unshuffled

    Output is a list of len BIG where all the elements have been shuffled
    [BATCH, PLANT, ....]

    melt the data for Global
    melt and filter the data for CV and KN

    Args:
        xys: The data
        c: some noise to mix into the shuffle
        train: if training or not

    Returns:

    """
    ub = unbatch(xys, c, train)
    ubs = make_vec_list(ub)
    return ubs