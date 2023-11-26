import math

import jax.numpy as jnp
import numpy as np
import pyresample as pr
import tensorflow as tf
from jaxtyping import UInt32, Array, UInt8
from pyproj import CRS

from zephyrus.data_pipelines.tile_utils import tilesUtils

# Constants\
HOUR_SECONDS = 60 * 60
DAY_SECONDS = 24 * 60 * 60  # seconds in a day
YEAR_SECONDS = 365.2425 * DAY_SECONDS  # seconds in a day year

def scale_cord(cord, o_range, t_range):
    """
    Scale from cords with variable range to a 0 ... t_range
    :param cord: value to convert
    :param o_range: a tuple (min, max) of cord range
    :param t_range: max value in zero based target range
    :return: Scaled cord value
    """
    _range = o_range[1] - o_range[0]
    scale = t_range / _range
    s_cord = (cord - o_range[0]) * scale
    return int(s_cord)


def lat_long_to_onehot_grid(lat, long,height=128,width=128,lat_range=(-14,3),long_range=(49,62)):
    """
    Create a 2D tensor with all zeros except for the cell corresponding to the given lat long
    :param lat: latitude (Lat -> Height)
    :param long: longitude (Long -> Width)
    :param height: output height
    :param width: output width
    :param lat_range: a tuple (min, max) of the range of latitude values
    :param long_range: a tuple (min, max) of the range of longitude values
    :return: tensors (height,width) of zeors where lat,long is 1
    """
    long_c = scale_cord(long, long_range, width)
    lat_c = scale_cord(lat, lat_range, height)
    return tf.scatter_nd([[lat_c,long_c]], [1], [height,width])


def latlonToTileId(plant):
    lat,lon = plant["Latitude"], plant["Longitude"]
    tiles7 = tilesUtils(7)
    pix = tiles7.fromLatLngToTilePixel(lat, lon, 7)
    area = {"tile_x": pix.tx, "tile_y" :pix.ty, "px_x": pix.px, "px_y" :pix.py}
    x = int((area["tile_x"] - 59) * 256 + area["px_x"])
    y = int((area["tile_y"] - 36) * 256 + area["px_y"])
    return x,y


def latlongtoimage(lat, lon, img=(500,500), rng=((61, 48), (-12,5))):
    """
    Lat is N/S 61,48,
    Lon is E/W -12, 5
    :param plant:
    :param img: a tuple of the image size in the form (height, width)
    :param rng: a tuple of area the image covers in the form ((N, S), (W,E))
    :return: a tuple (x,y) for pixel that is plant
    """
    def scale(point, source, size):
        source_range = source[1] - source[0]
        conv_p = ((point - source[0]) / source_range) * size
        return conv_p

    new_y = int(scale(lat, rng[0], img[0]))
    new_x = int(scale(lon, rng[1], img[1]))

    return new_x, new_y

def lat_lon_to_x_y(lat, lon):
    """
    Convert lat lon to image pixel locations
    Args:
        lat:
        lon:
    Returns: a tuple of the index in the form(cols/rows)
    """
    area_extent = (-12., 48., 5., 61.)
    area_id = "UK"  # Any ID
    proj_crs = CRS.from_user_input(4326)  # Target Projection EPSG:4326 standard lat lon geograpic
    output_res = (500, 500)  # Target res in pixels
    area_def = pr.geometry.AreaDefinition.from_extent(area_id, proj_crs, output_res, area_extent)
    return area_def.get_array_indices_from_lonlat(lon, lat)

def vectorize(speed, dir_deg=None, dir_rad=None):
    """
    Convert as speed and direction into its component vectors
    Args:
        speed: magnitude of vector
        dir_deg: direction in degrees
        dir_rad: direction in radians

    Returns: A tuple of (x_component, y_component)
    """
    # Convert to radians.
    if dir_rad is None:
        dir_deg = tf.cast(dir_deg, tf.float32)
        dir_rad = dir_deg * math.pi / 180

    speed = tf.cast(speed, tf.float32)
    # Calculate the x and y components.
    x_comp = speed * tf.math.cos(dir_rad)
    y_comp = speed * tf.math.sin(dir_rad)
    return x_comp, y_comp

def vectorize_np(speed, dir_deg=None, dir_rad=None):
    """
    Convert as speed and direction into its component vectors
    Args:
        speed: magnitude of vector
        dir_deg: direction in degrees
        dir_rad: direction in radians

    Returns: A tuple of (x_component, y_component)
    """
    # Convert to radians.
    if dir_rad is None:
        dir_rad = dir_deg * math.pi / 180

    # Calculate the x and y components.
    x_comp = speed * np.cos(dir_rad)
    y_comp = speed * np.sin(dir_rad)
    return x_comp, y_comp

def sin_cos_scale(col, scale):
    """
    Scale a cyclical value e.g day of year, to a sin and cos function
    Args:
        col: value to be scaled
        scale: periodicity e.g for a hour timestamp 24, or 365 for a day of year

    Returns: a tuple of (sin_component, cos_component)

    """
    col = tf.cast(col, tf.float32)
    sin = tf.math.sin(col * (2 * math.pi / scale))
    cos = tf.math.cos(col * (2 * math.pi / scale))
    return sin, cos

def sin_cos_scale_np(col, scale):
    """
    Scale a cyclical value e.g day of year, to a sin and cos function
    Args:
        col: value to be scaled
        scale: periodicity e.g for a hour timestamp 24, or 365 for a day of year

    Returns: a tuple of (sin_component, cos_component)

    """
    # col = np.cast(col, np.float32)
    sin = np.sin(np.multiply(col, (2 * math.pi / scale)))
    cos = np.cos(col * (2 * math.pi / scale))
    return sin, cos



"""
Images are stored as packed uint32 in the look-up table.
Since each pixel is an uint8, we can have 4 images embedded per int32 'pixel'.

The values are packed as:
bits:  |31 - 23 | 22 - 16 |  15 - 8 | 7 - 0 |
image: | img1   |  img2   |   im3   |  img4 |

We have 3 "channels" of int32 for a total of 12 images per key in the lookup table.
The input piepline reads a vector of keys from the lookup table giving us an ndarray
[... T, H, W, 3] - where `T` is the number of time steps in the vector (it can be 1).

When we upack the values, since we use vectorized operations, the order will be changed.
The flow is as flows;

We use the revers bit packing operations to extract the channels (cn)
to unpacked groups (up_imgn)

    |up_img1|up_img2|up_img3|up_img4|
c0: [img01  | img02 | img03 | img04]
c1: [img05  | img06 | img07 | img08]
c2: [img09  | img10 | img11 | img12]

We can then stack the unpacked images along their channel axis to give us an ndarray of shape
[... T, H, W, 12]. However the images are out of order:

        up_img1      +       up_img2      +       up_img3     +       up_img4
[img01, img05, img09, img02, img06, img10, img03, img07, img11, img04, img08, img12]

Thus, the final step is to reader them using the indices:
[0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
"""

@tf.function
def pack_tf(x: UInt8[tf.Tensor, "... N"], ls=tf.bitwise.left_shift) -> UInt32[tf.Tensor, "... N/4"]:
    """
    pack the last dimensions of a tensors of type uint8,
    into a tensor of shape [..., D/4] with type int32
    This is done by casting by packing the bits.
    Bit      | 31 - 23 | 22 - 16 |  15 - 8 |  7 - 0 |
    Chan(%4) |  c1    |    c2   |   c3    |   c4   |
    """
    x = tf.cast(x, tf.uint32)
    shape = tf.shape(x)
    # assert shape[-1] % 4 == 0, "Can only pack values with a shape (..., 4n). If your values dont conform, pad with 0s"

    num_dims = tf.cast(tf.math.ceil(shape[-1] / 4), tf.int32)
    x = tf.reshape(x, (-1, num_dims, 4)) #*shape[:-1]

    """
    Pack data into the stack tensor.
    We shift stack by 8 bits each time so the previously added value moves up a segment of bits.
    """
    stack = x[..., 0] 
    stack = x[..., 1] + ls(stack, 8)
    stack = x[..., 2] + ls(stack, 8)
    stack = x[..., 3] + ls(stack, 8)
    return stack

def unpack_jax(x: UInt32[Array, "... N"]) -> UInt32[Array, "... 4*N"]:
    shape = jnp.shape(x)
    x = _raw_unpack(x, rs=jnp.right_shift, band=jnp.bitwise_and)
    x = jnp.stack(x, axis=-1)
    x = jnp.reshape(x, (*shape[:-1], -1))
    x = x.astype(jnp.uint8)
    return x
def unpack_tf(x: UInt32[tf.Tensor, "... N"]) -> UInt32[tf.Tensor, "... 4*N"]:
    shape = tf.shape(x)
    x = _raw_unpack(x)
    x = tf.stack(x, axis=-1)
    x = tf.reshape(x, (*shape[:-1], -1))
    x = tf.cast(x, tf.uint8)
    return x

def _raw_unpack(x, rs=tf.bitwise.right_shift, band=tf.bitwise.bitwise_and):
    """
    unpack pack 4 tensors of type int8 from one of type uint32
    The values are assumed to be packed:
    |31 - 23 | 22 - 16 |  15 - 8 | 7 - 0 |
    | up_1   |  up_2   |  up_3   |  up_4 |
    """
    up_4 = band((x), 0xff) # Take the last image, no need to shift
    up_3 = band(rs(x, 8), 0xff)
    up_2 = band(rs(x, 16), 0xff)
    up_1 = rs(x, 24) # No need to mask since right shift all the way over
    return (up_1, up_2, up_3, up_4)


def fmt(s):
    s = bin(s)
    s = s[2:]
    p = "".join((['0'] * 31))
    s = p[: 32 - len(s)] + s
    return f"b {s[0:8]} {s[8:16]} {s[16:24]} {s[24:]}"