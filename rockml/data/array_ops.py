""" Copyright 2019 IBM Research. All Rights Reserved.

    - Functions to modify seisfast that operates on numpy arrays.
"""

import warnings
from typing import Callable, Tuple, Union

import numpy as np
from PIL import Image
from scipy.interpolate import interpn
from scipy.signal.windows import gaussian
from skimage import util, exposure as exp


def crop_2d(image: np.ndarray, crop_top: int, crop_bottom: int, crop_left: int,
            crop_right: int) -> np.ndarray:
    """ Crop *image* using the values *crop_top*, *crop_bottom*, *crop_left* and *crop_right*.

    Args:
        image: 3D ndarray (channel axis on last dimension) or 2D array without channel axis.
        crop_top: (int) number of pixels to crop on top of *image*.
        crop_bottom: (int) number of pixels to crop on bottom of *image*.
        crop_left: (int) number of pixels to crop on left of *image*.
        crop_right: (int) number of pixels to crop on right of *image*.

    Returns:
        ndarray with cropped image.
    """
    return image[crop_top:image.shape[0] - crop_bottom, crop_left:image.shape[1] - crop_right, ...]


def crop_1d(array: np.ndarray, crop_top: int, crop_bottom: int) -> np.ndarray:
    """ Crop *image* using the values *crop_top*, *crop_bottom*, *crop_left* and *crop_right*.

    Args:
        array: 1D ndarray.
        crop_top: (int) number of points to crop on top of *image*.
        crop_bottom: (int) number of points to crop on bottom of *image*.

    Returns:
        ndarray with cropped image.
    """
    return array[crop_top:array.shape[0] - crop_bottom]


def scale_intensity(image: np.ndarray, gray_levels: int, percentile: float) -> np.ndarray:
    """ Rescale *image* intensity to specified number of gray levels. Before rescaling,
        remove percentile outliers. Final result is in uint8 interval (0 - 255) with
        *gray_levels* different levels.

    Args:
        image: numpy array of floats.
        gray_levels: (int) number of levels of quantization.
        percentile: (float) percentile of outliers to be cut (from 0.0 to 100.0)

    Returns:
        image in uint8 format, quantized to *gray_levels* levels.
    """

    pmin, pmax = np.percentile(image, (percentile, 100.0 - percentile))
    im = exp.rescale_intensity(
        image, in_range=(pmin, pmax), out_range=(0, gray_levels - 1)
    ).astype(np.uint8)

    out = exp.rescale_intensity(
        im, in_range=(0, gray_levels - 1), out_range=(0, 255)
    ).astype(np.uint8)

    return out


def view_as_windows(array: np.ndarray, window_shape: Union[Tuple[int, ...], int],
                    stride: Union[Tuple[int, ...], int]) -> np.ndarray:
    """ Sliding window function to create tiles of size *window_shape* from *array*.

    Args:
        array: ndarray to be broken into tiles.
        window_shape: tuple or int containing the size of the sliding window.
        stride: tuple or int containing the steps the window will skip in each direction.

    Returns:
        ndarray of tiles with shape (tiles_per_dim0, ..., tiles_per_dimN, window_shape).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tiles = util.view_as_windows(array, window_shape, stride)

    # if type(window_shape) == int:
    #     window_shape = (window_shape,) * array.ndim

    # tiles = tiles.reshape((-1,) + window_shape)

    return tiles


def reconstruct_from_windows(tile_array: np.ndarray,
                             final_shape: Tuple[int, ...],
                             stride: Tuple[int, ...],
                             overlapping_fn: Callable = None) -> np.ndarray:
    """ Reconstruct an 2D array from the tiles in *tile_array* skipping *stride* in each dimension. Assumes that
        *tile_array* comes exactly from an image with shape *final_shape*. In other words, when breaking this image into
        the *tile_array* no pixels should be left over.

    Args:
        tile_array: ndarray containing the tiles.
        final_shape: original array shape.
        stride: tuple containing the number of points to skip in each dimension; if smaller than the tile size, there
            will be overlapping pixels.
        overlapping_fn: a function that defines how overlapping pixels should be weighted and summed. If None, the
            pixels are simply overwritten by the next tile.

    Returns:
        reconstructed array.
    """

    tile_shape = tile_array.shape[1:]
    tiles_per_dim, _ = _get_num_tiles(shape=final_shape, window_shape=tile_shape, stride=stride)

    if overlapping_fn:
        final_array = np.zeros(final_shape)
    else:
        final_array = np.zeros(final_shape, dtype=tile_array.dtype)

    idx = 0
    rows = range(0, tiles_per_dim[0] * stride[0], stride[0])
    for row in rows:
        columns = range(0, tiles_per_dim[1] * stride[1], stride[1])
        for col in columns:
            if overlapping_fn:
                weighted_tile = overlapping_fn(tile_array[idx])
                final_array[row:row + tile_shape[0], col:col + tile_shape[1]] += weighted_tile
            else:
                final_array[row:row + tile_shape[0], col:col + tile_shape[1]] = tile_array[idx]
            idx += 1

    return final_array


def get_tiles_indexes(shape: Tuple[int, ...], stride: Tuple[int, ...]) -> np.ndarray:
    """ In a array containing the tiles in each dimension (i.e. shape *shape*) calculate the tiles positions in the
        original array.

    Args:
        shape: tuple in the form (tiles_per_dim0, tiles_per_dim1, window_shape).
        stride: tuple containing the steps the window will skip in each direction.

    Returns:
        np.ndarray representing the upper left position of tiles relative to the original array.
    """

    coords = np.meshgrid(*map(np.arange, shape[0:-len(stride)]), indexing='ij')
    coords = [(coord * s_i).reshape(-1) for coord, s_i in zip(coords, stride)]
    indexes = np.stack(coords, axis=1)

    return indexes


def fill_segmentation_mask(horizon_mask: np.ndarray):
    """ Fill the *horizon_mask* ndarray inplace with horizon boundaries values.

    Example:

        horizon_mask before:

            array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 2., 2., 2., 2., 2.],
                   [2., 2., 2., 2., 2., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

        horizon_mask after:

            array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                   [1., 1., 1., 1., 1., 2., 2., 2., 2., 2.],
                   [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                   [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]])


    Args:
        horizon_mask: 2D ndarray with zeros and horizon lines numbers.
    """

    line = np.zeros(shape=horizon_mask.shape[1], dtype=np.uint8)

    for idx in range(horizon_mask.shape[0]):
        line = np.where(horizon_mask[idx, :] != 0, horizon_mask[idx, :], line)
        horizon_mask[idx, :] = np.where(line != 0, line, horizon_mask[idx, :])


def binarize_array(horizon_mask: np.ndarray):
    """ Transform inplace all non-zero values in *horizon_mask* into 1, so that the mask will have only ones and zeros.

    Args:
        horizon_mask: 2D ndarray with zeros and horizon lines numbers.
    """

    horizon_mask[np.where(horizon_mask != 0)] = 1


def thicken_lines(horizon_mask: np.ndarray, n_points: int):
    """ Thicken non-zero lines inplace in *horizon_mask* by *n_points* above and *n_points* below.

    Args:
        horizon_mask: 2D ndarray with zeros and horizon lines numbers.
        n_points: (int) number of points to expand the lines in each direction (top and bottom).
    """

    for col in range(horizon_mask.shape[1]):
        idx = np.where(horizon_mask[:, col] != 0)[0]
        for i in idx:
            bottom = max(i - n_points, 0)
            top = min(i + n_points + 1, horizon_mask.shape[0])

            horizon_mask[bottom:top, col] = horizon_mask[i, col]


def _calculate_pad(strides: Tuple[int, ...], remain_points: Tuple[int, ...]) -> Tuple[int, ...]:
    """ Calculate the necessary pad so that array can completely fit a integer number of tiles in all dimensions.

    Args:
        strides: tuple of step sizes of the sliding window.
        remain_points: tuple containing the number of remaining points to the end of the array
            in each dimension.

    Returns:
        (tuple) pad values
    """

    assert len(strides) == len(remain_points), \
        "*strides* and *remain_points* must have the same size!"

    return tuple((strides[i] - remain_points[i]) if remain_points[i] > 0 else 0 for i in range(len(strides)))


def _get_num_tiles(shape: Tuple[int, ...], window_shape: Tuple[int, ...],
                   stride: Tuple[int, ...]) -> (Tuple[int, ...], Tuple[int, ...]):
    """ Gets the number of tiles that fits in *length* points and the remaining points to the
        end of the *length* value.

    Args:
        shape: original array shape.
        window_shape: tuple containing the shape of the sliding window.
        stride: tuple containing the step sizes for the sliding window.

    Returns:
        list containing the number of tiles.
        list containing the number of remaining points to the end of the length value.
    """

    def _num_tiles(length: int, window_size: int, stride: int) -> Tuple[int, int]:
        """ Gets the number of tiles that fits in *length* points and the remaining points to the
            end of the *length* value.

        Args:
            length: original array size.
            window_size: (int) size of the window.
            stride: (int) step size of the sliding window.

        Returns:
            (int32) number of tiles.
            (int32) number of remaining points to the end of the length value.
        """

        total_tiles = ((length - window_size) // stride) + 1
        remainder_points = (length - window_size) % stride

        return total_tiles, remainder_points

    assert len(stride) == len(window_shape) == len(shape), \
        "*strides*, and *window_shape* and *shape* must have the same size!"

    num_tiles = [0] * len(stride)
    remain_points = [0] * len(stride)

    for i in range(len(num_tiles)):
        num_tiles[i], remain_points[i] = _num_tiles(shape[i], window_shape[i], stride[i])

    return tuple(num_tiles), tuple(remain_points)


def exact_pad(image: np.ndarray, window_shape: Union[Tuple[int, ...], int], stride: Union[Tuple[int, ...], int],
              mode: str = 'symmetric') -> (np.ndarray, Tuple[int, ...]):
    """ Pad *image* in order to get an exact number of tiles in view_as_windows(), considering the window size
        *window_shape* and *stride*. Notice that this function pads *image* at the end of each axis only.

    Args:
        image: ndarray with image to be padded.
        window_shape: tuple or int containing the size of the sliding window.
        stride: tuple or int containing the steps the window will skip in each direction.
        mode: (str) The user can choose the one of the numpy methods for padding:

            'constant' -> Pads with a constant value.

            'edge' -> Pads with the edge values of array.

            'linear_ramp' -> Pads with the linear ramp between end_value and the array edge value.

            'maximum' -> Pads with the maximum value of all or part of the vector along each axis.

            'mean' -> Pads with the mean value of all or part of the vector along each axis.

            'median' -> Pads with the median value of all or part of the vector along each axis.

            'minimum' -> Pads with the minimum value of all or part of the vector along each axis.

            'reflect' -> Pads with the reflection of the vector mirrored on the first and last values of the vector

            'symmetric' -> Pads with the reflection of the vector mirrored along the edge of the array.

            'wrap' -> Pads with the wrap of the vector along the axis. The first values are used to pad the end and the
                      end values are used to pad the beginning.

    Returns:
        tuple containing the padded image and the calculated pad in each direction.
    """

    if type(window_shape) == int:
        window_shape = tuple(window_shape for _ in range(image.ndim))

    if type(stride) == int:
        stride = tuple(stride for _ in range(image.ndim))

    _, remain = _get_num_tiles(shape=image.shape, window_shape=window_shape, stride=stride)

    pad = _calculate_pad(strides=stride, remain_points=remain)

    image = np.pad(image, [(0, i) for i in pad], mode)

    return image, pad


def resize_2d(image: np.ndarray, height: int, width: int, mode: str) -> np.ndarray:
    """ Resize *image* with final size defined by *height* and *width*. The resample *mode* can be either 'linear' or
        cubic 'spline'. In this method, each channel is resized independently using PIL functions.

    Args:
        image: np.ndarray representing an image; shape = (H, W, C).
        height: (int) final height of the image.
        width: (int) final width of the image.
        mode: (str) one of 'linear' or 'spline'.

    Returns:
        np.ndarray of shape = (*height*, *width*, channels).
    """

    if mode == 'spline':
        resample = Image.BICUBIC
    elif mode == 'linear':
        resample = Image.BILINEAR
    else:
        raise ValueError(f"Invalid mode {mode}. Should be either 'linear' or 'spline'.")

    new_image = [None] * image.shape[-1]

    # Resize each channel independently
    for channel in range(image.shape[-1]):
        img_temp = Image.fromarray(image[:, :, channel].astype(np.float32))
        img_temp = img_temp.resize(size=(width, height), resample=resample)

        new_image[channel] = np.array(img_temp)

    if len(new_image) > 1:
        new_image = np.stack(new_image, axis=-1)
    else:
        new_image = new_image[0][:, :, np.newaxis]

    return new_image


def interpolate_image_2d(image: np.ndarray, height_amp_factor: int, width_amp_factor: int, mode: str) -> np.ndarray:
    """ Interpolate *image* using linear or spline interpolation (one interpolation per channel).
        Use scipy.interpolate.interpn(), which respect the values of the original points.

        Final shape = (image.shape[0]*height_amp_factor, image.shape[1]*width_amp_factor, image.shape[2])

    Args:
        image: np.ndarray representing an image; shape = (H, W, C).
        height_amp_factor: (int) factor to multiply the original height.
        width_amp_factor: (int) factor to multiply the original width.
        mode: (str) one of 'linear' or 'spline'.

    Returns:
        resampled np.ndarray.
    """

    if mode == 'spline':
        mode = 'splinef2d'
    elif mode != 'linear':
        raise ValueError(f"Invalid mode {mode}. Should be either 'linear' or 'spline'.")

    height = image.shape[1] * height_amp_factor
    width = image.shape[0] * width_amp_factor

    x = np.arange(0, width, width_amp_factor)
    y = np.arange(0, height, height_amp_factor)

    grid_x, grid_y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))

    new_image = [None] * image.shape[-1]

    # Interpolate each channel independently
    for channel in range(image.shape[-1]):
        new_image[channel] = np.array(interpn((y, x), image[:, :, channel], (grid_y, grid_x),
                                              method=mode, fill_value=0, bounds_error=False))

    if len(new_image) > 1:
        new_image = np.stack(new_image, axis=-1)
    else:
        new_image = new_image[0][:, :, np.newaxis]

    return new_image


def scale_minmax(array: np.ndarray, feature_range: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """ Transform features by scaling all the features to a given range. Notice that, if you want to separate the
        scaling by features, you have to call it per slice.

    Args:
        array: ndarray. Array to be scaled.
        feature_range: tuple(int, int). Desired range of transformed seisfast.

    Returns:
        ndarray scaled between the indicated range.
        See: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """

    array_std = (array - array.min()) / (array.max() - array.min())
    return array_std * (feature_range[1] - feature_range[0]) + feature_range[0]


def gauss_weight_map(shape: Tuple[int, int], sigma_shape: Tuple[int, int]) -> np.ndarray:
    """ Creates a rectangular gaussian map of weights.

    Args:
        shape: tuple indicating the shape of the output map.
        sigma_shape: tuple indicating the size of the sigma for each direction.

    Returns:
        2D ndarray representing a map with a gaussian function.
    """

    # Creates 2 1D gaussian functions
    kx = gaussian(shape[0], sigma_shape[0]).reshape(shape[0], 1)
    ky = gaussian(shape[1], sigma_shape[1]).reshape(shape[1], 1)

    # Calculate the outer product to get the 2D gaussian
    weight_map = np.outer(kx, ky)

    # Scale values so that maximum and minimum are 1.0 and 0.0, respectively.
    weight_map = scale_minmax(weight_map)

    return weight_map
