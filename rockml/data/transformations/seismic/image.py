""" Copyright 2019 IBM Research. All Rights Reserved.

    - Transformation classes for seismic seisfast.
"""

from typing import List, Tuple, Callable

import numpy as np
from PIL import Image
from rockml.data import array_ops
from rockml.data.adapter import Datum
from rockml.data.adapter.seismic.segy.poststack import Direction, PostStackDatum
from rockml.data.transformations import Transformation
from skimage.exposure import equalize_hist


def _check_dimension(array: np.ndarray, dim: int) -> bool:
    """ Verify if *array* dimensionality is the same as *dim*.

    Args:
        array: ndarray.
        dim: (int) representing a dimensionality.

    Returns:
        True if *dim* is equal to *array* dimensionality
    """

    if type(array) != np.ndarray:
        return False

    if array.ndim != dim:
        return False

    return True


class Crop2D(Transformation):
    def __init__(self, crop_left: int, crop_right: int, crop_top: int, crop_bottom: int,
                 ignore_label=False):
        """ Initialize Crop2D.

        Args:
            crop_top: (int) number of pixels to crop on top.
            crop_bottom: (int) number of pixels to crop on bottom.
            crop_left: (int) number of pixels to crop on left.
            crop_right: (int) number of pixels to crop on right.
            ignore_label: (bool) if True, the labels in SeismicDatums are not cropped, otherwise the labels are
                cropped exactly as the features.
        """
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.ignore_label = ignore_label

    def __call__(self, datum: PostStackDatum) -> PostStackDatum:
        """ Crop inplace *datum*.features using the values *self.crop_top*, *self.crop_bottom*,
            *self.crop_left* and *self.crop_right*. If *self.ignore_label* is False, the crop
            operation will be performed on the label as well. Note that if *self.ignore_label*
            is False and label.ndim != 2, an exception will be raised.

        Args:
            datum: PostStackDatum containing a 2D ndarray as features and (possibly) a 2D array
                as label.

        Returns:
            PostStackDatum with cropped features and (possibly) label.
        """

        if not _check_dimension(datum.features, dim=3):
            raise ValueError(f'datum.features does not have ndim = 3.')

        if not self.ignore_label:
            if not _check_dimension(datum.label, dim=2):
                raise ValueError(f'datum.label does not have ndim = 2.')
            else:
                datum.label = array_ops.crop_2d(datum.label,
                                                crop_top=self.crop_top,
                                                crop_bottom=self.crop_bottom,
                                                crop_left=self.crop_left,
                                                crop_right=self.crop_right)

        datum.features = array_ops.crop_2d(datum.features,
                                           crop_top=self.crop_top,
                                           crop_bottom=self.crop_bottom,
                                           crop_left=self.crop_left,
                                           crop_right=self.crop_right)
        datum.column += self.crop_left
        datum.pixel_depth += self.crop_top

        return datum

    def __str__(self):
        return f"<Crop2D LRTB: {self.crop_left, self.crop_right, self.crop_top, self.crop_bottom}>"


class ScaleIntensity(Transformation):
    def __init__(self, gray_levels: int, percentile: float):
        """ Initialize ScaleIntensity.

        Args:
            gray_levels: (int) number of levels of quantization.
            percentile: (float) percentile of outliers to be cut (from 0.0 to 100.0).
        """

        self.gray_levels = gray_levels
        self.percentile = percentile

    def __call__(self, datum: PostStackDatum) -> PostStackDatum:
        """ Rescale inplace *datum*.features intensity to specified number of gray levels per channel. Before
            rescaling, remove *self.percentile* outliers. Final result is in uint8 interval
            (0 - 255) with *self.gray_levels* different levels.

        Args:
            datum: PostStackDatum containing a ndarray as features.

        Returns:
            PostStackDatum with rescaled features.
        """

        channels = datum.features.shape[-1]
        scaled_features = np.zeros(shape=datum.features.shape, dtype=np.uint8)
        for c in range(channels):
            scaled_features[..., c] = array_ops.scale_intensity(datum.features[..., c],
                                                                gray_levels=self.gray_levels,
                                                                percentile=self.percentile)
        datum.features = scaled_features

        return datum

    def __str__(self):
        return f"<ScaleIntensity {self.gray_levels} levels, {self.percentile} % >"


class FillSegmentationMask(Transformation):

    def __call__(self, datum: PostStackDatum) -> PostStackDatum:
        """ Modify *datum*.label inplace so that the pixels between 2 non-zero horizons are filled with the upper
            boundary value.

        Args:
            datum: PostStackDatum containing a 2D ndarray as label.

        Returns:
            PostStackDatum with label modified to be a segmentation mask.
        """

        if not _check_dimension(datum.label, dim=2):
            raise ValueError(f'datum.label does not have ndim = 2.')

        array_ops.fill_segmentation_mask(horizon_mask=datum.label)

        return datum

    def __str__(self):
        return f"<FillSegmentation>"


class BinarizeMask(Transformation):

    def __call__(self, datum: PostStackDatum) -> PostStackDatum:
        """ Transform inplace all non-zero values in *datum*.label into 1, so that it contains only ones and zeros.

        Args:
            datum: PostStackDatum containing a 2D ndarray as label.

        Returns:
            PostStackDatum with binary label values.
        """

        if not _check_dimension(datum.label, dim=2):
            raise ValueError(f'datum.label does not have ndim = 2.')

        array_ops.binarize_array(horizon_mask=datum.label)

        return datum

    def __str__(self):
        return f"<BinarizeMask>"


class ThickenLinesMask(Transformation):
    def __init__(self, n_points: int):
        """ Initialize ThickenLinesMask.

        Args:
            n_points: (int) number of points to expand the lines in each direction (top and bottom).
        """

        self.n_points = n_points

    def __call__(self, datum: PostStackDatum) -> PostStackDatum:
        """ Thicken non-zero lines inplace in *datum*.label by *self.n_points* above and *self.n_points* below.

        Returns:
            PostStackDatum with thickened label lines.
        """

        if not _check_dimension(datum.label, dim=2):
            raise ValueError(f'datum.label does not have ndim = 2.')

        array_ops.thicken_lines(horizon_mask=datum.label, n_points=self.n_points)

        return datum

    def __str__(self):
        return f"<ThickenLinesMask n_points: {self.n_points}>"


class ViewAsWindows(Transformation):
    def __init__(self, tile_shape: Tuple[int, int],
                 stride_shape: Tuple[int, int],
                 auto_pad: bool = False,
                 filters: list = None):
        """ Initialize ViewAsWindows.

        Args:
            tile_shape: (int, int) height and width of the tiles.
            stride_shape: (int, int) steps the window will skip in the height
                and width directions.
            auto_pad: (bool) whether the output will include all pixels in the image.
                This will add some padding in the image.
            filters: list of initialized filter objects.
        """

        self.tile_shape = tile_shape
        self.stride_shape = stride_shape
        self.auto_pad = auto_pad
        self.filters = filters if filters is not None else []

    def __str__(self):
        return f"ViewAsWindows[{','.join(map(str, self.filters))}]"

    def __call__(self, datum: PostStackDatum) -> List[PostStackDatum]:
        """ Break *datum* features and labels into tiles using a sliding window function. The size of the tiles are
            determined by *self.tile_height* and *self.tile_width* and the stride for the sliding window are defined by
            *self.stride_height* and *self.stride_width*. The resulting list of tiles will be filtered according to the
            list *self.filters*. Finally, a list of SeismicDatums is created, where each datum corresponds to a tile
            features and label. Also, a channel axis is added to the features tiles, as some deep learning libraries
            require input images with a channel axis.

        Args:
            datum: PostStackDatum containing a 2D ndarray as features and a 2D ndarray as label.

        Returns:
            list of PostStackDatum.
        """

        features = datum.features
        label = datum.label
        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=-1)

        ft_tile_shape = self.tile_shape + features.shape[-1:]
        ft_stride_shape = self.stride_shape + features.shape[-1:]
        lb_tile_shape = self.tile_shape + label.shape[-1:]
        lb_stride_shape = self.stride_shape + label.shape[-1:]

        if self.auto_pad:
            features, pad = array_ops.exact_pad(
                features,
                window_shape=ft_tile_shape,
                stride=ft_stride_shape
            )
            if ft_tile_shape[:-1] == lb_tile_shape[:-1]:
                label, pad = array_ops.exact_pad(
                    label,
                    window_shape=lb_tile_shape,
                    stride=lb_stride_shape
                )

        feat_tiles = array_ops.view_as_windows(
            features,
            window_shape=ft_tile_shape,
            stride=ft_stride_shape
        )

        label_tiles = array_ops.view_as_windows(
            label,
            window_shape=lb_tile_shape,
            stride=lb_stride_shape
        )

        indexes = array_ops.get_tiles_indexes(
            label_tiles.shape,
            tuple(ft_stride_shape)
        )

        feat_tiles = feat_tiles.reshape((-1,) + ft_tile_shape)
        label_tiles = label_tiles.reshape((-1,) + lb_tile_shape)
        if len(datum.label.shape) == 2:
            label_tiles = label_tiles.squeeze()
        datum_list = [None] * len(feat_tiles)
        last_idx = 0
        for tile, label, index in zip(feat_tiles, label_tiles, indexes):
            tile_datum = PostStackDatum(
                tile, label,
                direction=datum.direction,
                line_number=datum.line_number,
                pixel_depth=index[0] + datum.pixel_depth,
                column=index[1] + datum.column
            )

            for f in self.filters:
                tile_datum = f(tile_datum)
                if tile_datum is None:
                    break

            if tile_datum is None:
                continue

            datum_list[last_idx] = tile_datum
            last_idx += 1

        datum_list = datum_list[:last_idx]

        return datum_list


class ReconstructFromWindows(Transformation):
    def __init__(self,
                 inline_shape: Tuple[int, int],
                 crossline_shape: Tuple[int, int],
                 strides: Tuple[int, int],
                 overlapping_fn: Callable = None):
        """ Initialize ReconstructFromWindows.

        Args:
            inline_shape: tuple with the shape of an inline slice, considering crop if any.
            crossline_shape: tuple with the shape of an inline slice, considering crop if any.
            strides: tuple with the strides used to generate the tiles in the order: (height, width).
            overlapping_fn: function that defines how overlapping pixels should be treated.
        """
        self.inline_shape = inline_shape
        self.crossline_shape = crossline_shape
        self.strides = strides
        self.overlapping_fn = overlapping_fn

    def __call__(self, data: List[PostStackDatum]):
        """ Reconstruct the images from the list of PostStackDatum tiles. The tiles are separated by line number, and
            the images are reconstructed one at a time. If *self.strides* are smaller the tile size, pixels will be
            overlapped in the reconstruction. In this case, the tiles are overwritten by the ones more to the left and
            bottom of the image). However, *self.overlapping_fn* can be used to set a weighting function to the pixels,
            so they can be summed instead of overwritten.

        Args:
            data: list of PostStackDatum.

        Returns:
            list of PostStackDatum containing the reconstructed lines.
        """
        # Group tiles by line
        data = sorted(data, key=lambda e: (e.line_number, e.pixel_depth, e.column))
        lines = [e.line_number for e in data]

        final_lines = []
        line_tiles = []
        prev_line = lines[0]
        for idx, line in enumerate(lines):
            if prev_line == line:
                line_tiles.append(data[idx])
            else:
                final_lines.append(self._reconstruct_line(line_tiles))
                line_tiles = [data[idx]]
            prev_line = line
        final_lines.append(self._reconstruct_line(line_tiles))

        return final_lines

    def _reconstruct_line(self, data: List[PostStackDatum]):
        """ Reconstruct a single seismic line from the list of PostStackDatum tiles (assumed to be ordered by depth and
            column. The PostStackDatum.label is also reconstructed. The function considers the *self.overlapping_fn*,
            and the padding that was applied to the image before breaking it into tiles (it is assumed that
            ViewAsWindows was used with auto_pad=True).

        Args:
            data: list of PostStackDatum, ordered by depth and column.

        Returns:
            a single PostStackDatum, with features and label reconstructed from the list.
        """
        if data[0].direction == Direction.INLINE:
            shape = self.inline_shape + (1,)
        elif data[0].direction == Direction.CROSSLINE:
            shape = self.crossline_shape + (1,)
        else:
            raise AttributeError('This transformation only accepts INLINE or CROSSLINE directions.')
        strides = self.strides + (1,)

        # Extract features and labels from each datum
        features = [None] * len(data)
        labels = [None] * len(data)
        for idx, datum in enumerate(data):
            assert data[0].direction == datum.direction
            assert data[0].line_number == datum.line_number
            assert len(datum.features.shape) == 3
            assert len(datum.label.shape) == 2 or len(datum.label.shape) == 3
            features[idx] = datum.features
            labels[idx] = datum.label

        features = np.stack(features)
        labels = np.stack(labels)
        if len(labels.shape) == 3:
            labels = np.expand_dims(labels, axis=-1)

        # Calculate the pad that was applied when breaking the image into tiles to result in a exact division.
        pad = array_ops.exact_pad(
            np.empty(shape),
            window_shape=data[0].features.shape,
            stride=strides
        )[1]

        # Reconstruct tiles considering padding
        features = array_ops.reconstruct_from_windows(
            tile_array=features,
            final_shape=(shape[0] + pad[0], shape[1] + pad[1], features.shape[-1]),
            stride=strides
        )

        labels = array_ops.reconstruct_from_windows(
            tile_array=labels,
            final_shape=(shape[0] + pad[0], shape[1] + pad[1], labels.shape[-1]),
            stride=strides,
            overlapping_fn=self.overlapping_fn
        )

        # Crop features and labels, to remove padding
        features = array_ops.crop_2d(features, 0, pad[0], 0, pad[1])
        labels = array_ops.crop_2d(labels, 0, pad[0], 0, pad[1])
        if len(data[0].label.shape) == 2:
            labels = labels.squeeze()

        return PostStackDatum(
            features=features,
            label=labels,
            direction=data[0].direction,
            line_number=data[0].line_number,
            pixel_depth=data[0].pixel_depth,
            column=data[0].column
        )


class Resize2D(Transformation):
    def __init__(self, height: int, width: int, mode: str, ignore_label: bool = False):
        """ Initialize Resize2D.

        Args:
            height: (int) final height.
            width: (int) final width.
            mode: (str) one of 'linear' or 'spline'.
            ignore_label: (bool) if True, the labels in SeismicDatums are not resized, otherwise the labels are
                resized exactly as the features.
        """

        self.height = height
        self.width = width
        self.mode = mode
        self.ignore_label = ignore_label

    def __call__(self, datum: PostStackDatum) -> PostStackDatum:
        """ Resize inplace *datum*.features using the values *self.height*, *self.width* and
            *self.mode*. If *self.ignore_label* is False, the resize
            operation will be performed on the label as well. Note that if *self.ignore_label*
            is False and label.ndim != 2, an exception will be raised.

        Args:
            datum: PostStackDatum containing a 2D ndarray as features and (possibly) a 2D array
                as label.

        Returns:
            PostStackDatum with cropped features and (possibly) label.
        """

        if not _check_dimension(datum.features, dim=3):
            raise ValueError(f'datum.features does not have ndim = 3.')

        if not self.ignore_label:
            if not _check_dimension(datum.label, dim=2):
                raise ValueError(f'datum.label does not have ndim = 2.')
            else:
                datum.label = datum.label[:, :, np.newaxis]
                datum.label = array_ops.resize_2d(datum.label,
                                                  height=self.height,
                                                  width=self.width,
                                                  mode=self.mode)
                datum.label = datum.label[:, :]

        datum.features = array_ops.resize_2d(datum.features,
                                             height=self.height,
                                             width=self.width,
                                             mode=self.mode)
        return datum

    def __str__(self):
        return f"<Resize2D mode {self.mode}, H x W: {self.height}, {self.width}>"


class Interpolate2D(Transformation):
    def __init__(self, height_amp_factor: int, width_amp_factor: int, mode: str, ignore_label: bool = False):
        """ Initialize Interpolate2D.

        Args:
            height_amp_factor: (int) final height.
            width_amp_factor: (int) final width.
            mode: (str) one of 'linear' or 'spline'.
            ignore_label: (bool) if True, the labels in SeismicDatums are not interpolated, otherwise the labels are
                interpolated exactly as the features.
        """

        self.height_amp_factor = height_amp_factor
        self.width_amp_factor = width_amp_factor
        self.mode = mode
        self.ignore_label = ignore_label

    def __call__(self, datum: PostStackDatum) -> PostStackDatum:
        """ Interpolate inplace *datum*.features using the values *self.height_amp_factor*, *self.width_amp_factor* and
            *self.mode*. If *self.ignore_label* is False, the interpolate
            operation will be performed on the label as well. Note that if *self.ignore_label*
            is False and label.ndim != 2, an exception will be raised.

        Args:
            datum: PostStackDatum containing a 2D ndarray as features and (possibly) a 2D array
                as label.

        Returns:
            PostStackDatum with interpolated features and (possibly) label.
        """

        if not _check_dimension(datum.features, dim=3):
            raise ValueError(f'datum.features does not have ndim = 3.')

        if not self.ignore_label:
            if not _check_dimension(datum.label, dim=2):
                raise ValueError(f'datum.label does not have ndim = 2.')
            else:
                datum.label = datum.label[:, :, np.newaxis]
                datum.label = array_ops.interpolate_image_2d(datum.label,
                                                             height_amp_factor=self.height_amp_factor,
                                                             width_amp_factor=self.width_amp_factor,
                                                             mode=self.mode)
                datum.label = datum.label[:, :]

        datum.features = array_ops.interpolate_image_2d(datum.features,
                                                        height_amp_factor=self.height_amp_factor,
                                                        width_amp_factor=self.width_amp_factor,
                                                        mode=self.mode)
        return datum

    def __str__(self):
        return f"<Interpolate mode {self.mode}, H x W: {self.height_amp_factor}, {self.width_amp_factor}>"


class Resize(Transformation):
    def __init__(self, height: int, width: int, mode: str, ignore_label: bool = False):
        """ Initialize Resize.

        Args:
            height: (int) final height.
            width: (int) final width.
            mode: (str) one of 'linear' or 'spline'.
            ignore_label: (bool) if True, the labels in SeismicDatums are not resized, otherwise the labels are
                resized exactly as the features.
        """

        self.height = height
        self.width = width
        self.mode = mode
        self.ignore_label = ignore_label

    def __call__(self, datum: Datum) -> Datum:
        """ Resize inplace *datum*.features using the values *self.height*, *self.width* and *self.mode*. If
            *self.ignore_label* is False, the resize operation will be performed on the label as well. Note that
            if *self.ignore_label* is False and label.ndim != 2, an exception will be raised.

        Args:
            datum: PostStackDatum containing a 2D ndarray as features and (possibly) a 2D array as label.

        Returns:
            PostStackDatum with cropped features and (possibly) label.
        """

        if datum.features.ndim not in [2, 3]:
            raise ValueError(f'datum.features does not have ndim = 2 or 3.')

        if not self.ignore_label:
            if datum.label.ndim not in [2, 3]:
                raise ValueError(f'datum.label does not have ndim = 2 or 3.')
            else:
                datum.label = self.resize(datum.features)

        datum.features = self.resize(datum.features)

        return datum

    def __str__(self):
        return f"<Resize mode: {self.mode}, H x W: {self.height}, {self.width}>"

    def resize(self, image: np.ndarray) -> np.ndarray:
        """ Resize *image* with final size defined by *self.height* and *self.width*. The resample *self.mode* can be
            either 'linear' or cubic 'spline' (corresponding to the PIL modes Image.BICUBIC, and Image.LINEAR,
            respectively).

        Args:
            image: np.ndarray representing an image; shape = (H, W, C).

        Returns:
            np.ndarray of shape = (*height*, *width*, channels).
        """

        if self.mode == 'spline':
            mode = Image.BICUBIC
        elif self.mode == 'linear':
            mode = Image.LINEAR
        else:
            raise ValueError(f"Invalid mode {self.mode}. Should be either 'linear' or 'spline'.")

        new_image = Image.fromarray(image)
        new_image = new_image.resize(size=(self.width, self.height), resample=mode)

        new_image = np.array(new_image)

        return new_image


class ScaleByteFeatures(Transformation):

    def __call__(self, datum: Datum):
        """  Rescale Datum features inplace (features = features / 255) and cast to type np.float32.

        Args:
            datum: a Datum containing "features" attribute.

        Returns:
            modified datum
        """

        datum.features = (datum.features / 255.).astype(np.float32)

        return datum


class PostStackLabelArgMax(Transformation):
    # TODO: add test here
    def __call__(self, dataset):
        dataset.label = np.argmax(dataset.label, axis=-1)

        return dataset


class EqualizeHistogram(Transformation):
    def __init__(self, n_bins: int):
        """ Initialize EqualizeHistogram.

        Args:
            n_bins: (int) number of bins to use in histogram equalization.
        """
        self.n_bins = n_bins

    def __call__(self, datum: Datum) -> Datum:
        """ Performs histogram equalization on datum.features (inplace)."""

        datum.features = equalize_hist(datum.features, nbins=self.n_bins)

        return datum

    def __str__(self):
        return f"<EqualizeHistogram n_bins: {self.n_bins}>"
