""" Copyright 2019 IBM Research. All Rights Reserved.

    - Test functions for seisfast.array_ops.
"""

import numpy as np
import pytest

import rockml.data.array_ops as ao


@pytest.mark.parametrize(
    "elements_after_crop, crop_settings",
    [
        ([0, 1, 2, 3, 4], {"crop_top": 0, "crop_bottom": 0, "crop_left": 0, "crop_right": 0}),
        ([0, 2, 3, 4], {"crop_top": 1, "crop_bottom": 0, "crop_left": 0, "crop_right": 0}),
        ([0, 1, 3, 4], {"crop_top": 0, "crop_bottom": 2, "crop_left": 0, "crop_right": 0}),
        ([0, 1, 2, 4], {"crop_top": 0, "crop_bottom": 0, "crop_left": 3, "crop_right": 0}),
        ([0, 1, 2, 3], {"crop_top": 0, "crop_bottom": 0, "crop_left": 0, "crop_right": 4}),
        ([0, 3, 4], {"crop_top": 1, "crop_bottom": 2, "crop_left": 0, "crop_right": 0}),
        ([0, 2, 4], {"crop_top": 1, "crop_bottom": 0, "crop_left": 3, "crop_right": 0}),
        ([0, 2, 3], {"crop_top": 1, "crop_bottom": 0, "crop_left": 0, "crop_right": 4}),
        ([0, 1, 4], {"crop_top": 0, "crop_bottom": 2, "crop_left": 3, "crop_right": 0}),
        ([0, 1, 3], {"crop_top": 0, "crop_bottom": 2, "crop_left": 0, "crop_right": 4}),
        ([0, 1, 2], {"crop_top": 0, "crop_bottom": 0, "crop_left": 3, "crop_right": 4}),
        ([0, 4], {"crop_top": 1, "crop_bottom": 2, "crop_left": 3, "crop_right": 0}),
        ([0, 1], {"crop_top": 0, "crop_bottom": 2, "crop_left": 3, "crop_right": 4}),
        ([0], {"crop_top": 1, "crop_bottom": 2, "crop_left": 3, "crop_right": 4}),
        ([], {"crop_top": 5, "crop_bottom": 5, "crop_left": 0, "crop_right": 0}),
        ([], {"crop_top": 5, "crop_bottom": 5, "crop_left": 5, "crop_right": 5}),

        pytest.param([5], {"crop_top": 0, "crop_bottom": 0, "crop_left": 0, "crop_right": 0},
                     marks=pytest.mark.xfail(strict=True)),
        pytest.param([0], {"crop_top": 5, "crop_bottom": 5, "crop_left": 0, "crop_right": 0},
                     marks=pytest.mark.xfail(strict=True))
    ]
)
def test_crop2d(elements_after_crop, crop_settings):
    # Arrange:
    x = [[[3], [3], [3], [1], [1], [1], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [2], [2], [2], [4], [4], [4], [4]],
         [[3], [3], [3], [2], [2], [2], [4], [4], [4], [4]]]
    fake_image = np.asarray(x)

    # Act:
    actual = ao.crop_2d(fake_image, **crop_settings)

    # Assert:
    assert np.all(np.isin(elements_after_crop, actual))


@pytest.mark.parametrize(
    "elements_after_crop, crop_settings",
    [
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], {"crop_top": 0, "crop_bottom": 0}),
        ([0, 1, 2, 3, 4, 5, 6], {"crop_top": 0, "crop_bottom": 3}),
        ([3, 4, 5, 6, 7, 8, 9], {"crop_top": 3, "crop_bottom": 0}),
        ([3, 4, 5, 6], {"crop_top": 3, "crop_bottom": 3}),
        ([], {"crop_top": 5, "crop_bottom": 5}),
        ([], {"crop_top": 10, "crop_bottom": 10}),

        pytest.param([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], {"crop_top": 2, "crop_bottom": 0},
                     marks=pytest.mark.xfail(strict=True)),
        pytest.param([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], {"crop_top": 0, "crop_bottom": 5},
                     marks=pytest.mark.xfail(strict=True))
    ]
)
def test_crop1d(elements_after_crop, crop_settings):
    # Arrange:
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fake_image = np.asarray(x)

    # Act:
    actual = ao.crop_1d(fake_image, **crop_settings)

    # Assert:
    assert np.all(np.isin(elements_after_crop, actual))


@pytest.mark.parametrize(
    "fake_image, gray_level, percentile, expected_scaled_image",
    [
        ([[[100], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
          [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
          [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
          [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
          [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]], 11, 5.0,
         [[[255], [25], [51], [76], [102], [127], [153], [178], [204], [255]],
          [[0], [25], [51], [76], [102], [127], [153], [178], [204], [255]],
          [[0], [25], [51], [76], [102], [127], [153], [178], [204], [255]],
          [[0], [25], [51], [76], [102], [127], [153], [178], [204], [255]],
          [[0], [25], [51], [76], [102], [127], [153], [178], [204], [255]]]),

        ([[100, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 11, 0, [[255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),

        ([[100, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 11, 45.0, [[255, 0, 0, 0, 0, 255, 255, 255, 255, 255],
                                                      [0, 0, 0, 0, 0, 255, 255, 255, 255, 255],
                                                      [0, 0, 0, 0, 0, 255, 255, 255, 255, 255],
                                                      [0, 0, 0, 0, 0, 255, 255, 255, 255, 255],
                                                      [0, 0, 0, 0, 0, 255, 255, 255, 255, 255]]),

        ([[100, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 11, 35.0, [[255, 0, 0, 0, 76, 153, 255, 255, 255, 255],
                                                      [0, 0, 0, 0, 76, 153, 255, 255, 255, 255],
                                                      [0, 0, 0, 0, 76, 153, 255, 255, 255, 255],
                                                      [0, 0, 0, 0, 76, 153, 255, 255, 255, 255],
                                                      [0, 0, 0, 0, 76, 153, 255, 255, 255, 255]]),

        ([[100, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 5, 5.0, [[255, 0, 0, 63, 63, 127, 127, 191, 191, 255],
                                                    [0, 0, 0, 63, 63, 127, 127, 191, 191, 255],
                                                    [0, 0, 0, 63, 63, 127, 127, 191, 191, 255],
                                                    [0, 0, 0, 63, 63, 127, 127, 191, 191, 255],
                                                    [0, 0, 0, 63, 63, 127, 127, 191, 191, 255]]),

        ([[100, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 6, 5.0, [[255, 0, 51, 51, 102, 102, 153, 153, 204, 255],
                                                    [0, 0, 51, 51, 102, 102, 153, 153, 204, 255],
                                                    [0, 0, 51, 51, 102, 102, 153, 153, 204, 255],
                                                    [0, 0, 51, 51, 102, 102, 153, 153, 204, 255],
                                                    [0, 0, 51, 51, 102, 102, 153, 153, 204, 255]])
    ]
)
def test_scale_intensity(fake_image, gray_level, percentile, expected_scaled_image):
    # Arrange:
    fake_image = np.asarray(fake_image)

    # Act:
    actual_scaled_image = ao.scale_intensity(fake_image, gray_level, percentile)

    # Assert:
    assert np.array_equal(actual_scaled_image, expected_scaled_image)


@pytest.mark.parametrize(
    "fake_image, expected_tile_shapes, view_as_windows_settings",
    [
        (np.arange(100).reshape(10, 10, 1), (1, 1, 1, 10, 10, 1), {"window_shape": (10, 10, 1), "stride": 5}),
        (np.arange(100).reshape(10, 10, 1), (1, 1, 1, 7, 8, 1), {"window_shape": (7, 8, 1), "stride": 5}),
        (np.arange(100).reshape(10, 10, 1), (2, 2, 1, 5, 5, 1), {"window_shape": (5, 5, 1), "stride": (5, 5, 1)}),
        (np.arange(100).reshape(10, 10, 1), (2, 2, 1, 2, 2, 1), {"window_shape": (2, 2, 1), "stride": 5}),
        (np.arange(100).reshape(10, 10, 1), (3, 4, 1, 5, 4, 1), {"window_shape": (5, 4, 1), "stride": 2}),
        (np.arange(100).reshape(10, 10, 1), (2, 5, 1, 7, 2, 1), {"window_shape": (7, 2, 1), "stride": 2}),
        (np.arange(100).reshape(10, 10, 1), (1, 1, 1, 2, 3, 1), {"window_shape": (2, 3, 1), "stride": 10}),
        (np.arange(100).reshape(10, 10, 1), (8, 9, 1, 3, 2, 1), {"window_shape": (3, 2, 1), "stride": 1}),
        (np.arange(100).reshape(10, 10, 1), (9, 9, 1, 2, 2, 1), {"window_shape": (2, 2, 1), "stride": 1}),
        (np.arange(100).reshape(10, 10, 1), (10, 10, 1, 1, 1, 1), {"window_shape": (1, 1, 1), "stride": 1}),
        (np.arange(100).reshape(5, 5, 4), (2, 2, 2, 2, 2, 2), {"window_shape": (2, 2, 2), "stride": 2}),
        (np.arange(100).reshape(5, 5, 4), (2, 2, 3, 2, 2, 2), {"window_shape": (2, 2, 2), "stride": (2, 2, 1)}),

        pytest.param(np.arange(100).reshape(5, 5, 4), (2, 2, 3, 2, 2, 2), {"window_shape": (2, 2, 2), "stride": 2},
                     marks=pytest.mark.xfail(strict=True)),
        pytest.param(np.arange(100).reshape(10, 10, 1), (1, 3, 3, 5, 3), {"window_shape": (5, 5), "stride": 5},
                     marks=pytest.mark.xfail(strict=True)),
        pytest.param(np.arange(100).reshape(10, 10, 1), (3, 3, 9, 5, 5, 1), {"window_shape": 5, "stride": (2, 2, 1)},
                     marks=pytest.mark.xfail(strict=True))
    ]
)
def test_view_as_windows_return_shape(fake_image, expected_tile_shapes, view_as_windows_settings):
    # Arrange:
    fake_image = fake_image

    # Act:
    actual = ao.view_as_windows(fake_image, **view_as_windows_settings)

    # Assert:
    assert actual.shape == expected_tile_shapes


@pytest.mark.parametrize(
    "tile_coordinates, elements_of_test_window, view_as_windows_settings",
    [
        ((0, 0), [2], {"window_shape": 5, "stride": 5}),
        ((0, 1), [1], {"window_shape": (5, 5), "stride": 5}),
        ((1, 0), [3], {"window_shape": (5, 5), "stride": 5}),
        ((1, 1), [4], {"window_shape": (5, 5), "stride": 5}),
        ((0, 0), [2], {"window_shape": 2, "stride": (5, 5)}),
        ((0, 1), [1], {"window_shape": (2, 2), "stride": 5}),
        ((1, 0), [3], {"window_shape": (2, 2), "stride": 5}),
        ((1, 1), [4], {"window_shape": (2, 2), "stride": 5}),
        ((0, 0), [2, 3], {"window_shape": (10, 2), "stride": 2}),
        ((0, 1), [2, 3], {"window_shape": (10, 2), "stride": 2}),
        ((0, 2), [1, 2, 3, 4], {"window_shape": (10, 2), "stride": 2}),
        ((0, 3), [1, 4], {"window_shape": (10, 2), "stride": 2}),
        ((0, 4), [1, 4], {"window_shape": (10, 2), "stride": 2}),
        ((0, 0), [1, 2, 3, 4], {"window_shape": (10, 10), "stride": 2}),
        ((2, 2), [1, 2, 3, 4], {"window_shape": (2, 2), "stride": 2}),
        ((0, 0), [1, 2, 3, 4], {"window_shape": (7, 9), "stride": 4}),

        pytest.param((0, 0), [1], {"window_shape": (5, 5), "stride": 5}, marks=pytest.mark.xfail(strict=True)),
        pytest.param((0, 1), [1, 2, 3, 4], {"window_shape": (7, 9), "stride": 4}, marks=pytest.mark.xfail(strict=True))
    ]
)
def test_view_as_windows_returned_tile(tile_coordinates, elements_of_test_window, view_as_windows_settings):
    # Arrange:
    x = [[2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
         [3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
         [3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
         [3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
         [3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
         [3, 3, 3, 3, 3, 4, 4, 4, 4, 4]]

    fake_image = np.asarray(x)

    # Act:
    actual_tiles = ao.view_as_windows(fake_image, **view_as_windows_settings)

    # Assert:
    assert np.all(np.isin(elements_of_test_window, actual_tiles[tile_coordinates]))


@pytest.mark.parametrize(
    "fake_image, expected",
    [
        (

                [[0, 0, 0, 0, 0],
                 [1, 0, 0, 1, 1],
                 [0, 1, 1, 0, 0],
                 [2, 0, 2, 2, 2],
                 [0, 2, 0, 0, 0]],

                [[0, 0, 0, 0, 0],
                 [1, 0, 0, 1, 1],
                 [1, 1, 1, 1, 1],
                 [2, 1, 2, 2, 2],
                 [2, 2, 2, 2, 2]]

        ),

        (

                [[1, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 1],
                 [2, 0, 2, 0, 2],
                 [0, 2, 0, 0, 0]],

                [[1, 0, 0, 1, 0],
                 [1, 0, 0, 1, 0],
                 [1, 1, 1, 1, 1],
                 [2, 1, 2, 1, 2],
                 [2, 2, 2, 1, 2]]

        ),

        (

                [[1, 1, 0, 2, 0],
                 [0, 2, 0, 0, 1],
                 [2, 3, 1, 0, 0],
                 [3, 0, 2, 3, 2],
                 [0, 0, 3, 0, 0]],

                [[1, 1, 0, 2, 0],
                 [1, 2, 0, 2, 1],
                 [2, 3, 1, 2, 1],
                 [3, 3, 2, 3, 2],
                 [3, 3, 3, 3, 2]]

        ),

        pytest.param(

            [[0, 0, 0, 0, 0],
             [1, 0, 0, 1, 1],
             [0, 1, 1, 0, 0],
             [2, 0, 2, 2, 2],
             [0, 2, 0, 0, 0]],

            [[0, 0, 0, 0, 0],
             [1, 0, 0, 1, 1],
             [0, 1, 1, 0, 0],
             [2, 0, 2, 2, 2],
             [0, 2, 0, 0, 0]], marks=pytest.mark.xfail(strict=True))
    ]
)
def test_fill_segmentation_mask(fake_image, expected):
    # Arrange:
    fake_horizon_mask = np.asarray(fake_image)
    expected_segmentation_mask = np.asarray(expected)

    # Act:
    ao.fill_segmentation_mask(fake_horizon_mask)

    # Assert:
    assert np.array_equal(fake_horizon_mask, expected_segmentation_mask)


@pytest.mark.parametrize(
    "fake_image, expected",
    [
        (

                [[[0], [0], [0], [0], [0]],
                 [[1], [0], [0], [1], [1]],
                 [[0], [1], [1], [0], [0]],
                 [[2], [0], [2], [2], [2]],
                 [[0], [2], [0], [0], [0]]],

                [[[0], [0], [0], [0], [0]],
                 [[1], [0], [0], [1], [1]],
                 [[0], [1], [1], [0], [0]],
                 [[1], [0], [1], [1], [1]],
                 [[0], [1], [0], [0], [0]]]

        ),

        (

                [[1, 6, 6, 1, 4],
                 [3, 6, 6, 2, 4],
                 [4, 1, 1, 4, 1],
                 [2, 5, 2, 4, 2],
                 [5, 2, 5, 4, 5]],

                [[1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1]]

        ),

        (

                [[[0], [0], [0], [0], [0]],
                 [[0], [0], [0], [0], [0]],
                 [[0], [0], [0], [0], [0]],
                 [[0], [0], [0], [0], [0]],
                 [[0], [0], [0], [0], [0]]],

                [[[0], [0], [0], [0], [0]],
                 [[0], [0], [0], [0], [0]],
                 [[0], [0], [0], [0], [0]],
                 [[0], [0], [0], [0], [0]],
                 [[0], [0], [0], [0], [0]]]

        ),

        (

                [[1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1]],

                [[1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1]]

        ),

        pytest.param(

            [[0, 0, 0, 0, 0],
             [1, 0, 0, 1, 1],
             [0, 1, 1, 0, 0],
             [2, 0, 2, 2, 2],
             [0, 2, 0, 0, 0]],

            [[0, 0, 0, 0, 0],
             [1, 0, 0, 1, 1],
             [0, 1, 1, 0, 0],
             [2, 0, 2, 2, 2],
             [0, 2, 0, 0, 0]], marks=pytest.mark.xfail(strict=True))
    ]
)
def test_binarize_array(fake_image, expected):
    # Arrange:
    fake_horizon_mask = np.asarray(fake_image)
    expected_binarizes_mask = np.asarray(expected)

    # Act:
    ao.binarize_array(fake_horizon_mask)

    # Assert:
    assert np.array_equal(fake_horizon_mask, expected_binarizes_mask)


@pytest.mark.parametrize(
    "fake_image, thicken_by_n, expected",
    [
        (
                [[[0], [0], [0], [1], [0]],
                 [[1], [0], [1], [0], [0]],
                 [[0], [1], [0], [0], [1]],
                 [[0], [0], [0], [0], [0]],
                 [[0], [0], [2], [0], [0]],
                 [[2], [2], [0], [2], [0]],
                 [[0], [0], [0], [0], [2]]], 1, [[[1], [0], [1], [1], [0]],
                                                 [[1], [1], [1], [1], [1]],
                                                 [[1], [1], [1], [0], [1]],
                                                 [[0], [1], [2], [0], [1]],
                                                 [[2], [2], [2], [2], [0]],
                                                 [[2], [2], [2], [2], [2]],
                                                 [[2], [2], [0], [2], [2]]]
        ),

        (
                [[0, 0, 0, 1, 0],
                 [1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 1],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 2, 0],
                 [0, 2, 2, 0, 0],
                 [0, 0, 0, 0, 0],
                 [2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 2]], 2, [[1, 1, 0, 1, 1],
                                       [1, 1, 1, 1, 1],
                                       [1, 1, 1, 1, 1],
                                       [1, 1, 1, 2, 1],
                                       [0, 2, 2, 2, 1],
                                       [0, 2, 2, 2, 0],
                                       [2, 2, 2, 2, 0],
                                       [2, 2, 2, 2, 0],
                                       [2, 2, 2, 0, 2],
                                       [2, 0, 0, 0, 2],
                                       [2, 0, 0, 0, 2]]
        ),

        (
                [[[0], [0], [0], [0], [0]],
                 [[1], [0], [0], [1], [1]],
                 [[0], [1], [1], [0], [0]],
                 [[2], [0], [2], [2], [2]],
                 [[0], [2], [0], [0], [0]]], 1, [[[1], [0], [0], [1], [1]],
                                                 [[1], [1], [1], [1], [1]],
                                                 [[2], [1], [1], [2], [2]],
                                                 [[2], [2], [1], [2], [2]],
                                                 [[2], [2], [1], [2], [2]]]
        ),

        (
                [[0, 0, 0, 0, 1],
                 [1, 0, 1, 1, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 0, 2, 0],
                 [2, 0, 2, 0, 2],
                 [0, 2, 0, 0, 0]], 3, [[1, 1, 1, 1, 1],
                                       [1, 1, 1, 1, 2],
                                       [1, 1, 1, 1, 2],
                                       [1, 1, 1, 1, 2],
                                       [1, 1, 1, 1, 2],
                                       [1, 1, 1, 1, 2]]
        ),

        pytest.param(
            [[0, 0, 0, 0, 0],
             [1, 0, 0, 1, 1],
             [0, 1, 1, 0, 0],
             [2, 0, 2, 2, 2],
             [0, 2, 0, 0, 0]], 1,

            [[1, 0, 0, 1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [2, 1, 2, 2, 2],
             [2, 2, 2, 2, 2]], marks=pytest.mark.xfail(strict=True))
    ]
)
def test_thicken_lines(fake_image, thicken_by_n, expected):
    # Arrange:
    fake_horizon_mask = np.asarray(fake_image)
    expected_thicker_mask = np.asarray(expected)

    # Act:
    ao.thicken_lines(fake_horizon_mask, thicken_by_n)

    # Assert:
    assert np.array_equal(fake_horizon_mask, expected_thicker_mask)


@pytest.mark.parametrize(
    "tiles, arrange_dim, steps, expected",
    [
        (
                [[[0, 0], [0, 1]], [[0, 0], [1, 1]], [[1, 2], [0, 0]], [[0, 0], [2, 2]]],
                (4, 4), (2, 2),

                [[0, 0, 0, 0],
                 [0, 1, 1, 1],
                 [1, 2, 0, 0],
                 [0, 0, 2, 2]]

        ),

        # (
        #         [[[0, 0], [0, 1]], [[0, 0], [1, 1]], [[1, 2], [0, 0]], [[0, 0], [2, 2]]],
        #         (1, 4), (1, 2),
        #
        #         [[0, 0, 0, 0, 1, 2, 0, 0],
        #          [0, 1, 1, 1, 0, 0, 2, 2]]
        #
        # ),

        (
                [[[0, 0], [0, 1]], [[0, 0], [1, 1]], [[1, 2], [0, 0]], [[0, 0], [2, 2]]],
                (8, 2), (2, 1),

                [[0, 0],
                 [0, 1],
                 [0, 0],
                 [1, 1],
                 [1, 2],
                 [0, 0],
                 [0, 0],
                 [2, 2]]

        ),

        (
                [[[0, 0], [0, 1]], [[0, 0], [1, 1]], [[1, 2], [0, 0]], [[0, 0], [2, 2]]],
                (6, 2), (2, 1),

                [[0, 0],
                 [0, 1],
                 [0, 0],
                 [1, 1],
                 [1, 2],
                 [0, 0]]

        ),

        pytest.param(

            [[[[0], [0]], [[0], [1]]], [[[0], [0]], [[1], [1]]], [[[1], [2]], [[0], [0]]], [[[0], [0]], [[2], [2]]]],
            (8, 2), (2, 1),

            [[[0], [0]],
             [[0], [1]],
             [[0], [0]],
             [[1], [1]],
             [[1], [2]],
             [[0], [0]],
             [[0], [0]],
             [[2], [2]]], marks=pytest.mark.xfail(strict=True)),

        pytest.param(

            [[[0, 0], [0, 1]], [[0, 0], [1, 1]], [[1, 2], [0, 0]], [[0, 0], [2, 2]]],
            (4, 0), (2, 0),

            [[[0], [0]],
             [[0], [1]],
             [[0], [0]],
             [[1], [1]],
             [[1], [2]],
             [[0], [0]],
             [[0], [0]],
             [[2], [2]]], marks=pytest.mark.xfail(strict=True))
    ]
)
def test_reconstruct_from_tiles(tiles, arrange_dim, steps, expected):
    # Arrange:
    fake_tiles = np.asarray(tiles)
    expected_image = np.asarray(expected)

    # Act:
    actual_reconstructed_image = ao.reconstruct_from_windows(
        fake_tiles,
        arrange_dim,
        steps
    )

    # Assert:
    assert np.array_equal(actual_reconstructed_image, expected_image)


@pytest.mark.parametrize(
    "fake_image, window_shape, stride, mode, expected",
    [
        (
                [[[0], [0], [0], [1], [0]],
                 [[1], [0], [1], [0], [0]],
                 [[0], [1], [0], [0], [1]],
                 [[2], [2], [0], [2], [0]],
                 [[0], [0], [0], [0], [2]]], (3, 3, 1), (3, 3, 1), "constant", [[[0], [0], [0], [1], [0], [0]],
                                                                                [[1], [0], [1], [0], [0], [0]],
                                                                                [[0], [1], [0], [0], [1], [0]],
                                                                                [[2], [2], [0], [2], [0], [0]],
                                                                                [[0], [0], [0], [0], [2], [0]],
                                                                                [[0], [0], [0], [0], [0], [0]]]
        ),

        (
                [[[0], [0], [0], [1], [0]],
                 [[1], [0], [1], [0], [0]],
                 [[0], [1], [0], [0], [1]],
                 [[0], [0], [0], [0], [0]],
                 [[0], [0], [2], [0], [0]],
                 [[2], [2], [0], [2], [0]],
                 [[0], [0], [0], [0], [2]]], (7, 5, 1), (3, 5, 1), "edge", [[[0], [0], [0], [1], [0]],
                                                                            [[1], [0], [1], [0], [0]],
                                                                            [[0], [1], [0], [0], [1]],
                                                                            [[0], [0], [0], [0], [0]],
                                                                            [[0], [0], [2], [0], [0]],
                                                                            [[2], [2], [0], [2], [0]],
                                                                            [[0], [0], [0], [0], [2]]]
        ),

        (
                [[2, 2, 2, 2],
                 [2, 2, 2, 2],
                 [2, 2, 2, 2],
                 [2, 2, 2, 2]], (2, 2), (2, 2), "linear_ramp", [[2, 2, 2, 2],
                                                                [2, 2, 2, 2],
                                                                [2, 2, 2, 2],
                                                                [2, 2, 2, 2]]
        ),

        (
                [[0, 0, 0, 1, 0],
                 [1, 0, 1, 0, 0],
                 [0, 1, 0, 0, 1],
                 [2, 2, 0, 2, 0],
                 [0, 0, 0, 0, 2]], (2, 3), (2, 3), "maximum", [[0, 0, 0, 1, 0, 1],
                                                               [1, 0, 1, 0, 0, 1],
                                                               [0, 1, 0, 0, 1, 1],
                                                               [2, 2, 0, 2, 0, 2],
                                                               [0, 0, 0, 0, 2, 2],
                                                               [2, 2, 1, 2, 2, 2]]
        ),

        pytest.param(
            [[0, 0, 0, 0, 0],
             [1, 0, 0, 1, 1],
             [0, 1, 1, 0, 0],
             [2, 0, 2, 2, 2],
             [0, 2, 0, 0, 0]], (2, 2), (1, 4), "constant",

            [[0, 0, 0, 0, 0],
             [1, 0, 0, 1, 1],
             [0, 1, 1, 0, 0],
             [2, 0, 2, 2, 2],
             [0, 2, 0, 0, 0]], marks=pytest.mark.xfail(strict=True))
    ]
)
def test_exact_pad_returned_padded_image(fake_image, window_shape, stride, mode, expected):
    # Arrange:
    fake_image = np.asarray(fake_image)
    expected_padded_image = np.asarray(expected)

    # Act:
    actual_padded_image, calculated_pad = ao.exact_pad(fake_image, window_shape, stride, mode)

    # Assert:
    assert np.array_equal(actual_padded_image, expected_padded_image)


@pytest.mark.parametrize(
    "fake_image, window_shape, stride, mode, expected_pad",
    [
        (
                [[[0], [0], [0], [1], [0]],
                 [[1], [0], [1], [0], [0]],
                 [[0], [1], [0], [0], [1]],
                 [[2], [2], [0], [2], [0]],
                 [[0], [0], [0], [0], [2]]], (3, 3, 1), (3, 3, 1), "constant", (1, 1, 0)
        ),

        (
                [[[0], [0], [0], [1], [0]],
                 [[1], [0], [1], [0], [0]],
                 [[0], [1], [0], [0], [1]],
                 [[0], [0], [0], [0], [0]],
                 [[0], [0], [2], [0], [0]],
                 [[2], [2], [0], [2], [0]],
                 [[0], [0], [0], [0], [2]]], (7, 5, 1), (3, 5, 1), "edge", (0, 0, 0)
        ),

        (
                [[2, 2, 2, 2],
                 [2, 2, 2, 2],
                 [2, 2, 2, 2],
                 [2, 2, 2, 2]], 2, 2, "linear_ramp", (0, 0)
        ),

        (
                [[0, 0, 0, 1, 0],
                 [1, 0, 1, 0, 0],
                 [0, 1, 0, 0, 1],
                 [2, 2, 0, 2, 0]], (2, 3), (2, 3), "maximum", (0, 1)
        ),

        pytest.param(
            [[0, 0, 0, 0, 0],
             [1, 0, 0, 1, 1],
             [0, 1, 1, 0, 0],
             [2, 0, 2, 2, 2],
             [0, 2, 0, 0, 0]], (2, 2), (2, 3), "constant", (0, 1), marks=pytest.mark.xfail(strict=True)),

        pytest.param(
            [[[0], [0], [0], [1], [0]],
             [[1], [0], [1], [0], [0]],
             [[0], [1], [0], [0], [1]],
             [[2], [2], [0], [2], [0]],
             [[0], [0], [0], [0], [2]]], (3, 3), (3, 3), "constant", (1, 1), marks=pytest.mark.xfail(strict=True)),

        pytest.param(
            [[[0], [0], [0], [1], [0]],
             [[1], [0], [1], [0], [0]],
             [[0], [1], [0], [0], [1]],
             [[2], [2], [0], [2], [0]],
             [[0], [0], [0], [0], [2]]], 3, 3, "constant", (1, 1), marks=pytest.mark.xfail(strict=True))
    ]
)
def test_exact_pad_returned_pad_shape(fake_image, window_shape, stride, mode, expected_pad):
    # Arrange:
    fake_image = np.asarray(fake_image)

    # Act:
    actual_padded_image, actual_pad = ao.exact_pad(fake_image, window_shape, stride, mode)

    # Assert:
    assert actual_pad == expected_pad


@pytest.mark.parametrize(
    "array_shape_after_resize, resize_settings",
    [
        ((10, 10, 1), {"height": 10, "width": 10, "mode": "linear"}),
        ((5, 10, 1), {"height": 5, "width": 10, "mode": "spline"}),
        ((15, 10, 1), {"height": 15, "width": 10, "mode": "linear"}),
        ((10, 5, 1), {"height": 10, "width": 5, "mode": "spline"}),
        ((10, 15, 1), {"height": 10, "width": 15, "mode": "linear"}),
        ((15, 15, 1), {"height": 15, "width": 15, "mode": "spline"}),
        ((5, 5, 1), {"height": 5, "width": 5, "mode": "linear"}),

        pytest.param((9, 9, 1), {"height": 0, "width": 0, "mode": "linear"},
                     marks=pytest.mark.xfail(strict=True)),
        pytest.param((8, 8, 1), {"height": 0, "width": 0, "mode": "linear"},
                     marks=pytest.mark.xfail(strict=True))
    ]
)
def test_resize2d(array_shape_after_resize, resize_settings):
    # Arrange:
    x = [[[3], [3], [3], [1], [1], [1], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [2], [2], [2], [4], [4], [4], [4]],
         [[3], [3], [3], [2], [2], [2], [4], [4], [4], [4]]]
    fake_image = np.asarray(x)

    # Act:
    actual = ao.resize_2d(fake_image, **resize_settings)

    # Assert:
    assert np.array_equal(array_shape_after_resize, actual.shape)


@pytest.mark.parametrize(
    "array_shape_after_interpolate, interpolate_settings",
    [
        ((20, 20, 1), {"height_amp_factor": 2, "width_amp_factor": 2, "mode": "linear"}),
        ((50, 30, 1), {"height_amp_factor": 5, "width_amp_factor": 3, "mode": "spline"}),
        ((30, 40, 1), {"height_amp_factor": 3, "width_amp_factor": 4, "mode": "linear"}),
        ((40, 40, 1), {"height_amp_factor": 4, "width_amp_factor": 4, "mode": "spline"}),

        pytest.param((9, 9, 1), {"height_amp_factor": 0, "width_amp_factor": 0, "mode": "linear"},
                     marks=pytest.mark.xfail(strict=True)),
        pytest.param((8, 8, 1), {"height_amp_factor": 0, "width_amp_factor": 0, "mode": "linear"},
                     marks=pytest.mark.xfail(strict=True))
    ]
)
def test_interpolate2d(array_shape_after_interpolate, interpolate_settings):
    # Arrange:
    x = [[[3], [3], [3], [1], [1], [1], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [0], [0], [0], [4], [4], [4], [4]],
         [[3], [3], [3], [2], [2], [2], [4], [4], [4], [4]],
         [[3], [3], [3], [2], [2], [2], [4], [4], [4], [4]]]
    fake_image = np.asarray(x)

    # Act:
    actual = ao.interpolate_image_2d(fake_image, **interpolate_settings)

    # Assert:
    assert np.array_equal(array_shape_after_interpolate, actual.shape)


@pytest.mark.parametrize(
    "array_shape, strides",
    [
        ((120, 80), (20, 30)),
    ]
)
def test_gauss_weight_map(array_shape, strides):
    # Act
    actual = ao.gauss_weight_map(array_shape, strides)

    # Assert
    assert np.amax(actual) == 1.0
    assert np.amin(actual) == 0.0
    assert actual.shape == (120, 80)
