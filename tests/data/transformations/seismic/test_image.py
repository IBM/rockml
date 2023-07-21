""" Copyright 2023 IBM Research. All Rights Reserved.

    - Test functions for seisfast.transformations.seismic.image.
"""

from functools import partial
from typing import Tuple
from unittest.mock import Mock

import numpy as np
import pytest
from rockml.data.adapter import FEATURE_NAME, LABEL_NAME
from rockml.data.adapter.seismic.segy.poststack import Direction, _PostStackProps as props, PostStackDatum
from rockml.data.array_ops import gauss_weight_map
from rockml.data.transformations import Composer
from rockml.data.transformations.seismic import image as im
from rockml.data.transformations.seismic.image import ReconstructFromWindows, ViewAsWindows


def _get_stub_post_stack_datum(features, label, direction, line_number, pixel_depth, column):
    seismic_stub = Mock()
    seismic_stub.features = features
    seismic_stub.label = label
    seismic_stub.direction = direction
    seismic_stub.line_number = line_number
    seismic_stub.pixel_depth = pixel_depth
    seismic_stub.column = column
    return seismic_stub


def _get_stub_filter(is_none):
    filter_stub = Mock()
    if not is_none:
        filter_stub.side_effect = lambda x: x
    else:
        filter_stub.side_effect = lambda x: None
    filter_stub.__str__ = lambda x: "<Fake_Filter!>"
    return filter_stub


class TestCrop2D:
    @pytest.mark.parametrize(
        "crop2d_settings",
        [
            ({"crop_left": 0, "crop_right": 0, "crop_top": 0, "crop_bottom": 0, "ignore_label": False}),
            ({"crop_left": 5, "crop_right": 10, "crop_top": 5, "crop_bottom": 9, "ignore_label": True}),
            ({"crop_left": 7, "crop_right": 0, "crop_top": 3, "crop_bottom": 10, "ignore_label": False})
        ]
    )
    def test_crop_init(self, crop2d_settings):
        # Arrange:
        crop_transformation = im.Crop2D(**crop2d_settings)

        # Act:
        actual_crop_left = crop_transformation.crop_left
        actual_crop_right = crop_transformation.crop_right
        actual_crop_top = crop_transformation.crop_top
        actual_crop_bottom = crop_transformation.crop_bottom
        actual_ignore_label = crop_transformation.ignore_label

        # Assert:
        assert actual_crop_left == crop2d_settings["crop_left"]
        assert actual_crop_right == crop2d_settings["crop_right"]
        assert actual_crop_top == crop2d_settings["crop_top"]
        assert actual_crop_bottom == crop2d_settings["crop_bottom"]
        assert actual_ignore_label == crop2d_settings["ignore_label"]

    @pytest.mark.parametrize(
        "datum_settings, crop2d_settings, expected_dict",
        [
            ({FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                        [[1], [1], [2], [2]],
                                        [[3], [3], [4], [4]],
                                        [[3], [3], [4], [4]]]),
              LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                      [1, 1, 2, 2],
                                      [3, 3, 4, 4],
                                      [3, 3, 4, 4]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             {"crop_left": 0, "crop_right": 2, "crop_top": 0, "crop_bottom": 2, "ignore_label": False},
             {"expected_features": np.asarray([[[1], [1]],
                                               [[1], [1]]]),
              "expected_label": np.asarray([[1, 1],
                                            [1, 1]]),
              "expected_depth": 0, "expected_column": 0}),

            ({FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                        [[1], [1], [2], [2]],
                                        [[3], [3], [4], [4]],
                                        [[3], [3], [4], [4]]]),
              LABEL_NAME: 5,
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             {"crop_left": 1, "crop_right": 1, "crop_top": 1, "crop_bottom": 1, "ignore_label": True},
             {"expected_features": np.asarray([[[1], [2]],
                                               [[3], [4]]]),
              "expected_label": 5,
              "expected_depth": 1, "expected_column": 1}),

            pytest.param({FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                                    [[1], [1], [2], [2]],
                                                    [[3], [3], [4], [4]],
                                                    [[3], [3], [4], [4]]]),
                          LABEL_NAME: 5,
                          props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
                          props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
                         {"crop_left": 1, "crop_right": 1, "crop_top": 1, "crop_bottom": 1, "ignore_label": False},
                         {"expected_features": np.asarray([[[1], [2]],
                                                           [[3], [4]]]),
                          "expected_label": 5,
                          "expected_depth": 1, "expected_column": 1}, marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_crop_call_ignoring_label_or_not(self, datum_settings, crop2d_settings, expected_dict):
        # Arrange:
        crop_transformation = im.Crop2D(**crop2d_settings)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        actual_datum = crop_transformation(fake_datum)

        # Assert:
        assert np.array_equal(actual_datum.features, expected_dict["expected_features"])
        assert np.array_equal(actual_datum.label, expected_dict["expected_label"])
        assert actual_datum.pixel_depth == expected_dict["expected_depth"]
        assert actual_datum.column == expected_dict["expected_column"]

    @pytest.mark.parametrize(
        "datum_settings, crop2d_settings, expected_dict",
        [
            ({FEATURE_NAME: np.asarray([[1, 1, 2, 2],
                                        [1, 1, 2, 2],
                                        [3, 3, 4, 4],
                                        [3, 3, 4, 4]]),
              LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                      [1, 1, 2, 2],
                                      [3, 3, 4, 4],
                                      [3, 3, 4, 4]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             {"crop_left": 0, "crop_right": 2, "crop_top": 0, "crop_bottom": 2, "ignore_label": True},
             {"expected_features": np.asarray([[1, 1],
                                               [1, 1]]),
              "expected_label": np.asarray([[1, 1],
                                            [1, 1]]),
              "expected_depth": 0, "expected_column": 0}),

            ({FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                        [[1], [1], [2], [2]],
                                        [[3], [3], [4], [4]],
                                        [[3], [3], [4], [4]]]),
              LABEL_NAME: 5,
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             {"crop_left": 1, "crop_right": 1, "crop_top": 1, "crop_bottom": 1, "ignore_label": False},
             {"expected_features": np.asarray([[[1], [2]],
                                               [[3], [4]]]),
              "expected_label": 5,
              "expected_depth": 1, "expected_column": 1})
        ]
    )
    def test_crop_call_value_error_for_feature_not_dim3_and_label_dim_not_2(self, datum_settings, crop2d_settings,
                                                                            expected_dict):
        # Arrange:
        crop_transformation = im.Crop2D(**crop2d_settings)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Assert:
        with pytest.raises(ValueError):
            # Act:
            crop_transformation(fake_datum)

    @pytest.mark.parametrize(
        "crop_left, crop_right, crop_top, crop_bottom, ignore_label",
        [
            (0, 0, 0, 0, False)
        ]
    )
    def test_crop_str(self, crop_left, crop_right, crop_top, crop_bottom, ignore_label):
        # Arrange:
        crop_transformation = im.Crop2D(crop_left, crop_right, crop_top, crop_bottom, ignore_label)
        expected_output_string = f"<Crop2D LRTB: {crop_left, crop_right, crop_top, crop_bottom}>"

        # Act:
        actual_output_string = crop_transformation.__str__()

        # Assert:
        assert actual_output_string == expected_output_string


class TestScaleIntensity:
    @pytest.mark.parametrize(
        "scale_intensity_settings",
        [
            ({"gray_levels": 0, "percentile": 0.0}),
            ({"gray_levels": 5, "percentile": 10.0}),
            ({"gray_levels": 7, "percentile": 35.0})
        ]
    )
    def test_scale_intensity_init(self, scale_intensity_settings):
        # Arrange:
        scale_object = im.ScaleIntensity(**scale_intensity_settings)

        # Act:
        actual_gray_levels = scale_object.gray_levels
        actual_percentile = scale_object.percentile

        # Assert:
        assert actual_gray_levels == scale_intensity_settings["gray_levels"]
        assert actual_percentile == scale_intensity_settings["percentile"]

    @pytest.mark.parametrize(
        "scale_intensity_settings, datum_settings, expected_features",
        [
            ({"gray_levels": 11, "percentile": 5.0},
             {FEATURE_NAME: np.asarray([[[100], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
                                        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
                                        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
                                        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
                                        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]]),
              LABEL_NAME: np.zeros([5, 10]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             [[[255], [25], [51], [76], [102], [127], [153], [178], [204], [255]],
              [[0], [25], [51], [76], [102], [127], [153], [178], [204], [255]],
              [[0], [25], [51], [76], [102], [127], [153], [178], [204], [255]],
              [[0], [25], [51], [76], [102], [127], [153], [178], [204], [255]],
              [[0], [25], [51], [76], [102], [127], [153], [178], [204], [255]]])
        ]
    )
    def test_scale_intensity_call(self, scale_intensity_settings, datum_settings, expected_features):
        # Arrange:
        scale_object = im.ScaleIntensity(**scale_intensity_settings)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        actual_datum = scale_object(fake_datum)

        # Assert:
        assert np.array_equal(actual_datum.features, expected_features)

    @pytest.mark.parametrize(
        "scale_intensity_settings, expected_string_output",
        [
            ({"gray_levels": 0, "percentile": 0.0}, "<ScaleIntensity 0 levels, 0.0 % >"),
            ({"gray_levels": 5, "percentile": 10.0}, "<ScaleIntensity 5 levels, 10.0 % >"),
            ({"gray_levels": 7, "percentile": 35.0}, "<ScaleIntensity 7 levels, 35.0 % >")
        ]
    )
    def test_scale_intensity_str(self, scale_intensity_settings, expected_string_output):
        # Arrange:
        scale_object = im.ScaleIntensity(**scale_intensity_settings)

        # Act:
        actual_string_output = scale_object.__str__()

        # Assert:
        assert actual_string_output == expected_string_output


class TestFillSegmentationMask:
    @pytest.mark.parametrize(
        "datum_settings, expected_label",
        [
            ({FEATURE_NAME: np.asarray([[[1], [0], [0], [2], [0]],
                                        [[1], [0], [0], [1], [1]],
                                        [[1], [2], [1], [1], [1]],
                                        [[2], [1], [2], [2], [1]],
                                        [[2], [2], [2], [1], [2]]]),
              LABEL_NAME: np.asarray([[0, 0, 0, 0, 0],
                                      [1, 0, 0, 1, 1],
                                      [0, 1, 1, 0, 0],
                                      [2, 0, 2, 2, 2],
                                      [0, 2, 0, 0, 0]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             [[0, 0, 0, 0, 0],
              [1, 0, 0, 1, 1],
              [1, 1, 1, 1, 1],
              [2, 1, 2, 2, 2],
              [2, 2, 2, 2, 2]])
        ]
    )
    def test_fill_segmentation_mask_call(self, datum_settings, expected_label):
        # Arrange:
        fill_object = im.FillSegmentationMask()
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        actual_datum = fill_object(fake_datum)

        # Assert:
        assert np.array_equal(actual_datum.label, expected_label)

    @pytest.mark.parametrize(
        "datum_settings, expected_label",
        [
            ({FEATURE_NAME: np.asarray([[[1], [0], [0], [2], [0]],
                                        [[1], [0], [0], [1], [1]]]),
              LABEL_NAME: 5,
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             5)
        ]
    )
    def test_fill_segmentation_mask_call_with_invalid_label(self, datum_settings, expected_label):
        # Arrange:
        fill_object = im.FillSegmentationMask()
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Assert:
        with pytest.raises(ValueError):
            # Act:
            fill_object(fake_datum)

    @pytest.mark.parametrize(
        "expected_string_output",
        [
            "<FillSegmentation>",

            pytest.param("FillSegmentation", marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_fill_segmentation_mask_str(self, expected_string_output):
        # Arrange:
        fill_object = im.FillSegmentationMask()

        # Act:
        actual_string_output = fill_object.__str__()

        # Assert:
        assert actual_string_output == expected_string_output


class TestBinarizeMask:
    @pytest.mark.parametrize(
        "datum_settings, expected_label",
        [
            ({FEATURE_NAME: np.asarray([[[1], [0], [0], [2], [0]],
                                        [[1], [0], [0], [1], [1]],
                                        [[1], [2], [1], [1], [1]],
                                        [[2], [1], [2], [2], [1]],
                                        [[2], [2], [2], [1], [2]]]),
              LABEL_NAME: np.asarray([[0, 0, 0, 0, 0],
                                      [1, 0, 0, 1, 1],
                                      [0, 1, 1, 0, 0],
                                      [2, 0, 2, 2, 2],
                                      [0, 2, 0, 0, 0]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             [[0, 0, 0, 0, 0],
              [1, 0, 0, 1, 1],
              [0, 1, 1, 0, 0],
              [1, 0, 1, 1, 1],
              [0, 1, 0, 0, 0]])
        ]
    )
    def test_binarize_mask_call(self, datum_settings, expected_label):
        # Arrange:
        binarize_object = im.BinarizeMask()
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        actual_datum = binarize_object(fake_datum)

        # Assert:
        assert np.array_equal(actual_datum.label, expected_label)

    @pytest.mark.parametrize(
        "datum_settings, expected_label",
        [
            ({FEATURE_NAME: np.asarray([[[1], [0], [0], [2], [0]],
                                        [[1], [0], [0], [1], [1]]]),
              LABEL_NAME: 5,
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             5)
        ]
    )
    def test_binarize_mask_call_with_invalid_label(self, datum_settings, expected_label):
        # Arrange:
        binarize_object = im.BinarizeMask()
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Assert:
        with pytest.raises(ValueError):
            # Act:
            binarize_object(fake_datum)

    @pytest.mark.parametrize(
        "expected_string_output",
        [
            "<BinarizeMask>",

            pytest.param("BinarizeMask", marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_binarize_mask_str(self, expected_string_output):
        # Arrange:
        binarize_object = im.BinarizeMask()

        # Act:
        actual_string_output = binarize_object.__str__()

        # Assert:
        assert actual_string_output == expected_string_output


class TestThickenLinesMask:
    @pytest.mark.parametrize(
        "n_points",
        [
            1,
            2,
            3
        ]
    )
    def test_thicken_lines_mask_init(self, n_points):
        # Arrange:
        thicken_object = im.ThickenLinesMask(n_points)

        # Act:
        actual_n_points = thicken_object.n_points

        # Assert:
        assert actual_n_points == n_points

    @pytest.mark.parametrize(
        "n_points, datum_settings, expected_label",
        [
            (1,
             {FEATURE_NAME: np.asarray([[[0], [0], [0], [1], [1]],
                                        [[1], [0], [1], [0], [1]],
                                        [[1], [1], [0], [0], [1]],
                                        [[1], [2], [0], [1], [1]],
                                        [[1], [2], [0], [0], [1]],
                                        [[2], [2], [2], [2], [1]],
                                        [[1], [2], [2], [2], [2]]]),
              LABEL_NAME: np.asarray([[0, 0, 0, 1, 0],
                                      [1, 0, 1, 0, 0],
                                      [0, 1, 0, 0, 1],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 2, 0, 0],
                                      [2, 2, 0, 2, 0],
                                      [0, 0, 0, 0, 2]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             [[1, 0, 1, 1, 0],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 0, 1],
              [0, 1, 2, 0, 1],
              [2, 2, 2, 2, 0],
              [2, 2, 2, 2, 2],
              [2, 2, 0, 2, 2]])
        ]
    )
    def test_thicken_lines_mask_call(self, n_points, datum_settings, expected_label):
        # Arrange:
        thicken_object = im.ThickenLinesMask(n_points)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        actual_datum = thicken_object(fake_datum)

        # Assert:
        assert np.array_equal(actual_datum.label, expected_label)

    @pytest.mark.parametrize(
        "n_points, datum_settings, expected_label",
        [
            (1,
             {FEATURE_NAME: np.asarray([[[0], [0], [0], [1], [1]],
                                        [[1], [0], [1], [0], [1]],
                                        [[1], [1], [0], [0], [1]],
                                        [[1], [2], [0], [1], [1]],
                                        [[1], [2], [0], [0], [1]],
                                        [[2], [2], [2], [2], [1]],
                                        [[1], [2], [2], [2], [2]]]),
              LABEL_NAME: 5,
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             5)
        ]
    )
    def test_thicken_lines_mask_call_with_invalid_label(self, n_points, datum_settings, expected_label):
        # Arrange:
        thicken_object = im.ThickenLinesMask(n_points)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Assert:
        with pytest.raises(ValueError):
            # Act:
            thicken_object(fake_datum)

    @pytest.mark.parametrize(
        "n_points, expected_string_output",
        [
            (1, "<ThickenLinesMask n_points: 1>"),
            (2, "<ThickenLinesMask n_points: 2>"),
            (3, "<ThickenLinesMask n_points: 3>")
        ]
    )
    def test_thicken_lines_mask_str(self, n_points, expected_string_output):
        # Arrange:
        thicken_object = im.ThickenLinesMask(n_points)

        # Act:
        actual_string_output = thicken_object.__str__()

        # Assert:
        assert actual_string_output == expected_string_output


class TestViewAsWindows:
    @pytest.mark.parametrize(
        "view_as_windows_settings",
        [
            ({"tile_shape": (10, 10),
              "stride_shape": (5, 5),
              "filters": []})
        ]
    )
    def test_view_as_windows_init(self, view_as_windows_settings):
        # Arrange:
        view_a_windows_object = im.ViewAsWindows(**view_as_windows_settings)

        # Act:
        actual_tile_shape = view_a_windows_object.tile_shape
        actual_stride_shape = view_a_windows_object.stride_shape
        actual_filters = view_a_windows_object.filters

        # Assert:
        assert actual_tile_shape == view_as_windows_settings["tile_shape"]
        assert actual_stride_shape == view_as_windows_settings["stride_shape"]
        assert actual_filters == view_as_windows_settings["filters"]

    @pytest.mark.parametrize(
        "view_as_windows_settings, datum_settings, expected_datum_list_length",
        [
            ({"tile_shape": (2, 2), "stride_shape": (2, 2),
              "filters": [_get_stub_filter(False), _get_stub_filter(False)]
              },
             {FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                        [[1], [1], [2], [2]],
                                        [[3], [3], [3], [3]],
                                        [[3], [3], [3], [3]]]),
              LABEL_NAME: np.asarray([[0, 0, 4, 4],
                                      [0, 0, 4, 4],
                                      [5, 5, 6, 6],
                                      [5, 5, 6, 6]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             4),

            ({"tile_shape": (2, 2), "stride_shape": (2, 2),
              "filters": [_get_stub_filter(True), _get_stub_filter(False)]
              },
             {FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                        [[1], [1], [2], [2]],
                                        [[3], [3], [3], [3]],
                                        [[3], [3], [3], [3]]]),
              LABEL_NAME: np.asarray([[0, 0, 4, 4],
                                      [0, 0, 4, 4],
                                      [5, 5, 6, 6],
                                      [5, 5, 6, 6]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             0)
        ]
    )
    def test_view_as_windows_returned_datum_list_length(self, view_as_windows_settings, datum_settings,
                                                        expected_datum_list_length):
        # Arrange:
        view_as_windows_settings = view_as_windows_settings
        view_a_windows_object = im.ViewAsWindows(**view_as_windows_settings)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        actual_datum_list = view_a_windows_object(fake_datum)

        # Assert:
        assert len(actual_datum_list) == expected_datum_list_length

    @pytest.mark.parametrize(
        "view_as_windows_settings, datum_settings, which_datum, expected_dict",
        [
            ({"tile_shape": (2, 2), "stride_shape": (2, 2),
              "filters": [_get_stub_filter(False), _get_stub_filter(False)]
              },
             {FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                        [[1], [1], [2], [2]],
                                        [[3], [3], [3], [3]],
                                        [[3], [3], [3], [3]]]),
              LABEL_NAME: np.asarray([[0, 0, 4, 4],
                                      [0, 0, 4, 4],
                                      [5, 5, 6, 6],
                                      [5, 5, 6, 6]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             3,
             {"expected_features": np.asarray([[[3], [3]],
                                               [[3], [3]]]),
              "expected_label": np.asarray([[6, 6],
                                            [6, 6]]),
              "expected_depth": 2, "expected_column": 2})
        ]
    )
    def test_view_as_windows_returned_datum_features_and_labels(self, view_as_windows_settings, datum_settings,
                                                                which_datum, expected_dict):
        # Arrange:
        view_as_windows_settings = view_as_windows_settings
        view_a_windows_object = im.ViewAsWindows(**view_as_windows_settings)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        actual_datum_list = view_a_windows_object(fake_datum)

        # Assert:
        assert np.array_equal(actual_datum_list[which_datum].features, expected_dict["expected_features"])
        assert np.array_equal(actual_datum_list[which_datum].label, expected_dict["expected_label"])
        assert actual_datum_list[which_datum].pixel_depth == expected_dict["expected_depth"]
        assert actual_datum_list[which_datum].column == expected_dict["expected_column"]

    @pytest.mark.parametrize(
        "view_as_windows_settings, expected_string_output",
        [
            ({"tile_shape": (10, 10), "stride_shape": (5, 5),
              "filters": [_get_stub_filter(True), _get_stub_filter(False)]},
             "ViewAsWindows[<Fake_Filter!>,<Fake_Filter!>]")
        ]
    )
    def test_view_as_windows_str(self, view_as_windows_settings, expected_string_output):
        # Arrange:
        view_a_windows_object = im.ViewAsWindows(**view_as_windows_settings)

        # Act:
        actual_string_output = view_a_windows_object.__str__()

        # Assert:
        assert actual_string_output == expected_string_output


class TestReconstructFromWindows:
    @pytest.mark.parametrize(
        "image_shape, label_shape, tile_shape, stride_shape, auto_pad",
        [
            ((100, 100, 1), (100, 100, 3), (10, 10), (5, 5), False),
            # Testing basic functionality
            ((100, 100, 1), (100, 100), (10, 10), (10, 10), False),
            ((100, 100, 1), (100, 100, 1), (10, 10), (10, 10), False),
            ((100, 100, 3), (100, 100, 6), (10, 10), (10, 10), False),
            # Testing auto_pad
            ((100, 100, 1), (100, 100), (10, 10), (10, 10), True),
            ((100, 100, 1), (100, 100, 1), (10, 10), (10, 10), True),
            ((100, 100, 3), (100, 100, 6), (10, 10), (10, 10), True),
            ((100, 100, 1), (100, 100), (13, 13), (11, 11), True),
            ((100, 100, 1), (100, 100, 1), (13, 13), (11, 11), True),
            ((100, 100, 3), (100, 100, 6), (13, 13), (11, 11), True),
            # Testing non-squared windowing
            ((100, 100, 1), (100, 100), (10, 5), (10, 5), False),
            ((100, 100, 1), (100, 100, 1), (10, 5), (10, 5), False),
            ((100, 100, 3), (100, 100, 6), (10, 5), (10, 5), False),
            ((100, 100, 1), (100, 100), (10, 5), (10, 5), True),
            ((100, 100, 1), (100, 100, 1), (10, 5), (10, 5), True),
            ((100, 100, 3), (100, 100, 6), (10, 5), (10, 5), True),
            ((100, 100, 1), (100, 100), (13, 7), (11, 6), True),
            ((100, 100, 1), (100, 100, 1), (13, 7), (11, 6), True),
            ((100, 100, 3), (100, 100, 6), (13, 7), (11, 6), True),
        ]
    )
    def test_basic(self, image_shape, label_shape, tile_shape, stride_shape, auto_pad):
        def make_datum(line):
            img_shape = image_shape[0] * image_shape[1] * image_shape[2]
            lbl_shape = label_shape[0] * label_shape[1]
            if len(label_shape) == 3:
                lbl_shape *= label_shape[2]
            return PostStackDatum(
                features=np.arange(img_shape, dtype=np.float32).reshape(image_shape),
                label=np.arange(lbl_shape, dtype=np.int32).reshape(label_shape),
                direction=Direction.INLINE,
                line_number=line,
                pixel_depth=100,
                column=100
            )

        datum_list_original = [
            make_datum(100),
            make_datum(101)
        ]
        datum_list_proc = Composer(
            [
                ViewAsWindows(tile_shape, stride_shape, auto_pad=auto_pad)
            ]
        ).apply(datum_list_original)

        datum_list_proc = ReconstructFromWindows(
            image_shape[:-1], image_shape[:-1],
            stride_shape
        ).__call__(datum_list_proc)

        assert np.sum(datum_list_proc[0].features - datum_list_original[0].features) == 0
        assert np.sum(datum_list_proc[0].label - datum_list_original[0].label) == 0
        assert datum_list_proc[0].line_number == datum_list_original[0].line_number
        assert datum_list_proc[0].features.shape == datum_list_original[0].features.shape
        assert datum_list_proc[0].label.shape == datum_list_original[0].label.shape
        assert datum_list_proc[0].column == datum_list_original[0].column
        assert datum_list_proc[0].pixel_depth == datum_list_original[0].pixel_depth

    @pytest.mark.parametrize(
        "image_shape, label_shape, tile_shape, stride_shape, auto_pad",
        [
            ((100, 100, 1), (100, 100, 3), (10, 10), (5, 5), False),
            ((100, 100, 1), (100, 100, 3), (10, 10), (5, 5), True),
            ((100, 100, 1), (100, 100, 3), (10, 10), (6, 6), True),
        ]
    )
    def test_overlapping(self, image_shape, label_shape, tile_shape, stride_shape, auto_pad):
        def make_datum(line):
            img_shape = image_shape[0] * image_shape[1] * image_shape[2]
            lbl_shape = label_shape[0] * label_shape[1]
            if len(label_shape) == 3:
                lbl_shape *= label_shape[2]
            return PostStackDatum(
                features=np.arange(img_shape, dtype=np.float32).reshape(image_shape),
                label=np.ones(lbl_shape, dtype=np.int32).reshape(label_shape),
                direction=Direction.INLINE,
                line_number=line,
                pixel_depth=0,
                column=0
            )

        def weight_map(array: np.ndarray, strides: Tuple[int, int]) -> np.ndarray:
            x = np.expand_dims(
                gauss_weight_map(array.shape[:-1], strides),
                axis=-1
            )

            # return x * array / (array.shape[0] / strides[0] * array.shape[1] / strides[1])
            return x * array

        datum_list_original = [
            make_datum(100),
            make_datum(101)
        ]
        datum_list_proc = Composer(
            [
                ViewAsWindows(tile_shape, stride_shape, auto_pad=auto_pad)
            ]
        ).apply(datum_list_original)

        datum_list_proc = ReconstructFromWindows(
            image_shape[:-1], image_shape[:-1],
            stride_shape,
            overlapping_fn=partial(weight_map, strides=stride_shape)
        ).__call__(datum_list_proc)

        assert np.all(datum_list_proc[0].label[:, :, 0] == datum_list_proc[0].label[:, :, 1])
        assert np.all(datum_list_proc[0].label[:, :, 1] == datum_list_proc[0].label[:, :, 2])


class TestResize2D:
    @pytest.mark.parametrize(
        "resize2d_settings",
        [
            ({"height": 100, "width": 100, "mode": "linear", "ignore_label": False}),
            ({"height": 100, "width": 100, "mode": "linear", "ignore_label": True}),
            ({"height": 200, "width": 300, "mode": "spline", "ignore_label": False}),
            ({"height": 200, "width": 300, "mode": "spline", "ignore_label": True}),
        ]
    )
    def test_resize_init(self, resize2d_settings):
        # Arrange:
        resize_transformation = im.Resize2D(**resize2d_settings)

        # Act:
        actual_height = resize_transformation.height
        actual_width = resize_transformation.width
        actual_mode = resize_transformation.mode
        actual_ignore_label = resize_transformation.ignore_label

        # Assert:
        assert actual_height == resize2d_settings["height"]
        assert actual_width == resize2d_settings["width"]
        assert actual_mode == resize2d_settings["mode"]
        assert actual_ignore_label == resize2d_settings["ignore_label"]

    @pytest.mark.parametrize(
        "datum_settings, resize2d_settings, expected_dict",
        [
            ({FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                        [[1], [1], [2], [2]],
                                        [[3], [3], [4], [4]],
                                        [[3], [3], [4], [4]]]),
              LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                      [1, 1, 2, 2],
                                      [3, 3, 4, 4],
                                      [3, 3, 4, 4]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             {"height": 4, "width": 2, "mode": "linear", "ignore_label": False},
             {"expected_features_shape": (4, 2, 1),
              "expected_label_shape": (4, 2, 1)}),

            ({FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                        [[1], [1], [2], [2]],
                                        [[3], [3], [4], [4]],
                                        [[3], [3], [4], [4]]]),
              LABEL_NAME: 5,
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             {"height": 2, "width": 2, "mode": "spline", "ignore_label": True},
             {"expected_features_shape": (2, 2, 1),
              "expected_label": 5}),

            pytest.param({"features": np.asarray([[[1], [1], [2], [2]],
                                                  [[1], [1], [2], [2]],
                                                  [[3], [3], [4], [4]],
                                                  [[3], [3], [4], [4]]]),
                          "label": 5,
                          props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
                          props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
                         {"height": 2, "width": 2, "mode": "spline", "ignore_label": False},
                         {"expected_features": (2, 2, 1),
                          "expected_label": 5}, marks=pytest.mark.xfail(strict=True)),

        ]
    )
    def test_resize_call_ignoring_label_or_not(self, datum_settings, resize2d_settings, expected_dict):
        # Arrange:
        resize_transformation = im.Resize2D(**resize2d_settings)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        actual_datum = resize_transformation(fake_datum)
        # Assert:
        assert np.array_equal(actual_datum.features.shape, expected_dict["expected_features_shape"])

        if resize2d_settings["ignore_label"] is False:
            assert np.array_equal(actual_datum.label.shape, expected_dict["expected_label_shape"])
        else:
            assert actual_datum.label == expected_dict["expected_label"]

    @pytest.mark.parametrize(
        "datum_settings, resize2d_settings, expected_dict",
        [
            ({FEATURE_NAME: np.asarray([[1, 1, 2, 2],
                                        [1, 1, 2, 2],
                                        [3, 3, 4, 4],
                                        [3, 3, 4, 4]]),
              LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                      [1, 1, 2, 2],
                                      [3, 3, 4, 4],
                                      [3, 3, 4, 4]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             {"height": 2, "width": 2, "mode": "linear", "ignore_label": True},
             {"expected_features_shape": (2, 2, 1),
              "expected_label_shape": (2, 2, 1)}),

            ({FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                        [[1], [1], [2], [2]],
                                        [[3], [3], [4], [4]],
                                        [[3], [3], [4], [4]]]),
              LABEL_NAME: 5,
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             {"height": 2, "width": 2, "mode": "spline", "ignore_label": False},
             {"expected_features": np.asarray([[[1], [2]],
                                               [[3], [4]]]),
              "expected_label": 5})
        ]
    )
    def test_resize_call_value_error_for_feature_not_dim3_and_label_dim_not_2(self, datum_settings, resize2d_settings,
                                                                              expected_dict):
        # Arrange:
        resize_transformation = im.Resize2D(**resize2d_settings)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Assert:
        with pytest.raises(ValueError):
            # Act:
            resize_transformation(fake_datum)

    @pytest.mark.parametrize(
        "height, width, mode, ignore_label",
        [
            (10, 0, "linear", False),
            (10, 0, "spline", False)
        ]

    )
    def test_resize_str(self, height, width, mode, ignore_label):
        # Arrange:
        resize_transformation = im.Resize2D(height, width, mode, ignore_label)
        expected_output_string = f"<Resize2D mode {mode}, H x W: {height}, {width}>"

        # Act:
        actual_output_string = resize_transformation.__str__()

        # Assert:
        assert actual_output_string == expected_output_string


class TestInterpolate2D:
    @pytest.mark.parametrize(
        "interpolate2d_settings",
        [
            ({"height_amp_factor": 100, "width_amp_factor": 100, "mode": "linear", "ignore_label": False}),
            ({"height_amp_factor": 100, "width_amp_factor": 100, "mode": "spline", "ignore_label": True}),
            ({"height_amp_factor": 200, "width_amp_factor": 300, "mode": "linear", "ignore_label": False}),
            ({"height_amp_factor": 200, "width_amp_factor": 300, "mode": "spline", "ignore_label": True}),
        ]
    )
    def test_interpolate_init(self, interpolate2d_settings):
        # Arrange:
        interpolate_transformation = im.Interpolate2D(**interpolate2d_settings)

        # Act:
        actual_height = interpolate_transformation.height_amp_factor
        actual_width = interpolate_transformation.width_amp_factor
        actual_mode = interpolate_transformation.mode
        actual_ignore_label = interpolate_transformation.ignore_label

        # Assert:
        assert actual_height == interpolate2d_settings["height_amp_factor"]
        assert actual_width == interpolate2d_settings["width_amp_factor"]
        assert actual_mode == interpolate2d_settings["mode"]
        assert actual_ignore_label == interpolate2d_settings["ignore_label"]

    @pytest.mark.parametrize(
        "datum_settings, interpolate2d_settings, expected_dict",
        [
            ({FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                        [[1], [1], [2], [2]],
                                        [[3], [3], [4], [4]],
                                        [[3], [3], [4], [4]]]),
              LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                      [1, 1, 2, 2],
                                      [3, 3, 4, 4],
                                      [3, 3, 4, 4]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             {"height_amp_factor": 4, "width_amp_factor": 2, "mode": "linear", "ignore_label": False},
             {"expected_features_shape": (16, 8, 1),
              "expected_label_shape": (16, 8, 1)}),

            ({FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                        [[1], [1], [2], [2]],
                                        [[3], [3], [4], [4]],
                                        [[3], [3], [4], [4]]]),
              LABEL_NAME: 5,
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             {"height_amp_factor": 2, "width_amp_factor": 2, "mode": "spline", "ignore_label": True},
             {"expected_features_shape": (8, 8, 1),
              "expected_label": 5}),

            pytest.param({FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                                    [[1], [1], [2], [2]],
                                                    [[3], [3], [4], [4]],
                                                    [[3], [3], [4], [4]]]),
                          LABEL_NAME: 5,
                          "info": {"any": "meta_data"}},
                         {"height_amp_factor": 2, "width_amp_factor": 2, "mode": "spline", "ignore_label": False},
                         {"expected_features": (8, 8, 1),
                          "expected_label": 5}, marks=pytest.mark.xfail(strict=True)),

        ]
    )
    def test_interpolate_call_ignoring_label_or_not(self, datum_settings, interpolate2d_settings, expected_dict):
        # Arrange:
        interpolate_transformation = im.Interpolate2D(**interpolate2d_settings)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        actual_datum = interpolate_transformation(fake_datum)
        # Assert:
        assert np.array_equal(actual_datum.features.shape, expected_dict["expected_features_shape"])

        if interpolate2d_settings["ignore_label"] is False:
            assert np.array_equal(actual_datum.label.shape, expected_dict["expected_label_shape"])
        else:
            assert actual_datum.label == expected_dict["expected_label"]

    @pytest.mark.parametrize(
        "datum_settings, interpolate2d_settings, expected_dict",
        [
            ({FEATURE_NAME: np.asarray([[1, 1, 2, 2],
                                        [1, 1, 2, 2],
                                        [3, 3, 4, 4],
                                        [3, 3, 4, 4]]),
              LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                      [1, 1, 2, 2],
                                      [3, 3, 4, 4],
                                      [3, 3, 4, 4]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             {"height_amp_factor": 2, "width_amp_factor": 2, "mode": "linear", "ignore_label": True},
             {"expected_features_shape": (8, 8, 1),
              "expected_label_shape": (8, 8, 1)})
        ]
    )
    def test_interpolate_call_value_error_for_feature_not_dim3_and_label_dim_not_2(self, datum_settings,
                                                                                   interpolate2d_settings,
                                                                                   expected_dict):
        # Arrange:
        interpolate_transformation = im.Interpolate2D(**interpolate2d_settings)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Assert:
        with pytest.raises(ValueError):
            # Act:
            interpolate_transformation(fake_datum)

    @pytest.mark.parametrize(
        "height_amp_factor, width_amp_factor, mode, ignore_label",
        [
            (2, 2, "linear", False),
            (2, 2, "spline", False)
        ]

    )
    def test_resize_str(self, height_amp_factor, width_amp_factor, mode, ignore_label):
        # Arrange:
        interpolate_transformation = im.Interpolate2D(height_amp_factor, width_amp_factor, mode, ignore_label)
        expected_output_string = f"<Interpolate mode {mode}, H x W: {height_amp_factor}, {width_amp_factor}>"

        # Act:
        actual_output_string = interpolate_transformation.__str__()

        # Assert:
        assert actual_output_string == expected_output_string


class TestPostStackLabelArgMax:
    @pytest.mark.parametrize(
        "input_label",
        [
            np.stack((np.zeros((3, 3)), np.ones((3, 3))), axis=-1),
            np.zeros((10, 10, 7), dtype=np.uint8),
        ]
    )
    def test_arg_max(self, input_label):
        datum = PostStackDatum(
            features=np.arange(int(np.prod(input_label.shape)), dtype=np.float32).reshape(input_label.shape),
            label=input_label,
            direction=Direction.INLINE,
            line_number=100,
            pixel_depth=100,
            column=100
        )

        result = im.PostStackLabelArgMax()(datum)

        assert result.label.shape[0] == input_label.shape[0]
        assert result.label.shape[1] == input_label.shape[1]
        assert len(result.label.shape) == len(input_label.shape) - 1
