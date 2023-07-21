""" Copyright 2019 IBM Research. All Rights Reserved.

    - Test functions for seisfast.transformations.seismic.horizon.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from rockml.data.adapter import FEATURE_NAME, LABEL_NAME
from rockml.data.adapter.seismic.segy.poststack import Direction, _PostStackProps as props
from rockml.data.transformations.seismic import horizon as hor


def _get_stub_post_stack_datum(features, label, direction, line_number, pixel_depth, column):
    seismic_stub = Mock()
    seismic_stub.features = features
    seismic_stub.label = label
    seismic_stub.direction = direction
    seismic_stub.line_number = line_number
    seismic_stub.pixel_depth = pixel_depth
    seismic_stub.column = column
    return seismic_stub


class TestConvertHorizon:
    @pytest.mark.parametrize(
        "horizon_settings",
        [
            ({"horizon_names": ['hrz_0', 'hrz_1', 'hrz_2', 'hrz_3', 'hrz_4', 'hrz_5', 'hrz_6', 'hrz_7'],
              "crop_top": 0, "crop_left": 0, "inline_resolution": 1, "crossline_resolution": 1, "correction": 10}),
        ]
    )
    def test_convert_horizon_init(self, horizon_settings):
        # Arrange
        convert = hor.ConvertHorizon(**horizon_settings)

        # Act
        horizon_names = convert.horizon_names
        crop_top = convert.crop_top
        crop_left = convert.crop_left
        inline_resolution = convert.inline_resolution
        crossline_resolution = convert.crossline_resolution
        correction = convert.correction

        # Assert
        assert horizon_names == horizon_settings["horizon_names"]
        assert crop_top == horizon_settings["crop_top"]
        assert crop_left == horizon_settings["crop_left"]
        assert inline_resolution == horizon_settings["inline_resolution"]
        assert crossline_resolution == horizon_settings["crossline_resolution"]
        assert correction == horizon_settings["correction"]

    @pytest.mark.parametrize(
        "horizon_settings, datum_settings",
        [
            ({"horizon_names": ['hrz_0', 'hrz_1', 'hrz_2', 'hrz_3', 'hrz_4', 'hrz_5', 'hrz_6', 'hrz_7'],
              "crop_top": 0, "crop_left": 0, "inline_resolution": 1, "crossline_resolution": 0, "correction": 10},

             {FEATURE_NAME: np.random.rand(463, 951, 1),
              LABEL_NAME: np.random.rand(463, 951),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 470,
              props.COLUMN.value: 300,
              props.PIXEL_DEPTH.value: 0})])
    def test_convert_horizon_call(self, horizon_settings, datum_settings):
        # Arrange
        convert = hor.ConvertHorizon(**horizon_settings)

        # Act
        fake_datum = _get_stub_post_stack_datum(**datum_settings)
        actual_datum = convert(fake_datum)

        # Assert

        assert len(actual_datum) == 8
        assert actual_datum[0].horizon_name == "hrz_0"
        assert actual_datum[0].columns.INLINE.value == "inline"


class TestConcatenateHorizon:
    @pytest.mark.parametrize(
        "horizon_settings, datum_settings",
        [
            ({"horizon_names": ['hrz_0', 'hrz_1', 'hrz_2', 'hrz_3', 'hrz_4', 'hrz_5', 'hrz_6', 'hrz_7'],
              "crop_top": 0, "crop_left": 0, "inline_resolution": 1, "crossline_resolution": 0, "correction": 10},

             {FEATURE_NAME: np.random.rand(463, 951, 1),
              LABEL_NAME: np.random.rand(463, 951),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 460,
              props.COLUMN.value: 300,
              props.PIXEL_DEPTH.value: 0})])
    def test_concatenate_horizon_call(self, horizon_settings, datum_settings):
        # Arrange
        convert = hor.ConvertHorizon(**horizon_settings)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)
        converted_datum = convert(fake_datum)

        # Act
        actual_datum = hor.ConcatenateHorizon().__call__(converted_datum)

        assert len(actual_datum) == 8
        assert actual_datum[0].point_map.shape == (951, 1)


class TestRemoveMutedTraces:
    @pytest.mark.parametrize(
        "valid_mask_settings, segy_info_settings",
        [
            (np.random.rand(651, 951),

             {"range_inlines": [100, 750],
              "range_crosslines": [300, 1250],
              "num_inlines": 651, "num_crosslines": 951, "range_time_depth": [0, 1848.0],
              "num_time_depth": 463, "res_inline": 1, "res_crossline": 1, "res_time_depth": 4.0,
              "range_x": [6054167, 6295763],
              "range_y": [60735564, 60904632]})])
    def test_remove_muted_traces_init(self, valid_mask_settings, segy_info_settings):
        remove = hor.RemoveMutedTraces(valid_mask_settings, segy_info_settings)

        valid_mask = remove.valid_mask
        segy_info = remove.segy_info

        assert valid_mask_settings.all() == valid_mask.all()
        assert segy_info_settings == segy_info

    @pytest.mark.parametrize(
        "valid_mask_settings, segy_info_settings, horizon_settings, datum_settings",
        [
            (np.random.rand(651, 951),

             {"range_inlines": [100, 750],
              "range_crosslines": [300, 1250],
              "num_inlines": 651, "num_crosslines": 951, "range_time_depth": [0, 1848.0],
              "num_time_depth": 463, "res_inline": 1, "res_crossline": 1, "res_time_depth": 4.0,
              "range_x": [6054167, 6295763],
              "range_y": [60735564, 60904632]},

             {"horizon_names": ['hrz_0', 'hrz_1', 'hrz_2', 'hrz_3', 'hrz_4', 'hrz_5', 'hrz_6', 'hrz_7'],
              "crop_top": 0, "crop_left": 0, "inline_resolution": 1, "crossline_resolution": 0, "correction": 10},

             {FEATURE_NAME: np.random.rand(463, 951, 1),
              LABEL_NAME: np.random.rand(463, 951),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 460,
              props.COLUMN.value: 300,
              props.PIXEL_DEPTH.value: 0})])
    def test_remove_muted_traces_call(self, valid_mask_settings, segy_info_settings, horizon_settings, datum_settings):
        # Arrange
        convert = hor.ConvertHorizon(**horizon_settings)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)
        converted_datum = convert(fake_datum)

        # Act
        removed = hor.RemoveMutedTraces(valid_mask_settings, segy_info_settings).__call__(converted_datum[0])

        # Assert
        assert removed.horizon_name == 'hrz_0'
        assert removed.point_map.shape == (951, 1)
