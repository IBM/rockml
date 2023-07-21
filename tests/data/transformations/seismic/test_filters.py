""" Copyright 2023 IBM Research. All Rights Reserved.

    - Test functions for seisfast.transformations.seismic.filters.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from rockml.data.adapter import FEATURE_NAME, LABEL_NAME
from rockml.data.adapter.seismic.segy.poststack import Direction, _PostStackProps as props
from rockml.data.transformations.seismic import filters


def _get_stub_post_stack_datum(features, label, direction, line_number, pixel_depth, column):
    seismic_stub = Mock()
    seismic_stub.features = features
    seismic_stub.label = label
    seismic_stub.direction = direction
    seismic_stub.line_number = line_number
    seismic_stub.pixel_depth = pixel_depth
    seismic_stub.column = column
    return seismic_stub


class TestMinimumTextureFilter:
    @pytest.mark.parametrize(
        "min_texture_in_features",
        [
            0.5
        ]
    )
    def test_minimum_texture_filter_init(self, min_texture_in_features):
        # Arrange:
        minimum_texture_filter = filters.MinimumTextureFilter(min_texture_in_features)

        # Act:
        actual_min_texture_in_features = minimum_texture_filter.min_texture_in_features

        # Assert:
        assert actual_min_texture_in_features == min_texture_in_features

    @pytest.mark.parametrize(
        "min_texture_in_features, datum_settings",
        [
            (0.5,
             {FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                        [[1], [1], [2], [2]],
                                        [[3], [3], [4], [4]],
                                        [[3], [3], [4], [4]]]),
              LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                      [1, 1, 2, 2],
                                      [3, 3, 4, 4],
                                      [3, 3, 4, 4]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}),

            pytest.param(1.0,
                         {"features": np.asarray([[[0], [0], [0], [0]],
                                                  [[0], [0], [0], [0]],
                                                  [[0], [0], [0], [0]],
                                                  [[0], [3], [4], [4]]]),
                          "label": np.asarray([[1, 1, 2, 2],
                                               [1, 1, 2, 2],
                                               [3, 3, 4, 4],
                                               [3, 3, 4, 4]]),
                          props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
                          props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
                         marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_minimum_texture_filter_call_returns_datum(self, min_texture_in_features, datum_settings):
        # Arrange:
        minimum_texture_filter = filters.MinimumTextureFilter(min_texture_in_features)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        actual_datum = minimum_texture_filter(fake_datum)

        # Assert:
        assert np.array_equal(actual_datum.features, datum_settings["features"])

    @pytest.mark.parametrize(
        "min_texture_in_features, datum_settings",
        [
            (1.0,
             {FEATURE_NAME: np.asarray([[[0], [0], [0], [0]],
                                        [[0], [0], [0], [0]],
                                        [[0], [0], [0], [0]],
                                        [[0], [0], [0], [0]]]),
              LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                      [1, 1, 2, 2],
                                      [3, 3, 4, 4],
                                      [3, 3, 4, 4]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}),

            (0.3,
             {FEATURE_NAME: np.asarray([[[0], [0], [0], [0]],
                                        [[0], [0], [0], [0]],
                                        [[0], [0], [0], [0]],
                                        [[1], [1], [2], [3]]]),
              LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                      [1, 1, 2, 2],
                                      [3, 3, 4, 4],
                                      [3, 3, 4, 4]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}),

            (0.5,
             {FEATURE_NAME: np.asarray([[[1], [0], [0], [0]],
                                        [[1], [0], [0], [0]],
                                        [[2], [0], [0], [0]],
                                        [[2], [0], [0], [0]]]),
              LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                      [1, 1, 2, 2],
                                      [3, 3, 4, 4],
                                      [3, 3, 4, 4]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}),

            pytest.param(0.5,
                         {FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                                    [[1], [1], [2], [2]],
                                                    [[3], [3], [4], [4]],
                                                    [[3], [3], [4], [4]]]),
                          LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                                  [1, 1, 2, 2],
                                                  [3, 3, 4, 4],
                                                  [3, 3, 4, 4]]),
                          props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
                          props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
                         marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_minimum_texture_filter_call_returns_none(self, min_texture_in_features, datum_settings):
        # Arrange:
        minimum_texture_filter = filters.MinimumTextureFilter(min_texture_in_features)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        actual_datum = minimum_texture_filter(fake_datum)

        # Assert:
        assert actual_datum is None

    @pytest.mark.parametrize(
        "min_texture_in_features, expected_output_string",
        [
            (0.5, "<MinimumTextureFilter min_texture_in_tile: 0.5>")
        ]
    )
    def test_minimum_texture_filter_str(self, min_texture_in_features, expected_output_string):
        # Arrange:
        minimum_texture_filter = filters.MinimumTextureFilter(min_texture_in_features)

        # Act:
        actual_output_string = minimum_texture_filter.__str__()

        # Assert:
        assert actual_output_string == expected_output_string


class TestClassificationFilter:
    @pytest.mark.parametrize(
        "noise",
        [
            0.5
        ]
    )
    def test_classification_filter_init(self, noise):
        # Arrange:
        classification_filter = filters.ClassificationFilter(noise)

        # Act:
        classification_filter_noise = classification_filter.noise

        # Assert:
        assert classification_filter_noise == noise

    @pytest.mark.parametrize(
        "noise, datum_settings",
        [
            (0.2,
             {FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                        [[1], [1], [2], [2]],
                                        [[3], [3], [4], [4]],
                                        [[3], [3], [4], [4]]]),
              LABEL_NAME: np.asarray([[1, 0, 0, 1],
                                      [1, 1, 1, 1],
                                      [1, 1, 1, 1],
                                      [1, 1, 1, 1]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}),

            pytest.param(0.7,
                         {FEATURE_NAME: np.asarray([[[0], [0], [0], [0]],
                                                    [[0], [0], [0], [0]],
                                                    [[0], [0], [0], [0]],
                                                    [[0], [3], [4], [4]]]),
                          LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                                  [1, 1, 2, 2],
                                                  [3, 3, 4, 4],
                                                  [3, 3, 4, 4]]),
                          props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
                          props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
                         marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_classification_filter_call_returns_datum(self, noise, datum_settings):
        # Arrange:
        classification_filter = filters.ClassificationFilter(noise)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        actual_datum = classification_filter(fake_datum)

        # Assert:
        assert np.array_equal(actual_datum.features, datum_settings["features"])

    @pytest.mark.parametrize(
        "noise, datum_settings",
        [
            (0.7,
             {"features": np.asarray([[[0], [0], [0], [0]],
                                      [[0], [0], [0], [0]],
                                      [[0], [0], [0], [0]],
                                      [[0], [0], [0], [0]]]),
              "label": np.asarray([[1, 1, 2, 2],
                                   [1, 1, 2, 2],
                                   [3, 3, 4, 4],
                                   [3, 3, 4, 4]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}),

            (0.1,
             {FEATURE_NAME: np.asarray([[[0], [0], [0], [0]],
                                        [[0], [0], [0], [0]],
                                        [[0], [0], [0], [0]],
                                        [[1], [1], [2], [3]]]),
              LABEL_NAME: np.asarray([[0, 0, 0, 0],
                                      [0, 0, 0, 0],
                                      [0, 3, 0, 0],
                                      [3, 3, 0, 0]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}),

            (0.4,
             {FEATURE_NAME: np.asarray([[[1], [0], [0], [0]],
                                        [[1], [0], [0], [0]],
                                        [[2], [0], [0], [0]],
                                        [[2], [0], [0], [0]]]),
              LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                      [1, 1, 2, 2],
                                      [1, 1, 4, 4],
                                      [1, 1, 4, 4]]),
              props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}),

            pytest.param(0.6,
                         {FEATURE_NAME: np.asarray([[[1], [1], [2], [2]],
                                                    [[1], [1], [2], [2]],
                                                    [[3], [3], [4], [4]],
                                                    [[3], [3], [4], [4]]]),
                          LABEL_NAME: np.asarray([[1, 1, 2, 2],
                                                  [1, 1, 2, 2],
                                                  [1, 1, 4, 4],
                                                  [1, 1, 4, 4]]),
                          props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
                          props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
                         marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_classification_filter_call_returns_none(self, noise, datum_settings):
        # Arrange:
        classification_filter = filters.ClassificationFilter(noise)
        fake_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        actual_datum = classification_filter(fake_datum)

        # Assert:
        assert actual_datum is None

    @pytest.mark.parametrize(
        "noise, expected_output_string",
        [
            (0.5, "<ClassificationFilter noise: 0.5>")
        ]
    )
    def test_classification_filter_str(self, noise, expected_output_string):
        # Arrange:
        classification_filter = filters.ClassificationFilter(noise)

        # Act:
        actual_output_string = classification_filter.__str__()

        # Assert:
        assert actual_output_string == expected_output_string
