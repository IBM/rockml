""" Copyright 2019 IBM Research. All Rights Reserved.

    - Test functions for seisfast.transformations.__init__.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from rockml.data.transformations import Composer


def _get_stub_filter(is_none):
    filter_stub = Mock()
    if not is_none:
        filter_stub.side_effect = lambda x: x
    else:
        filter_stub.side_effect = lambda x: None
    filter_stub.__str__ = lambda x: "<Fake_Filter!>"
    return filter_stub


def _get_stub_seismic_datum(features, label, info):
    seismic_stub = Mock()
    seismic_stub.features = features
    seismic_stub.label = label
    seismic_stub.info = info
    return seismic_stub


class TestComposer:
    @pytest.mark.parametrize(
        "transformations",
        [
            ([_get_stub_filter(False), _get_stub_filter(False)])
        ]
    )
    def test_composer_init(self, transformations):
        # Arrange:
        composer_object = Composer(transformations)

        # Act:
        actual_transformations = composer_object.transformations

        # Assert:
        assert len(actual_transformations) == len(transformations)
        assert actual_transformations[0].__str__() == transformations[0].__str__()
        assert actual_transformations[-1].__str__() == transformations[-1].__str__()

    @pytest.mark.parametrize(
        "transformations, expected_composer_string_output",
        [
            ([_get_stub_filter(False), _get_stub_filter(False)], "Composition[<Fake_Filter!>,<Fake_Filter!>]")
        ]
    )
    def test_composer_str(self, transformations, expected_composer_string_output):
        # Arrange:
        composer_object = Composer(transformations)

        # Act:
        actual_composer_string_output = composer_object.__str__()

        # Assert:
        assert actual_composer_string_output == expected_composer_string_output

    # @pytest.mark.parametrize(
    #     "transformations, path",
    #     [
    #         ([_get_stub_filter(False), _get_stub_filter(False)], "../../trash.txt")
    #     ]
    # )
    # def test_composer_dump(self, monkeypatch, transformations, path):
    #     # Arrange:
    #     composer_object = Composer(transformations)
    #     m = mock_open()
    #
    #     def mock_dump(object, to_file):
    #         return None
    #
    #     with patch('__main__.open', m):
    #         monkeypatch.setattr(pickle, 'dump', mock_dump)
    #         # Act:
    #         composer_object.dump(path)
    #
    #     # Assert:
    #     m.assert_called_once()

    @pytest.mark.parametrize(
        "transformations, bad_dataset",
        [
            ([_get_stub_filter(False), _get_stub_filter(False)], None)
        ]
    )
    def test_composer_apply_dataset_is_none(self, transformations, bad_dataset):
        # Arrange:
        composer_object = Composer(transformations)

        # Act:
        actual_dataset = composer_object.apply(bad_dataset)

        # Assert:
        assert actual_dataset is None

    @pytest.mark.parametrize(
        "transformations, dataset_is_not_list",
        [
            ([_get_stub_filter(False), _get_stub_filter(False)], _get_stub_seismic_datum([[[1], [1], [2], [2]],
                                                                                          [[1], [1], [2], [2]],
                                                                                          [[3], [3], [4], [4]],
                                                                                          [[3], [3], [4], [4]]],
                                                                                         [[1, 1, 2, 2],
                                                                                          [1, 1, 2, 2],
                                                                                          [3, 3, 4, 4],
                                                                                          [3, 3, 4, 4]],
                                                                                         {"any": "meta_data"}))
        ]
    )
    def test_composer_apply_dataset_is_not_list(self, transformations, dataset_is_not_list):
        # Arrange:
        composer_object = Composer(transformations)

        # Act:
        actual_dataset = composer_object.apply(dataset_is_not_list)

        # Assert:
        assert np.array_equal(actual_dataset.features, dataset_is_not_list.features)

    @pytest.mark.parametrize(
        "transformations, dataset_is_list",
        [
            ([_get_stub_filter(False), _get_stub_filter(False)],
             [_get_stub_seismic_datum([[[1], [1], [2], [2]],
                                       [[1], [1], [2], [2]],
                                       [[3], [3], [4], [4]],
                                       [[3], [3], [4], [4]]],
                                      [[1, 1, 2, 2],
                                       [1, 1, 2, 2],
                                       [3, 3, 4, 4],
                                       [3, 3, 4, 4]],
                                      {"any": "meta_data"}),
              _get_stub_seismic_datum([[[1], [1], [2], [2]],
                                       [[1], [1], [2], [2]],
                                       [[3], [3], [4], [4]],
                                       [[3], [3], [4], [4]]],
                                      [[1, 1, 2, 2],
                                       [1, 1, 2, 2],
                                       [3, 3, 4, 4],
                                       [3, 3, 4, 4]],
                                      {"any": "meta_data"})
              ]
             )
        ]
    )
    def test_composer_apply_dataset_is_a_valid_list(self, transformations, dataset_is_list):
        # Arrange:
        composer_object = Composer(transformations)

        # Act:
        actual_dataset = composer_object.apply(dataset_is_list)

        # Assert:
        assert np.array_equal(actual_dataset[0].features, dataset_is_list[0].features)
        assert np.array_equal(actual_dataset[-1].features, dataset_is_list[-1].features)

    @pytest.mark.parametrize(
        "transformations, dataset_is_list",
        [
            ([_get_stub_filter(True), _get_stub_filter(False)],
             [_get_stub_seismic_datum([[[1], [1], [2], [2]],
                                       [[1], [1], [2], [2]],
                                       [[3], [3], [4], [4]],
                                       [[3], [3], [4], [4]]],
                                      [[1, 1, 2, 2],
                                       [1, 1, 2, 2],
                                       [3, 3, 4, 4],
                                       [3, 3, 4, 4]],
                                      {"any": "meta_data"}),
              _get_stub_seismic_datum([[[1], [1], [2], [2]],
                                       [[1], [1], [2], [2]],
                                       [[3], [3], [4], [4]],
                                       [[3], [3], [4], [4]]],
                                      [[1, 1, 2, 2],
                                       [1, 1, 2, 2],
                                       [3, 3, 4, 4],
                                       [3, 3, 4, 4]],
                                      {"any": "meta_data"})
              ]
             ),
            ([_get_stub_filter(False), _get_stub_filter(True)],
             [_get_stub_seismic_datum([[[1], [1], [2], [2]],
                                       [[1], [1], [2], [2]],
                                       [[3], [3], [4], [4]],
                                       [[3], [3], [4], [4]]],
                                      [[1, 1, 2, 2],
                                       [1, 1, 2, 2],
                                       [3, 3, 4, 4],
                                       [3, 3, 4, 4]],
                                      {"any": "meta_data"}),
              _get_stub_seismic_datum([[[1], [1], [2], [2]],
                                       [[1], [1], [2], [2]],
                                       [[3], [3], [4], [4]],
                                       [[3], [3], [4], [4]]],
                                      [[1, 1, 2, 2],
                                       [1, 1, 2, 2],
                                       [3, 3, 4, 4],
                                       [3, 3, 4, 4]],
                                      {"any": "meta_data"})
              ]
             ),
            ([_get_stub_filter(True), _get_stub_filter(True)],
             [_get_stub_seismic_datum([[[1], [1], [2], [2]],
                                       [[1], [1], [2], [2]],
                                       [[3], [3], [4], [4]],
                                       [[3], [3], [4], [4]]],
                                      [[1, 1, 2, 2],
                                       [1, 1, 2, 2],
                                       [3, 3, 4, 4],
                                       [3, 3, 4, 4]],
                                      {"any": "meta_data"}),
              _get_stub_seismic_datum([[[1], [1], [2], [2]],
                                       [[1], [1], [2], [2]],
                                       [[3], [3], [4], [4]],
                                       [[3], [3], [4], [4]]],
                                      [[1, 1, 2, 2],
                                       [1, 1, 2, 2],
                                       [3, 3, 4, 4],
                                       [3, 3, 4, 4]],
                                      {"any": "meta_data"})
              ]
             )
        ]
    )
    def test_composer_apply_dataset_is_a_invalid_list(self, transformations, dataset_is_list):
        # Arrange:
        composer_object = Composer(transformations)

        # Act:
        actual_dataset = composer_object.apply(dataset_is_list)

        # Assert:
        assert len(actual_dataset) == 0
