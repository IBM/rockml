""" Copyright 2019 IBM Research. All Rights Reserved.

    - Test functions for seisfast.adapter.seismic.segy.
"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pytest
from rockml.data.adapter import FEATURE_NAME, LABEL_NAME
from rockml.data.adapter.seismic.segy.poststack import PostStackAdapter2D, Direction, _PostStackProps as props, \
    PostStackDatum, PostStackDataDumper
from seisfast.io import PostStackSEGY


class TestPostStackDatum:
    @pytest.mark.parametrize(
        "features, labels, info",
        [
            (np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
             {props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}),

            (np.asarray([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]), np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
             {props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 121,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}),

            pytest.param(np.asarray([8, 7, 6, 5, 4, 3, 2, 1]), np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                         {props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
                          props.PIXEL_DEPTH.value: 0,
                          props.COLUMN.value: 0}, marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_post_stack_datum_init_label_is_nd_array(self, features, labels, info):
        # Arrange:
        seismic_datum = PostStackDatum(features, labels, **info)

        expected_features = features
        expected_label = labels
        expected_direction = info[props.DIRECTION.value]
        expected_line_number = info[props.LINE_NUMBER.value]
        expected_pixel_depth = info[props.PIXEL_DEPTH.value]
        expected_pixel_column = info[props.COLUMN.value]

        # Act:
        actual_features = seismic_datum.features
        actual_label = seismic_datum.label
        actual_direction = seismic_datum.direction
        actual_line_number = seismic_datum.line_number
        actual_pixel_depth = seismic_datum.pixel_depth
        actual_pixel_column = seismic_datum.column

        # Assert:
        assert actual_features.shape == actual_label.shape
        assert np.array_equal(actual_features, expected_features)
        assert np.array_equal(actual_label, expected_label)
        assert actual_direction == expected_direction
        assert actual_line_number == expected_line_number
        assert actual_pixel_depth == expected_pixel_depth
        assert actual_pixel_column == expected_pixel_column

    @pytest.mark.parametrize(
        "features, labels, info",
        [
            (np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 9,
             {props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}),

            (np.asarray([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]), 5,
             {props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0})
        ]
    )
    def test_post_stack_datum_init_label_is_int(self, features, labels, info):
        # Arrange:
        seismic_datum = PostStackDatum(features, labels, **info)

        expected_label = labels
        int_number = 5

        # Act:
        actual_label = seismic_datum.label

        # Assert:
        assert actual_label == expected_label
        assert type(actual_label) == type(int_number)

    @pytest.mark.parametrize(
        "features, labels, info",
        [
            (np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 9.0,
             {props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}),

            (np.asarray([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]), 5.0,
             {props.DIRECTION.value: Direction.INLINE, props.LINE_NUMBER.value: 100,
              props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}),
        ]
    )
    def test_post_stack_datum_init_label_is_float(self, features, labels, info):
        # Arrange:
        seismic_datum = PostStackDatum(features, labels, **info)

        expected_label = labels
        float_number = 5.0

        # Act:
        actual_label = seismic_datum.label

        # Assert:
        assert actual_label == expected_label
        assert type(actual_label) == type(float_number)


def _get_stub_post_stack_datum(features, label, direction, line_number, pixel_depth, column):
    seismic_stub = Mock()
    seismic_stub.features = features
    seismic_stub.label = label
    seismic_stub.direction = direction
    seismic_stub.line_number = line_number
    seismic_stub.pixel_depth = pixel_depth
    seismic_stub.column = column
    return seismic_stub


class TestPostStackDataDumper:
    @pytest.mark.parametrize(
        "datum_settings, path",
        [
            ({FEATURE_NAME: np.asarray([[1, 1, 2, 2],
                                        [1, 1, 2, 2],
                                        [3, 3, 3, 3],
                                        [3, 3, 3, 3]]),
              LABEL_NAME: np.asarray([[0, 0, 4, 4],
                                      [0, 0, 4, 4],
                                      [5, 5, 6, 6],
                                      [5, 5, 6, 6]]),
              props.DIRECTION.value: Direction.INLINE,
              props.LINE_NUMBER.value: 100, props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             "/some_path"),
        ]
    )
    def test_to_hdf(self, datum_settings, path):  # , expected_features, expected_label, expected_info):
        # Arrange:
        fake_seismic_datum = _get_stub_post_stack_datum(**datum_settings)

        # Act:
        with patch('h5py.File') as h5:
            h5f = Mock()
            h5.return_value = h5f
            h5f.create_dataset.return_value = None
            PostStackDataDumper.to_hdf([fake_seismic_datum], path)

        # Assert:
        h5.assert_called_once_with(path, 'w')
        assert h5f.create_dataset.call_count == 6
        # h5f.create_dataset.assert_any_call(props.DIRECTION.value, chunks=True,
        #                                    data=np.asarray(datum_settings[props.DIRECTION.value], dtype=bytes))

    @pytest.mark.parametrize(
        "datum_settings, path",
        [
            ({FEATURE_NAME: np.asarray([[1, 1, 2, 2],
                                        [1, 1, 2, 2],
                                        [3, 3, 3, 3],
                                        [3, 3, 3, 3]]),
              LABEL_NAME: np.asarray([[0, 0, 4, 4],
                                      [0, 0, 4, 4],
                                      [5, 5, 6, 6],
                                      [5, 5, 6, 6]]),
              props.DIRECTION.value: Direction.INLINE,
              props.LINE_NUMBER.value: 100, props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0},
             "/some_path"),
        ]
    )
    def test_to_dict(self, datum_settings, path):  # , expected_features, expected_label, expected_info):
        # Arrange:
        fake_seismic_datum = _get_stub_post_stack_datum(**datum_settings)

        my_dict = PostStackDataDumper.to_dict([fake_seismic_datum])

        # Assert:
        assert (datum_settings['features'] == my_dict['features']).all()
        assert datum_settings['line_number'] == my_dict['line_number'][0]


class TestPostStackAdapter2D:
    def setup_method(self):
        self.segy_path = os.path.abspath('segy_test_data/test.sgy')
        self.horizon_path_list = [os.path.abspath('segy_test_data/horizons.txt')]

    @pytest.mark.parametrize(
        "segy_path, horizon_path_list, data_dict, expected_expanded_list",
        [
            ("path/to/segy.sgy",
             ["path/to/horizons.txt"],
             {Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             [(Direction.INLINE, 110), (Direction.CROSSLINE, 110)]
             ),
        ]
    )
    def test_post_stack_adapter2d_init(self, segy_path, horizon_path_list, data_dict, expected_expanded_list):
        # Arrange:
        segy = PostStackAdapter2D(segy_path=segy_path, horizons_path_list=horizon_path_list, data_dict=data_dict)

        # Act:
        actual_expanded_list = segy.expanded_list
        actual_segy_path = segy.segy_path
        actual_segy_raw_data = segy._segy_raw_data
        actual_segy_info = segy.segy_info
        actual_horizons_data = segy.horizons_data

        # Assert:
        assert actual_expanded_list == expected_expanded_list
        assert actual_segy_path == segy_path
        assert actual_segy_raw_data is None
        assert actual_segy_info is None
        assert actual_horizons_data == horizon_path_list

    @pytest.mark.parametrize(
        "segy_path, horizon_path_list, data_dict",
        [
            ("path/to/segy.sgy",
             [],
             None,
             ),
        ]
    )
    def test_post_stack_adapter2d_init_with_data_dict_as_none(self, segy_path, horizon_path_list, data_dict):
        # Arrange:
        segy = PostStackAdapter2D(segy_path=segy_path, horizons_path_list=horizon_path_list, data_dict=data_dict)

        # Act:
        actual_data_dict = segy.data_dict

        # Assert:
        assert actual_data_dict == dict()

    @pytest.mark.parametrize(
        "horizon_path_list, data_dict, expected_segy_info",
        [
            (["path/to/horizons.txt"],
             {Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             {'range_inlines': [110, 111], 'range_xlines': [300, 1250], 'num_ilines': 2, 'num_xlines': 951,
              'range_time_depth': [4, 1848.0], 'num_time_depth': 462, 'range_x': [6058278, 6295693],
              'range_y': [60738063, 60744948]}
             )
        ]
    )
    def test_post_stack_adapter2d_segy_raw_data(self, horizon_path_list, data_dict, expected_segy_info):
        # Arrange:
        segy = PostStackAdapter2D(segy_path=self.segy_path, horizons_path_list=horizon_path_list, data_dict=data_dict)

        # Act:

        actual_segy_raw_data = segy.segy_raw_data
        actual_segy_info = segy.segy_info

        # Assert:
        assert actual_segy_info['range_inlines'] == expected_segy_info['range_inlines']
        assert isinstance(actual_segy_raw_data, PostStackSEGY)

    @pytest.mark.parametrize(
        "horizon_path_list, data_dict, expected_segy_info",
        [
            (["path/to/horizons.txt"],
             {Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             {'range_inlines': [110, 111], 'range_xlines': [300, 1250], 'num_ilines': 2, 'num_xlines': 951,
              'range_time_depth': [4, 1848.0], 'num_time_depth': 462, 'range_x': [6058278, 6295693],
              'range_y': [60738063, 60744948]}
             )
        ]
    )
    def test_post_stack_adapter2d_initial_scan(self, horizon_path_list, data_dict, expected_segy_info):
        # Arrange:
        segy = PostStackAdapter2D(segy_path=self.segy_path, horizons_path_list=horizon_path_list, data_dict=data_dict)

        # Act:
        actual_segy_info = segy.initial_scan()

        # Assert:
        assert actual_segy_info['range_inlines'] == expected_segy_info['range_inlines']

    @pytest.mark.parametrize(
        "data_dict, direction, line_number, expected_segy_info, relevant_info_key",
        [
            ({Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             Direction.INLINE, 110,
             {'range_inlines': [110, 111], 'range_xlines': [300, 1250], 'num_ilines': 2, 'num_xlines': 951,
              'range_time_depth': [4, 1848.0], 'num_time_depth': 462, 'range_x': [6058278, 6295693],
              'range_y': [60738063, 60744948]},
             'num_xlines'
             ),
            ({Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             Direction.CROSSLINE, 110,
             {'range_inlines': [110, 111], 'range_xlines': [300, 1250], 'num_ilines': 2, 'num_xlines': 951,
              'range_time_depth': [4, 1848.0], 'num_time_depth': 462, 'range_x': [6058278, 6295693],
              'range_y': [60738063, 60744948]},
             'num_ilines'
             )
        ]
    )
    def test_post_stack_adapter2d_get_line(self, data_dict, direction, line_number, expected_segy_info,
                                           relevant_info_key):
        # Arrange:
        segy = PostStackAdapter2D(segy_path=self.segy_path, horizons_path_list=self.horizon_path_list,
                                  data_dict=data_dict)

        # Act:
        actual_datum = segy.get_line(direction, line_number)

        # Assert:
        assert actual_datum.features.shape == (expected_segy_info['num_time_depth'],
                                               expected_segy_info[relevant_info_key],
                                               1)

    @pytest.mark.parametrize(
        "data_dict, direction, line_number, expected_segy_info, relevant_info_key",
        [
            ({Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             Direction.BOTH, 110,
             {'range_inlines': [110, 111], 'range_xlines': [300, 1250], 'num_ilines': 2, 'num_xlines': 951,
              'range_time_depth': [4, 1848.0], 'num_time_depth': 462, 'range_x': [6058278, 6295693],
              'range_y': [60738063, 60744948]},
             'num_xlines'
             )
        ]
    )
    def test_post_stack_adapter2d_get_line_with_bad_direction_name(self, data_dict, direction, line_number,
                                                                   expected_segy_info, relevant_info_key):
        # Arrange:
        segy = PostStackAdapter2D(segy_path=self.segy_path, horizons_path_list=self.horizon_path_list,
                                  data_dict=data_dict)
        # Assert:
        with pytest.raises(ValueError):
            # Act:
            actual_datum = segy.get_line(direction, line_number)

    @pytest.mark.parametrize(
        "data_dict, direction, line_number",
        [
            ({Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             Direction.INLINE, 110
             ),

            ({Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             Direction.CROSSLINE, 110
             )
        ]
    )
    def test_post_stack_adapter2d_parse_info(self, data_dict, direction, line_number):
        # Arrange:
        segy = PostStackAdapter2D(segy_path=self.segy_path, horizons_path_list=self.horizon_path_list,
                                  data_dict=data_dict)
        _ = segy.segy_raw_data

        # Act:
        actual_info = segy.parse_info(direction, line_number)

        # Assert:
        assert actual_info[props.DIRECTION.value] == direction
        assert actual_info[props.LINE_NUMBER.value] == line_number

    @pytest.mark.parametrize(
        "segy_path, horizon_path_list, data_dict, expanded_list",
        [
            ("path/to/segy.sgy",
             ["path/to/horizons.txt"],
             {Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             [(Direction.INLINE, 110), (Direction.CROSSLINE, 110)]
             ),
        ]
    )
    def test_post_stack_adapter2d_len(self, segy_path, horizon_path_list, data_dict, expanded_list):
        # Arrange:
        segy = PostStackAdapter2D(segy_path=segy_path, horizons_path_list=horizon_path_list, data_dict=data_dict)

        # Act:
        actual_length = len(segy)

        # Assert:
        assert actual_length == len(expanded_list)

    @pytest.mark.parametrize(
        "data_dict, expanded_list",
        [
            ({Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             [(Direction.INLINE, 110), (Direction.CROSSLINE, 110)]
             ),
        ]
    )
    def test_post_stack_adapter2d_iter(self, data_dict, expanded_list):
        # Arrange:
        segy = PostStackAdapter2D(segy_path=self.segy_path, horizons_path_list=self.horizon_path_list,
                                  data_dict=data_dict)

        # Act:
        actual_datum = next(iter(segy))

        # Assert:
        assert isinstance(actual_datum, PostStackDatum)

    @pytest.mark.parametrize(
        "data_dict, expanded_list",
        [
            ({Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             [(Direction.INLINE, 110), (Direction.CROSSLINE, 110)]
             ),
        ]
    )
    def test_post_stack_adapter2d_getitem_int_key(self, data_dict, expanded_list):
        # Arrange:
        segy = PostStackAdapter2D(segy_path=self.segy_path, horizons_path_list=self.horizon_path_list,
                                  data_dict=data_dict)

        # Act:
        actual_datum = segy[0]

        # Assert:
        assert isinstance(actual_datum, PostStackDatum)

    @pytest.mark.parametrize(
        "data_dict, expanded_list",
        [
            ({Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             [(Direction.INLINE, 110), (Direction.CROSSLINE, 110)]
             ),
        ]
    )
    def test_post_stack_adapter2d_getitem_list_key(self, data_dict, expanded_list):
        # Arrange:
        segy = PostStackAdapter2D(segy_path=self.segy_path, horizons_path_list=self.horizon_path_list,
                                  data_dict=data_dict)

        # Act:
        actual_datum_list = segy[[0, 1]]

        # Assert:
        assert isinstance(actual_datum_list[0], PostStackDatum)

    @pytest.mark.parametrize(
        "data_dict, expanded_list",
        [
            ({Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             [(Direction.INLINE, 110), (Direction.CROSSLINE, 110)]
             ),
        ]
    )
    def test_post_stack_adapter2d_getitem_slice_key(self, data_dict, expanded_list):
        # Arrange:
        segy = PostStackAdapter2D(segy_path=self.segy_path, horizons_path_list=self.horizon_path_list,
                                  data_dict=data_dict)

        # Act:
        actual_datum_list = segy[:]

        # Assert:
        assert isinstance(actual_datum_list[0], PostStackDatum)

    @pytest.mark.parametrize(
        "data_dict, expanded_list",
        [
            ({Direction.INLINE.value: [[110, 111]], Direction.CROSSLINE.value: [[110, 111]]},
             [(Direction.INLINE, 110), (Direction.CROSSLINE, 110)]
             ),
        ]
    )
    def test_post_stack_adapter2d_getitem_key_error(self, data_dict, expanded_list):
        # Arrange:
        segy = PostStackAdapter2D(segy_path=self.segy_path, horizons_path_list=self.horizon_path_list,
                                  data_dict=data_dict)
        # Assert:
        with pytest.raises(KeyError):
            # Act:
            actual_datum_list = segy[""]
