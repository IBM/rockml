""" Copyright 2019 IBM Research. All Rights Reserved. """

import pytest

from rockml.data.adapter.seismic.horizon import *


class TestHorizonDatum:
    @pytest.mark.parametrize(
        "point_map, horizon_name",
        [
            (
                    pd.DataFrame({'pixel_depth': [123, 123, 122, 120]},
                                 index=[(470, 300), (470, 301), (470, 302), (470, 356)]), "hrz_0"),

        ]
    )
    def test_horizon_datum_init(self, point_map, horizon_name):
        datum = HorizonDatum(point_map, horizon_name)

        # Arrange
        expected_point_map = point_map
        expected_horizon_name = horizon_name

        # Act
        actual_point_map = datum.point_map
        actual_horizon_name = datum.horizon_name

        # Assert
        assert expected_horizon_name == actual_horizon_name
        assert pd.testing.assert_frame_equal(actual_point_map, expected_point_map) is None

    @pytest.mark.parametrize(
        "point_map, horizon_name",
        [
            (
                    pd.DataFrame({'pixel_depth': [123, 123, 122, 120]},
                                 index=[(470, 300), (470, 301), (470, 302), (470, 356)]), "hrz_1"),

        ]
    )
    def test_horizon_str(self, point_map, horizon_name):
        datum = HorizonDatum(point_map, horizon_name)

        expected_string = datum.__str__()

        actual_string = f"<HorizonDatum {horizon_name}>"

        assert actual_string == expected_string


class TestHorizonDataDumper:
    @pytest.mark.parametrize(
        "path, segy, point_map, horizon_name",
        [
            ("segy_test_data/",
             "segy_test_data/cropped_netherlands.sgy",
             pd.DataFrame({'pixel_depth': [123, 123, 122, 120]},
                          index=[(501, 551), (501, 552), (501, 570), (502, 570)]),
             "hrz_1")

        ]
    )
    def test_to_text_file(self, path, segy, point_map, horizon_name):
        datum_list = [HorizonDatum(point_map, horizon_name)]

        HorizonDataDumper.to_text_file(datum_list, path, segy)

        with open(path + "hrz_1.xyz", 'r+') as file:
            assert file.readline() == '"(501, 551)" 611828.19 6083752.50 496.00\n'


class TestHorizonAdapter:

    @pytest.mark.parametrize(
        "path_horizon, time_depth_resolution, initial_time",
        [
            (["segy_test_data/Germanic_Group.xyz",
              "segy_test_data/North_Sea_Group.xyz"],
             4.0, 4)

        ]
    )
    def test_horizon_adapter_init(self, path_horizon, time_depth_resolution, initial_time):
        adapter = HorizonAdapter(path_horizon, time_depth_resolution, initial_time)

        expected_path_horizon = path_horizon
        expected_time_depth = time_depth_resolution
        expected_initial_time = initial_time

        actual_path_horizon = adapter.horizons_paths
        actual_time_depth = adapter.time_depth_resolution
        actual_initial_time = adapter.initial_time

        assert expected_initial_time == actual_initial_time
        assert expected_path_horizon == actual_path_horizon
        assert expected_time_depth == actual_time_depth

    @pytest.mark.parametrize(
        "path_horizon, time_depth_resolution, initial_time, hor_path",
        [
            (["segy_test_data/Germanic_Group.xyz",
              "segy_test_data/North_Sea_Group.xyz"],
             4.0, 4, "segy_test_data/Germanic_Group.xyz")

        ]
    )
    def test_load_horizon(self, path_horizon, time_depth_resolution, initial_time, hor_path):
        adapter = HorizonAdapter(path_horizon, time_depth_resolution, initial_time)

        frame = adapter.load_horizon(hor_path)

        assert frame.shape == (5120, 1)

    @pytest.mark.parametrize(
        "path_horizon, series_a, series_b, series_c, time_depth_resolution, initial_time",
        [
            (["segy_test_data/Germanic_Group.xyz",
              "segy_test_data/North_Sea_Group.xyz"],
             pd.Series([1402.23999]), pd.Series([1403.18994]), pd.Series([1404.09998]),
             4.0, 4)

        ]
    )
    def test_z_to_pixel_depth(self, path_horizon, series_a, series_b, series_c, time_depth_resolution, initial_time):
        adapter = HorizonAdapter(path_horizon, time_depth_resolution, initial_time)

        a = adapter.z_to_pixel_depth(series_a, time_depth_resolution, initial_time)
        b = adapter.z_to_pixel_depth(series_b, time_depth_resolution, initial_time)
        c = adapter.z_to_pixel_depth(series_c, time_depth_resolution, initial_time)

        assert a == 350
        assert b == 350
        assert c == 350

    @pytest.mark.parametrize(
        "path_horizon, time_depth_resolution, initial_time",
        [
            (["segy_test_data/Germanic_Group.xyz",
              "segy_test_data/North_Sea_Group.xyz"],
             4.0, 4)

        ]
    )
    def test_len_horizon(self, path_horizon, time_depth_resolution, initial_time):
        adapter = HorizonAdapter(path_horizon, time_depth_resolution, initial_time)

        expected_horizons = len(path_horizon)

        actual_horizons = adapter.__len__()

        assert expected_horizons == actual_horizons
