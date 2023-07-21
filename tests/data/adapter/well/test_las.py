""" Copyright 2023 IBM Research. All Rights Reserved. """

import pytest

from rockml.data.adapter.well.las import LASDataAdapter, WellDatum


class TestLASDataAdapter:
    @pytest.mark.parametrize(
        "dir_path, numerical_logs, categorical_logs, depth_unit",
        [
            ('test_well', ['GR', 'DENS', 'RESD', 'NEUT'], ['TOP'], 'ft')
        ]
    )
    def test_las_data_adapter_init(self, dir_path, numerical_logs, categorical_logs, depth_unit):
        # Arrange:
        lass_data_adapter = LASDataAdapter(dir_path=dir_path, numerical_logs=numerical_logs,
                                           categorical_logs=categorical_logs,
                                           depth_unit=depth_unit)

        expected_dir_path = dir_path
        expected_numerical_logs = numerical_logs
        expected_categorical_logs = categorical_logs
        expected_depth_unit = depth_unit

        # Act:
        actual_dir_path = lass_data_adapter.dir_path
        actual_numerical_logs = lass_data_adapter.numerical_logs
        actual_categorical_logs = lass_data_adapter.categorical_logs
        actual_depth_unit = lass_data_adapter.depth_unit

        # Assert:
        assert actual_dir_path == expected_dir_path
        assert actual_numerical_logs == expected_numerical_logs
        assert actual_categorical_logs == expected_categorical_logs
        assert actual_depth_unit == expected_depth_unit

    @pytest.mark.parametrize(
        "dir_path, numerical_logs, categorical_logs, depth_unit, expected_length",
        [
            ('test_well', ['GR', 'DENS', 'RESD', 'NEUT'], ['TOP'], 'ft', 1)
        ]
    )
    def test_las_data_adapter_length(self, dir_path, numerical_logs, categorical_logs, depth_unit, expected_length):
        # Arrange/Act:
        lass_data_adapter = LASDataAdapter(dir_path=dir_path, numerical_logs=numerical_logs,
                                           categorical_logs=categorical_logs,
                                           depth_unit=depth_unit)

        # Assert:
        assert len(lass_data_adapter) == expected_length

    @pytest.mark.parametrize(
        "dir_path, numerical_logs, categorical_logs, depth_unit",
        [
            ('test_well', ['GR', 'DENS', 'RESD', 'NEUT'], ['TOP'], 'ft')
        ]
    )
    def test_las_data_adapter_iter(self, dir_path, numerical_logs, categorical_logs, depth_unit):
        # Arrange:
        lass_data_adapter = LASDataAdapter(dir_path=dir_path, numerical_logs=numerical_logs,
                                           categorical_logs=categorical_logs,
                                           depth_unit=depth_unit)

        # Act:
        actual_well_datum = next(iter(lass_data_adapter))

        # Assert:
        assert isinstance(actual_well_datum, WellDatum)

    @pytest.mark.parametrize(
        "dir_path, numerical_logs, categorical_logs, depth_unit",
        [
            ('test_well', ['GR', 'DENS', 'RESD', 'NEUT'], ['TOP'], 'ft')
        ]
    )
    def test_las_data_adapter_getitem_int_key(self, dir_path, numerical_logs, categorical_logs, depth_unit):
        # Arrange:
        lass_data_adapter = LASDataAdapter(dir_path=dir_path, numerical_logs=numerical_logs,
                                           categorical_logs=categorical_logs,
                                           depth_unit=depth_unit)

        # Act:
        actual_well_datum = lass_data_adapter[0]

        # Assert:
        assert isinstance(actual_well_datum, WellDatum)

    @pytest.mark.parametrize(
        "dir_path, numerical_logs, categorical_logs, depth_unit",
        [
            ('test_well', ['GR', 'DENS', 'RESD', 'NEUT'], ['TOP'], 'ft')
        ]
    )
    def test_las_data_adapter_getitem_list_key(self, dir_path, numerical_logs, categorical_logs, depth_unit):
        # Arrange:
        lass_data_adapter = LASDataAdapter(dir_path=dir_path, numerical_logs=numerical_logs,
                                           categorical_logs=categorical_logs,
                                           depth_unit=depth_unit)

        # Act:
        actual_well_datum_list = lass_data_adapter[[0]]

        # Assert:
        assert isinstance(actual_well_datum_list[0], WellDatum)

    @pytest.mark.parametrize(
        "dir_path, numerical_logs, categorical_logs, depth_unit",
        [
            ('test_well', ['GR', 'DENS', 'RESD', 'NEUT'], ['TOP'], 'ft')
        ]
    )
    def test_las_data_adapter_getitem_slice_key(self, dir_path, numerical_logs, categorical_logs, depth_unit):
        # Arrange:
        lass_data_adapter = LASDataAdapter(dir_path=dir_path, numerical_logs=numerical_logs,
                                           categorical_logs=categorical_logs,
                                           depth_unit=depth_unit)

        # Act:
        actual_well_datum_list = lass_data_adapter[:]

        # Assert:
        assert isinstance(actual_well_datum_list[0], WellDatum)

    @pytest.mark.parametrize(
        "dir_path, numerical_logs, categorical_logs, depth_unit",
        [
            ('test_well', ['GR', 'DENS', 'RESD', 'NEUT'], ['TOP'], 'ft')
        ]
    )
    def test_las_data_adapter_getitem_key_error(self, dir_path, numerical_logs, categorical_logs, depth_unit):
        # Arrange:
        lass_data_adapter = LASDataAdapter(dir_path=dir_path, numerical_logs=numerical_logs,
                                           categorical_logs=categorical_logs,
                                           depth_unit=depth_unit)

        # Assert:
        with pytest.raises(KeyError):
            # Act:
            actual_well_datum_list = lass_data_adapter[""]
