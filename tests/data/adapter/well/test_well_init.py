""" Copyright 2023 IBM Research. All Rights Reserved. """

import numpy as np
import pandas as pd
import pytest

from rockml.data.adapter.well import WellDatum


class TestWellDatum:
    @pytest.mark.parametrize(
        "df, well_name, numerical_logs, categorical_logs, coords",
        [
            (pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde')), "Any_Well_Name",
             ['GR', 'DENS', 'RESD', 'NEUT'], ['TOP'], False)
        ]
    )
    def test_well_datum_init(self, df, well_name, numerical_logs, categorical_logs, coords):
        # Arrange:
        well_datum = WellDatum(df, well_name=well_name, numerical_logs=numerical_logs,
                               categorical_logs=categorical_logs, coords=coords)

        expected_df = df
        expected_well_name = well_name
        expected_numerical_logs = numerical_logs
        expected_categorical_logs = categorical_logs
        expected_coords = coords

        # Act:
        actual_df = well_datum.df
        actual_well_name = well_datum.well_name
        actual_numerical_logs = well_datum.numerical_logs
        actual_categorical_logs = well_datum.categorical_logs
        actual_coords = well_datum.coords

        # Assert:
        assert actual_df.shape == expected_df.shape
        assert actual_well_name == expected_well_name
        assert actual_numerical_logs == expected_numerical_logs
        assert actual_categorical_logs == expected_categorical_logs
        assert actual_coords == expected_coords

    @pytest.mark.parametrize(
        "df, well_name, numerical_logs, categorical_logs, coords, expected_str",
        [
            (pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde')), "Any_Well_Name",
             ['GR', 'DENS', 'RESD', 'NEUT'], ['TOP'], False,
             "<WellDatum Any_Well_Name, numerical_logs: ['GR', 'DENS', 'RESD', 'NEUT'],"
             " categorical_logs: ['TOP']>")
        ]
    )
    def test_well_datum_str(self, df, well_name, numerical_logs, categorical_logs, coords, expected_str):
        # Arrange:
        well_datum = WellDatum(df, well_name=well_name, numerical_logs=numerical_logs,
                               categorical_logs=categorical_logs, coords=coords)

        # Act:
        actual_str = str(well_datum)

        # Assert:
        assert actual_str == expected_str
