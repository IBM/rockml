""" Copyright 2023 IBM Research. All Rights Reserved.

    - Well adapter base classes.
"""

from enum import Enum
from typing import List

import pandas as pd
from rockml.data.adapter import Datum, DataDumper


class _WellProps(Enum):
    """Define well exclusive properties names. """

    WELL_NAME = 'WELL_NAME'
    DEPTH_NAME = 'DEPT'
    X_NAME = 'X'
    Y_NAME = 'Y'


class WellDatum(Datum):
    """ Establish the structure of a datum for well seisfast, which includes features, label and info. """

    # Well exclusive names
    names = _WellProps

    def __init__(self, df: pd.DataFrame, well_name: str, numerical_logs: List[str], categorical_logs: List[str],
                 coords: bool = False):
        """ Initialize WellDatum class.

        Args:
            df: pd.DataFrame containing seisfast from a single well.
            well_name: (str) name of the well.
            numerical_logs: list of numerical logs to include in final dataframe.
            categorical_logs: list of categorical logs to include in final dataframe.
            coords: (bool) True if well has coords (x, y).
        """

        self.df = df
        self.numerical_logs = list(numerical_logs)
        self.categorical_logs = list(categorical_logs)
        self.coords = coords
        self.well_name = well_name

    def __str__(self):
        has_coords = ", has coords" if self.coords else ''
        return (f"<WellDatum {self.well_name}, numerical_logs: {self.numerical_logs}, "
                f"categorical_logs: {self.categorical_logs}{has_coords}>")


class WellDataDumper(DataDumper):
    """ Class for dumping wells dataframe list to disk. """

    def _update_df(self, df: pd.DataFrame, well_name: str) -> pd.DataFrame:
        """ Add a new column to *df* containing the *well_name*.

        Args:
            df: pandas DataFrame containing well log curves.

        Returns:
            pandas DataFrame with added well name column.
        """

        df[WellDatum.names.WELL_NAME.value] = well_name

        return df

    def concatenate(self, datum_list: List[WellDatum]) -> pd.DataFrame:
        """ Concatenate all seisfast in *datum_list* in a single df with an extra column indicating each well name.

        Args:
            datum_list: list of WellDatum containing well seisfast.

        Returns:
            concatenated pandas DataFrame, with all dfs in *datum_list*.
        """

        df_list = [self._update_df(datum.df, well_name=datum.well_name) for datum in datum_list]

        final_df = pd.concat(df_list, axis=0, ignore_index=True, sort=False)

        return final_df

    def to_pickle(self, datum_list: List[WellDatum], path: str, well_list: List[str] = None):
        """ Wrapper to save the list of wells in a pickle file.

        Args:
            datum_list: list of WellDatum containing well seisfast.
            well_list: list of wells to save; if None, all wells are saved.
            path: (str) path where to save the pickle file.
        """

        df = self.concatenate(datum_list)

        if well_list is None:
            df.to_pickle(path=path)
        else:
            df[df[WellDatum.names.WELL_NAME.value].isin(well_list)].to_pickle(path=path)

    def to_hdf(self, datum_list: List[WellDatum], path: str, well_list: List[str] = None):
        """ Wrapper to save the list of wells in a hdf5 file.

        Args:
            datum_list: list of WellDatum containing well seisfast.
            well_list: list of wells to save; if None, all wells are saved.
            path: (str) path where to save the hdf5 file.
        """

        df = self.concatenate(datum_list)

        if well_list is None:
            df.to_hdf(path, key='df', mode='w')
        else:
            df[df[WellDatum.names.WELL_NAME.value].isin(well_list)].to_hdf(path, key='df', mode='w')
