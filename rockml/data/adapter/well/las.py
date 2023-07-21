""" Copyright 2019 IBM Research. All Rights Reserved.

    - LAS adapter class definition.
"""

import os
import re
from typing import Union, List, Tuple

import lasio
import numpy as np
import pandas as pd
from rockml.data.adapter import BaseDataAdapter
from rockml.data.adapter.well import WellDatum


def natural_key(s):
    """ Key to use with sort() in order to sort string lists in natural order.
        Example: [1_1, 1_2, 1_5, 1_10, 1_13].
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', s)]


class LASDataAdapter(BaseDataAdapter):
    """ Specific class for reading and iterating well seisfast in LAS files. A single LAS file is the iteration element;
        it is associated with a single well."""

    def __init__(self, dir_path: str, numerical_logs: List[str], categorical_logs: List[str],
                 depth_unit: str, lat_long: Tuple[str, str] = None):
        """ Initialize LASDataAdapter class.

        Args:
            dir_path: (str) path to the SEGY file.
            numerical_logs: list of numerical logs to include in final dataframe.
            categorical_logs: list of categorical logs to include in final dataframe.
            depth_unit: (str) 'm' or 'ft'.
            lat_long: tuple indicating the names of the latitude and longitude fields in the LAS files; if None, no
                information about latitude or longitude will be included in the final dataframe.
        """

        super(LASDataAdapter, self).__init__()

        self.numerical_logs = numerical_logs
        self.categorical_logs = categorical_logs

        self.dir_path = dir_path
        self.las_paths = None
        self.depth_unit = depth_unit

        self.lat_long = lat_long

        self._get_las_files()

    def _get_las_files(self):
        """ Get list of las files in *self.dir_path*. """

        self.las_paths = [os.path.join(self.dir_path, f) for f in os.listdir(self.dir_path)
                          if f.lower().endswith('.las')]

        self.las_paths.sort(key=natural_key)

    def _load_las(self, file_path: str) -> WellDatum:
        """ Load LAS file in *file_path* into a pandas DataFrame and create a WellDatum.
            The following steps are executed:

            - Remove columns not listed in numerical or categorical lists;
            - Create a depth column with the name DEPTH_NAME;
            - Cast the categorical columns to categorical type;
            - Create new columns for (x,y) coords if self.lat_long is valid;
            - Get well name from LAS or file name;
            - Create a WellDatum containing the dataframe and other well information.

        Args:
            file_path: (str) path to the LAS file to be loaded.

        Returns:
            pd.DataFrame containing the selected well log curves.
        """

        las = lasio.read(file_path, null_policy='common')
        df = las.df()
        coords = False

        # Remove unwanted columns and add missing columns
        self._fix_columns(df)

        # Get depth from index and turn into a new column
        self._index_to_depth(las, df)

        # Categorical columns to categorical type :)
        df[self.categorical_logs] = df[self.categorical_logs].apply(pd.Categorical)

        if self.lat_long is not None:
            self._latlong_to_xy(las, df)
            coords = True

        # Get well name
        well_name = self._get_well_name(las, file_path)

        datum = WellDatum(df, well_name=well_name, numerical_logs=self.numerical_logs,
                          categorical_logs=self.categorical_logs, coords=coords)

        return datum

    def _fix_columns(self, df: pd.DataFrame):
        """ Remove columns not listed in *self.numerical_logs* or *self.categorical_logs* lists. Also adds missing
            *self.numerical_logs* and *self.categorical_logs* columns. Operations are inplace.

        Args:
            df: pandas DataFrame containing well seisfast.
        """

        drop_cols = [col for col in df.columns if col not in self.numerical_logs + self.categorical_logs]
        df.drop(drop_cols, axis=1, inplace=True)
        for col in self.numerical_logs + self.categorical_logs:
            if col not in df.columns:
                df[col] = np.nan

    def _index_to_depth(self, las: lasio.LASFile, df: pd.DataFrame):
        """ Get the depth curve in *las* in the unit indicated by *self.depth_unit* and add to *df*. Also reset the
            index in *df*. Operations are inplace.

        Args:
            las: loaded LASFile.
            df: pandas DataFrame containing the well log curves.
        """

        if self.depth_unit == 'm':
            depth = las.depth_m
        elif self.depth_unit == 'ft':
            depth = las.depth_ft
        else:
            raise ValueError(f"Depth unit should be 'm' or 'ft', not '{self.depth_unit}'")

        df.reset_index(drop=True, inplace=True)
        df[WellDatum.names.DEPTH_NAME.value] = depth

    @staticmethod
    def _get_well_name(las: lasio.LASFile, file_path: str) -> str:
        """ Get well name string. If provided, the well name in *las* is used; otherwise, the well file name is used.

        Args:
            las: loaded LASFile.
            file_path: (str) path to the loaded LAS file *las*.

        Returns:
            well name string.
        """

        if 'WELL' in las.well:
            well_name = las.well.WELL.value
        else:
            well_name = os.path.splitext(os.path.basename(file_path))

        return well_name

    def _latlong_to_xy(self, las: lasio.LASFile, df: pd.DataFrame):
        """ Get the latitude and longitude information in *las*, transform to x and y values, and add them as new
            columns in *df*. Operations are inplace.

        Args:
            las: loaded LASFile.
            df: pandas DataFrame containing the well log curves.
        """

        lat_long = [las.well[self.lat_long[0]], las.well[self.lat_long[1]]]

        # TODO: calculate (x,y) columns
        x = lat_long[0].value
        y = lat_long[1].value

        df[WellDatum.names.X_NAME.value] = x
        df[WellDatum.names.Y_NAME.value] = y

    def __len__(self):
        return len(self.las_paths)

    def __iter__(self) -> WellDatum:
        for f in self.las_paths:
            datum = self._load_las(f)
            yield datum

    def __getitem__(self, key: Union[int, slice, list]) -> Union[List[WellDatum], WellDatum]:
        """ Returns a WellDatum if *key* is int and list of WellDatum if *key* is either a slice or a list.

        Args:
            key: int, slice or list.

        Returns:
            WellDatum or list of WellDatum.
        """
        if type(key) == int:
            return self._load_las(self.las_paths[key])

        elif type(key) == list:
            return [self[i] for i in key]

        elif type(key) == slice:
            start = key.start if key.start else 0
            stop = key.stop if key.stop is not None else len(self)
            step = key.step if key.step is not None else 1

            return self[list(range(start, stop, step))]

        else:
            raise KeyError("This method only supports integer, slice or list.")
