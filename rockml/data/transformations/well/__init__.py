""" Copyright 2023 IBM Research. All Rights Reserved.

    - Defines well Transformation classes.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from rockml.data.adapter.well import WellDatum
from rockml.data.df_ops import interpolate_numeric_series, interpolate_categorical_series
from rockml.data.transformations import Transformation


class Crop(Transformation):

    def __init__(self, min_depth: float, max_depth: float):
        """ Initialize Crop class.

        Args:
            min_depth: (float) minimum depth to be considered; lower values will be cropped.
            max_depth: (float) maximum depth to be considered; higher values will be cropped.
        """

        self.min_depth = min_depth
        self.max_depth = max_depth

    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Apply crop on *df* (inplace). Fix the index afterwards.

        Args:
            df: pandas DataFrame containing well seisfast.

        Returns:
            cropped DataFrame with corrected index.
        """

        idx = df[(df[WellDatum.names.DEPTH_NAME.value] < self.min_depth) |
                 (df[WellDatum.names.DEPTH_NAME.value] > self.max_depth)].index
        df.drop(idx, axis=0, inplace=True)

        df.reset_index(drop=True, inplace=True)

        return df

    def __call__(self, datum: WellDatum) -> WellDatum:
        """ Apply crop on *datum*.df (inplace). Fix the index afterwards.

        Args:
            datum: WellDatum containing well seisfast.

        Returns:
            cropped datum.
        """

        datum.df = self._update_df(datum.df)

        return datum

    def __str__(self):
        return f"<Crop (min, max): {self.min_depth, self.max_depth}>"


class Interpolation(Transformation):

    def __init__(self, step_size: float, nan_threshold: int = 1):
        """ Initialize Interpolation class.

        Args:
            step_size: (float) step size of the resulting DataFrame based on *depth_column*.
            nan_threshold: (int) maximum number of consecutive NaN values that will be considered for interpolation;
                bigger sequences will be ignored in interpolation.
        """

        self.step_size = step_size
        self.nan_threshold = nan_threshold

    def _update_df(self, df: pd.DataFrame, numerical_logs: List[str], categorical_logs: List[str]) -> pd.DataFrame:
        """ Interpolates the *df* given the *self.step_size*. This method generates a new DataFrame instead of
            changing the *df* inplace.

        Args:
            df: pandas DataFrame to be interpolated.
            numerical_logs: list of column names that contain numeric information to be interpolated.
            categorical_logs: list of column names that contain categorical information to be interpolated.

        Returns:
            new df with interpolated seisfast.
        """

        df = df.sort_values(by=WellDatum.names.DEPTH_NAME.value)
        depth = df[WellDatum.names.DEPTH_NAME.value]

        if df.shape[0] < 1:
            return df

        # Defines new interpolation points, i.e., new depth curve
        if self.step_size is None:
            new_depth = depth
        else:
            new_depth = pd.Series(np.arange(min(depth), max(depth), self.step_size))

        # Interpolates all numeric columns
        num_dict = {name: interpolate_numeric_series(values, depth, new_depth, nan_threshold=self.nan_threshold)
                    for name, values in df.items()
                    if name in numerical_logs}

        # Interpolates all categorical columns
        cat_dict = {name: interpolate_categorical_series(values, depth_old=depth, depth_new=new_depth)
                    for name, values in df.items()
                    if name in categorical_logs}

        df_new = pd.DataFrame({**num_dict, **cat_dict})

        df_new[WellDatum.names.DEPTH_NAME.value] = new_depth

        return df_new

    def __call__(self, datum: WellDatum) -> WellDatum:
        """ Interpolates the *datum*.df given the *self.step_size*. The *datum* is modified inplace.

        Args:
            datum: WellDatum containing well seisfast.

        Returns:
            interpolated datum.
        """

        if datum.coords:
            numerical_logs = datum.numerical_logs + [WellDatum.names.X_NAME.value, WellDatum.names.Y_NAME.value]
        else:
            numerical_logs = datum.numerical_logs

        datum.df = self._update_df(datum.df, numerical_logs=numerical_logs, categorical_logs=datum.categorical_logs)

        return datum

    def __str__(self):
        return f"<Interpolation: step_size {self.step_size}, nan_threshold {self.nan_threshold}>"


class RemoveOutliers(Transformation):

    def __init__(self, numerical_logs: List[str], percentiles: Tuple[float, float] = (25.0, 75.0),
                 extreme_factor: float = 2.0):
        """ Initialize RemoveOutliers class.

        Args:
            numerical_logs: list of log names for which the outliers will be removed.
            percentiles: tuple of floats containing the lower and upper percentiles;
                0.0 < percentiles[0] < percentiles[1] < 100.0
            extreme_factor: (float) the factor to multiply the value range to calculate the extremes.
        """

        self.numerical_logs = numerical_logs
        self.percentiles = percentiles
        self.extreme_factor = extreme_factor

    def _extremes_to_nan(self, array: np.ndarray) -> np.ndarray:
        """ Replace extreme outliers in *array* with nan values. A value is considered an outlier if it falls outside
            the valid range, which is calculated as:

                    valid_range = upper_percentile_bound - lower_percentile_bound

                    upper_extreme = upper_percentile + extreme_factor * valid_range
                    lower_extreme = lower_percentile - extreme_factor * valid_range

        Args:
            array: np.ndarray of values.

        Returns:
            modified array with extremes as np.nan
        """

        lower_bound, upper_bound = np.nanpercentile(array, self.percentiles)

        value_range = upper_bound - lower_bound

        cut_up = upper_bound + self.extreme_factor * value_range
        cut_low = lower_bound - self.extreme_factor * value_range

        # This indexing is necessary so nan values will not throw a runtime error
        idx = np.greater(array, cut_up, where=~np.isnan(array))
        array[idx] = np.nan

        idx = np.less(array, cut_low, where=~np.isnan(array))
        array[idx] = np.nan

        return array

    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Remove outliers from *df* (inplace) using the method self.extremes_to_nan(). Each selected column is treated
            separately.

        Args:
            df: pandas DataFrame containing well seisfast.

        Returns:
            cropped DataFrame with corrected index.
        """

        for col in self.numerical_logs:
            df[col] = self._extremes_to_nan(df[col].values)

        return df

    def __call__(self, datum: WellDatum) -> WellDatum:
        """ Remove outliers from *datum*.df (inplace).

        Args:
            datum: WellDatum containing well seisfast.

        Returns:
            modified datum.
        """

        datum.df = self._update_df(datum.df)

        return datum

    def __str__(self):
        return f"<RemoveOutliers: percentiles {self.percentiles}, extreme_factor {self.extreme_factor}>"


class FillNaN(Transformation):

    def __init__(self, feature_logs: List[str], target: str):
        """ Initialize FillNaN class.

        Args:
            feature_logs: list of log names for which NaN values will be filled.
            target: the name of the *target* variable; the NaN rows in this column will be removed instead of filled.
        """

        self.feature_logs = feature_logs
        self.target = target

    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Fill NaN values in *df* using the following method:

            - Drop every line that has missing value in the *target* column;
            - Fill missing values using the closest value upwards, if not found, it uses the closest value downwards.

        Args:
            df: a pandas DataFrame with every name in *self.feature_logs* and the *self.target* as one of its columns.

        Returns:
            a pandas DataFrame with clean seisfast.
        """

        if self.target in df:
            df.dropna(subset=[self.target], inplace=True)

        df[self.feature_logs] = df[self.feature_logs].fillna(method='ffill')
        df[self.feature_logs] = df[self.feature_logs].fillna(method='bfill')

        return df

    def __call__(self, datum: WellDatum) -> WellDatum:
        """ Fill NaN values on *datum*.df (inplace). If provided, the *self.target* log will have its NaN rows removed.

        Args:
            datum: WellDatum containing well seisfast.

        Returns:
            clean datum.
        """

        datum.df = self._update_df(datum.df)

        return datum

    def __str__(self):
        return f"<FillNaN: features {self.feature_logs}, target {self.target}>"


class FillCategories(Transformation):

    def __init__(self, categorical_logs: List[str]):
        """ Initialize FillCategories class.

        Args:
            categorical_logs: list of log names that will be processed.
        """

        self.categorical_logs = categorical_logs

    def _update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Modify columns in *df* defined as boundaries of classes to filled regions.

            Example: top information is defined by numbers which exists only in the depth value where one finds a top;
            the regions between tops have NaN values.

            This function fills the NaN region below a top with the top value, until it finds another top, like this:

            Before     After

            TOP        TOP

            NaN        NaN
            1          1
            NaN        1
            NaN        1
            2          2
            NaN        2


        Args:
            df: a pandas DataFrame with every name in *self.categorical_logs* as one of its columns.

        Returns:
            a pandas DataFrame with processed seisfast.
        """

        df[self.categorical_logs] = df[self.categorical_logs].fillna(method='ffill')

        return df

    def __call__(self, datum: WellDatum) -> WellDatum:
        """ Modify columns *self.categorical_logs* in *datum*.df (inplace) filling NaN values with last valid
            observation forward to next valid observation. This Transformation is intended to expand top categories
            depth locations to filled regions. See example in *self._update_df*.

        Args:
            datum: WellDatum containing well seisfast.

        Returns:
            processed datum.
        """

        datum.df = self._update_df(datum.df)

        return datum

    def __str__(self):
        return f"<FillCategories: categorical logs {self.categorical_logs}>"
