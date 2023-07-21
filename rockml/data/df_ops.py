""" Copyright 2023 IBM Research. All Rights Reserved.

    - Functions to modify seisfast that operates on pandas DataFrames or Series.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev, interp1d


def _get_interpolation_idx(has_value: pd.Series, nan_threshold: int, depth_old: pd.Series,
                           depth_new: pd.Series) -> np.ndarray:
    """ Auxiliary function to get the indices where interpolation will be applied. Only the NaN sequences shorter than
        *nan_threshold* will be interpolated.
    
    Args:
        has_value: pd.Series representing a mask which indicates where the series to be interpolated has values.
        nan_threshold: (int) maximum number of consecutive NaN values that will be considered for interpolation; bigger
            sequences will be ignored in interpolation.
        depth_old: (pd.Series) original depth index.
        depth_new: (pd.Series) new depth index.

    Returns:
        np.ndarray with indices.
    """

    mask = np.ones(shape=(len(depth_new)), dtype=bool)

    nan_idx = np.nonzero(~has_value)[0]
    if len(nan_idx) == 0:
        return mask

    idx_list = np.split(nan_idx, np.where(np.diff(nan_idx) != 1)[0] + 1)

    # Get list of contiguous nan values
    nan_idx_list = [a for a in idx_list if len(a) >= nan_threshold]

    # Translate idx from depth_old to depth_new values...
    depth_ranges = [[depth_old.values[array.min()], depth_old.values[array.max()]] for array in nan_idx_list]

    new_idx_list = [np.where(np.logical_and(depth_new.values >= r[0],
                                            depth_new.values <= r[1]))[0] for r in depth_ranges]

    if len(new_idx_list) > 0:
        new_idx_list = np.concatenate(new_idx_list)

    mask[new_idx_list] = False

    return mask


def interpolate_numeric_series(series: pd.Series, depth_old: pd.Series, depth_new: pd.Series, s: float = None,
                               k: int = 2, nan_threshold: int = 1) -> pd.Series:
    """ Interpolates *series* using the B-spline representation of 1-D curve. Only NaN sequences smaller than
        *nan_threshold* will be interpolated

    Args:
        series: pd.Series with the numerical seisfast to be interpolated.
        depth_old: (pd.Series) original depth index.
        depth_new: (pd.Series) new depth index.
        s: (float) smoothness.
        k: (int) the degree of the spline fit. It is recommended to use cubic splines. Even values of k should be
            avoided especially with small s values. 1 <= k <= 5.
        nan_threshold: (int) maximum number of consecutive NaN values that will be considered for interpolation; bigger
            sequences will be ignored in interpolation.

    Returns:
        new pandas Series with the interpolated seisfast.
    """

    if all(series.isnull()):
        return pd.Series(np.full(depth_new.shape, np.nan))

    has_value = ~series.isna()
    spl = splrep(depth_old[has_value], series[has_value], s=s, k=k)

    out = np.full((len(depth_new),), np.nan)

    idx = _get_interpolation_idx(has_value, nan_threshold, depth_old, depth_new)

    out[idx] = splev(depth_new[idx], spl)

    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(depth_old.values[:], series.values[:])
    # ax[1].plot(depth_new.values[:], out[:])
    # fig.suptitle(series.name, fontsize=16)
    # plt.show()

    return pd.Series(out)


def interpolate_categorical_series(series: pd.Series, depth_old: pd.Series, depth_new: pd.Series) -> pd.Series:
    """ Interpolates *series* using the NNearest-neighbour interpolation method.

    Args:
        series: pandas Series with the categorical seisfast to be interpolated.
        depth_old: (pd.Series) original depth index.
        depth_new: (pd.Series) new depth index.

    Returns:
        new pandas Series with the interpolated seisfast.
    """

    if all(series.isnull()):
        return series

    categories = series.cat.categories
    f = interp1d(depth_old, series.cat.codes, kind='nearest')

    return pd.Categorical.from_codes(f(depth_new).astype('int'), categories)
