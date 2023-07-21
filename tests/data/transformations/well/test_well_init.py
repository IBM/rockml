""" Copyright 2019 IBM Research. All Rights Reserved. """

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from rockml.data.transformations.well import Crop, Interpolation, RemoveOutliers, FillNaN, FillCategories


def _get_stub_well_datum(df, coords=False, numerical_logs=['DEPT', 'GR'], categorical_logs=[]):
    well_stub = Mock()
    well_stub.df = df
    well_stub.coords = coords
    well_stub.numerical_logs = numerical_logs
    well_stub.categorical_logs = categorical_logs

    return well_stub


class TestCrop:
    @pytest.mark.parametrize(
        "min_depth, max_depth",
        [
            (1000, 1765)
        ]
    )
    def test_crop_init(self, min_depth: float, max_depth: float):
        # Arrange:
        crop_transformation = Crop(min_depth, max_depth)
        expected_min_depth = min_depth
        expected_max_depth = max_depth

        # Act:
        actual_min_depth = crop_transformation.min_depth
        actual_max_depth = crop_transformation.max_depth

        # Assert:
        assert actual_min_depth == expected_min_depth
        assert actual_max_depth == expected_max_depth

    @pytest.mark.parametrize(
        "min_depth, max_depth, depth_series, gr_series, expected_depth_series, expected_gr_series",
        [
            (1, 5,
             pd.Series([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]),
             pd.Series([1., 2., 3., 4., 5.]), pd.Series([1., 1., 2., 2., 3.])),

            (-11, 7,
             pd.Series([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]),
             pd.Series([1., 2., 3., 4., 5., 6., 7.]), pd.Series([1., 1., 2., 2., 3., 3., 4.])),

            (3, 17,
             pd.Series([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]),
             pd.Series([3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([2., 2., 3., 3., 4., 4., 5., 5.])),

            pytest.param(5, 3,
                         pd.Series([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]),
                         pd.Series([1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]),
                         pd.Series([3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([2., 2., 3., 3., 4., 4., 5., 5.]),
                         marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_crop_call(self, min_depth: float, max_depth: float,
                       depth_series, gr_series, expected_depth_series, expected_gr_series):
        # Arrange:
        crop_transformation = Crop(min_depth, max_depth)

        actual_frame = {'DEPT': depth_series, 'GR': gr_series}
        fake_well_df = pd.DataFrame(actual_frame)

        fake_well = _get_stub_well_datum(fake_well_df)

        expected_frame = {'DEPT': expected_depth_series, 'GR': expected_gr_series}
        expected_fake_well_df = pd.DataFrame(expected_frame)

        # Act:
        actual_fake_well = crop_transformation(fake_well)

        # Assert:
        assert actual_fake_well.df.equals(expected_fake_well_df)

    @pytest.mark.parametrize(
        "min_depth, max_depth, expected_crop_str",
        [
            (100, 300, "<Crop (min, max): (100, 300)>")
        ]
    )
    def test_crop_str(self, min_depth: float, max_depth: float, expected_crop_str):
        # Arrange:
        crop_transformation = Crop(min_depth, max_depth)

        # Act:
        actual_crop_str = str(crop_transformation)

        # Assert:
        assert actual_crop_str == expected_crop_str


class TestInterpolation:
    @pytest.mark.parametrize(
        "step_size, nan_threshold",
        [
            (3, 2)
        ]
    )
    def test_interpolation_init(self, step_size: float, nan_threshold: int):
        # Arrange:
        interpolation_transformation = Interpolation(step_size, nan_threshold)
        expected_step_size = step_size
        expected_nan_threshold = nan_threshold

        # Act:
        actual_step_size = interpolation_transformation.step_size
        actual_nan_threshold = interpolation_transformation.nan_threshold

        # Assert:
        assert actual_step_size == expected_step_size
        assert actual_nan_threshold == expected_nan_threshold

    @pytest.mark.parametrize(
        "step_size, nan_threshold, depth_series, gr_series, expected_depth_series, expected_gr_series",
        [
            (2, 1,
             pd.Series([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]),
             pd.Series([1., 3., 5., 7., 9.]), pd.Series([1., 2., 3., 4., 5.])),

            (2, 1,
             pd.Series([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., np.nan, 2., 3., 3., 4., 4., 5.,
                                                                              5.]),
             pd.Series([1., 3., 5., 7., 9.]), pd.Series([1., np.nan, 3., 4., 5.])),

            (2, 2,
             pd.Series([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., np.nan, 2., 3., 3., 4., 4., 5.,
                                                                              5.]),
             pd.Series([1., 3., 5., 7., 9.]), pd.Series([1., 1.2847623278542866, 3., 4., 5.])),

            (2, 1,
             pd.Series([1., np.nan, 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., np.nan, 2., 2., 3., 3., 4., 4.,
                                                                                  5., 5.]),
             pd.Series([1., 3., 5., 7., 9.]), pd.Series([1., 2., 3., 4., 5.])),

            (None, 1,
             pd.Series([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., np.nan, 2., 2., 3., 3., 4., 4., 5.,
                                                                              5.]),
             pd.Series([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., np.nan, 2., 2., 3., 3., 4., 4., 5.,
                                                                              5.])),

            (2, 1, pd.Series([]), pd.Series([]), pd.Series([]), pd.Series([])),

            pytest.param(2, 1,
                         pd.Series([1., 2., np.nan, 4., 5., 6., 7., 8., 9., 10.]),
                         pd.Series([1., np.nan, 2., 2., 3., 3., 4., 4., 5., 5.]),
                         pd.Series([1., np.nan, 5., 7., 9.]), pd.Series([1., 2., 3., 4., 5.]),
                         marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_interpolation_call(self, step_size: float, nan_threshold: int,
                                depth_series, gr_series, expected_depth_series, expected_gr_series):
        # Arrange:
        interpolation_transformation = Interpolation(step_size, nan_threshold)

        actual_frame = {'DEPT': depth_series, 'GR': gr_series}
        fake_well_df = pd.DataFrame(actual_frame)

        fake_well = _get_stub_well_datum(fake_well_df)

        expected_depth = expected_depth_series.values.astype(np.float32)
        expected_gr = expected_gr_series.values.astype(np.float32)

        # Act:
        actual_fake_well = interpolation_transformation(fake_well)

        actual_depth = actual_fake_well.df['DEPT'].astype(np.float32)
        actual_gr = actual_fake_well.df['GR'].astype(np.float32)

        # Assert:
        if not np.equal(actual_depth, expected_depth).all():
            idx_dif = np.where(~np.equal(actual_depth, expected_depth))[0][0]
            if np.isnan(actual_depth[idx_dif]) and np.isnan(expected_depth[idx_dif]):
                assert True
            else:
                assert np.equal(actual_depth, expected_depth).all()
        else:
            assert np.equal(actual_depth, expected_depth).all()

        if not np.equal(actual_gr, expected_gr).all():
            idx_dif = np.where(~np.equal(actual_gr, expected_gr))[0][0]
            if np.isnan(actual_gr[idx_dif]) and np.isnan(expected_gr[idx_dif]):
                assert True
            else:
                assert np.equal(actual_gr, expected_gr).all()
        else:
            assert np.equal(actual_gr, expected_gr).all()

    @pytest.mark.parametrize(
        "step_size, nan_threshold, expected_interpolation_str",
        [
            (2, 1, "<Interpolation: step_size 2, nan_threshold 1>")
        ]
    )
    def test_interpolation_str(self, step_size: float, nan_threshold: int, expected_interpolation_str):
        # Arrange:
        interpolation_transformation = Interpolation(step_size, nan_threshold)

        # Act:
        actual_interpolation_str = str(interpolation_transformation)

        # Assert:
        assert actual_interpolation_str == expected_interpolation_str


class TestRemoveOutliers:
    @pytest.mark.parametrize(
        "numerical_logs, percentiles, extreme_factor",
        [
            (['DEPT', 'GR'], (25.0, 75.0), 2.0)
        ]
    )
    def test_remove_outliers_init(self, numerical_logs, percentiles, extreme_factor):
        # Arrange:
        remove_outliers_transformation = RemoveOutliers(numerical_logs, percentiles, extreme_factor)
        expected_numerical_logs = numerical_logs
        expected_percentiles = percentiles
        expected_extreme_factor = extreme_factor

        # Act:
        actual_numerical_logs = remove_outliers_transformation.numerical_logs
        actual_percentiles = remove_outliers_transformation.percentiles
        actual_extreme_factor = remove_outliers_transformation.extreme_factor

        # Assert:
        assert actual_numerical_logs == expected_numerical_logs
        assert actual_percentiles == expected_percentiles
        assert actual_extreme_factor == expected_extreme_factor

    @pytest.mark.parametrize(
        "numerical_logs, percentiles, extreme_factor,"
        "depth_series, gr_series, expected_depth_series, expected_gr_series",
        [
            (['DEPT', 'GR'], (25.0, 75.0), 2.0,
             pd.Series([1., 2., 100., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., 3., 4., 4., 5.,
                                                                                5.]),
             pd.Series([1., 2., np.nan, 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., 3., 4., 4., 5.,
                                                                                  5.])
             ),
            (['DEPT', 'GR'], (25.0, 75.0), 2.0,
             pd.Series([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., 15., 4., 4., 5.,
                                                                              5.]),
             pd.Series([1., 2., 3, 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., np.nan, 4., 4.,
                                                                             5., 5.])
             ),
            (['DEPT', 'GR'], (25.0, 75.0), 2.0,
             pd.Series([1., 2., 100., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., 15., 4., 4., 5.,
                                                                                5.]),
             pd.Series([1., 2., np.nan, 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., np.nan, 4., 4.,
                                                                                  5., 5.])
             )
        ]
    )
    def test_remove_outliers_call(self, numerical_logs, percentiles, extreme_factor,
                                  depth_series, gr_series, expected_depth_series, expected_gr_series):
        # Arrange:
        remove_outliers_transformation = RemoveOutliers(numerical_logs, percentiles, extreme_factor)

        actual_frame = {'DEPT': depth_series, 'GR': gr_series}
        fake_well_df = pd.DataFrame(actual_frame)

        fake_well = _get_stub_well_datum(fake_well_df)

        expected_frame = {'DEPT': expected_depth_series, 'GR': expected_gr_series}
        expected_fake_well_df = pd.DataFrame(expected_frame)

        # Act:
        actual_fake_well = remove_outliers_transformation(fake_well)

        # Assert:
        assert actual_fake_well.df.equals(expected_fake_well_df)

    @pytest.mark.parametrize(
        "numerical_logs, percentiles, extreme_factor, expected_remove_outlier_str",
        [
            (['DEPT', 'GR'], (25.0, 75.0), 2.0, "<RemoveOutliers: percentiles (25.0, 75.0), extreme_factor 2.0>")
        ]
    )
    def test_remove_outliers_str(self, numerical_logs, percentiles, extreme_factor, expected_remove_outlier_str):
        # Arrange:
        remove_outlier_transformation = RemoveOutliers(numerical_logs, percentiles, extreme_factor)

        # Act:
        actual_remove_outlier_str = str(remove_outlier_transformation)

        # Assert:
        assert actual_remove_outlier_str == expected_remove_outlier_str


class TestFillNaN:
    @pytest.mark.parametrize(
        "feature_logs, target",
        [
            (['DEPT'], 'GR')
        ]
    )
    def test_fill_nan_init(self, feature_logs, target):
        # Arrange:
        fill_nan_transformation = FillNaN(feature_logs, target)
        expected_feature_logs = feature_logs
        expected_target = target

        # Act:
        actual_feature_logs = fill_nan_transformation.feature_logs
        actual_target = fill_nan_transformation.target

        # Assert:
        assert actual_feature_logs == expected_feature_logs
        assert actual_target == expected_target

    @pytest.mark.parametrize(
        "feature_logs, target,"
        "depth_series, gr_series, indexes, expected_depth_series, expected_gr_series",
        [
            (['DEPT'], None,
             pd.Series([1., 2., np.nan, 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., 3., 4., 4., 5.,
                                                                                  5.]),
             pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
             pd.Series([1., 2., 2., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., 3., 4., 4., 5., 5.])
             ),
            (['DEPT', 'GR'], None,
             pd.Series([1., 2., np.nan, 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., np.nan, 4., 4.,
                                                                                  5.,
                                                                                  5.]),
             pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
             pd.Series([1., 2., 2., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., 3., 4., 4., 5., 5.])
             ),
            (['DEPT'], 'GR',
             pd.Series([np.nan, np.nan, 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., 3., 4., 4.,
                                                                                      5., 5.]),
             pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
             pd.Series([3., 3., 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 3., 3., 4., 4., 5., 5.])
             ),
            (['DEPT'], 'GR',
             pd.Series([np.nan, np.nan, 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., np.nan, 2., 2., 3., 3., 4.,
                                                                                      4., 5., 5.]),
             pd.Series([0, 2, 3, 4, 5, 6, 7, 8, 9]),
             pd.Series([3., 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 2., 2., 3., 3., 4., 4., 5., 5.])
             )
        ]
    )
    def test_fill_nan_call(self, feature_logs, target,
                           depth_series, gr_series, indexes, expected_depth_series, expected_gr_series):
        # Arrange:
        fill_nan_transformation = FillNaN(feature_logs, target)

        actual_frame = {'DEPT': depth_series, 'GR': gr_series}
        fake_well_df = pd.DataFrame(actual_frame)

        fake_well = _get_stub_well_datum(fake_well_df)

        expected_frame = {'DEPT': expected_depth_series, 'GR': expected_gr_series}
        expected_fake_well_df = pd.DataFrame(expected_frame)

        # Act:
        actual_fake_well = fill_nan_transformation(fake_well)

        # Assert:
        expected_fake_well_df.set_index(indexes, inplace=True)
        assert actual_fake_well.df.equals(expected_fake_well_df)

    @pytest.mark.parametrize(
        "feature_logs, target, expected_fill_nan_str",
        [
            (['DEPT'], 'GR', "<FillNaN: features ['DEPT'], target GR>")
        ]
    )
    def test_fill_nan_str(self, feature_logs, target, expected_fill_nan_str):
        # Arrange:
        fill_nan_transformation = FillNaN(feature_logs, target)

        # Act:
        actual_fill_nan_str = str(fill_nan_transformation)

        # Assert:
        assert actual_fill_nan_str == expected_fill_nan_str


class TestFillCategories:
    @pytest.mark.parametrize(
        "categorical_logs",
        [
            (['CAT'])
        ]
    )
    def test_fill_categories_init(self, categorical_logs):
        # Arrange:
        fill_categories_transformation = FillCategories(categorical_logs)
        expected_categorical_logs = categorical_logs

        # Act:
        actual_categorical_logs = fill_categories_transformation.categorical_logs

        # Assert:
        assert actual_categorical_logs == expected_categorical_logs

    @pytest.mark.parametrize(
        "categorical_logs, depth_series, gr_series, expected_depth_series, expected_gr_series",
        [
            (['CAT'],
             pd.Series([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., np.nan, 2., np.nan, np.nan, 3., 4.,
                                                                              np.nan, 5., np.nan]),
             pd.Series([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]), pd.Series([1., 1., 2., 2., 2., 3., 4., 4., 5., 5.]))
        ]
    )
    def test_fill_categories_call(self, categorical_logs, depth_series, gr_series, expected_depth_series,
                                  expected_gr_series):
        # Arrange:
        fill_categories_transformation = FillCategories(categorical_logs)

        actual_frame = {'DEPT': depth_series, 'CAT': gr_series}
        fake_well_df = pd.DataFrame(actual_frame)

        fake_well = _get_stub_well_datum(fake_well_df)

        expected_frame = {'DEPT': expected_depth_series, 'CAT': expected_gr_series}
        expected_fake_well_df = pd.DataFrame(expected_frame)

        # Act:
        actual_fake_well = fill_categories_transformation(fake_well)

        # Assert:
        assert actual_fake_well.df.equals(expected_fake_well_df)

    @pytest.mark.parametrize(
        "categorical_logs, expected_crop_str",
        [
            (['CAT'], "<FillCategories: categorical logs ['CAT']>")
        ]
    )
    def test_fill_categories_str(self, categorical_logs, expected_crop_str):
        # Arrange:
        fill_categories_transformation = FillCategories(categorical_logs)

        # Act:
        actual_fill_categories_str = str(fill_categories_transformation)

        # Assert:
        assert actual_fill_categories_str == expected_crop_str
