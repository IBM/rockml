""" Copyright 2023 IBM Research. All Rights Reserved. """

import os
from typing import Union, List, Tuple

import pytest
from rockml.data.adapter import DataDumper
from rockml.data.adapter import FEATURE_NAME, LABEL_NAME
from rockml.data.adapter.seismic.segy.poststack import PostStackAdapter2D, PostStackDatum
from rockml.data.sampling import RandomSampler, split_dataset
from rockml.data.transformations import Composer
from rockml.data.transformations.seismic import image
from rockml.data.transformations.seismic.filters import MinimumTextureFilter


class TestDataDumper:
    def setup_method(self):
        self.segy_path = os.path.abspath('segy_test_data/cropped_netherlands.sgy')
        self.save_path = os.path.abspath('segy_test_data/dump.tfr')
        self.horizon_path_list = [
            {'path': os.path.abspath('segy_test_data/North_Sea_Group.xyz'),
             'range': [112, 116]},
            {'path': os.path.abspath('segy_test_data/SSN_Group.xyz'),
             'range': [258, 263]},
            {'path': os.path.abspath('segy_test_data/Germanic_Group.xyz'),
             'range': [346, 358]}
        ]

    @pytest.mark.parametrize(
        "train_slices, tile_shape, strides,"
        "crop, gray_levels, percentile, valid_ratio",
        [
            ({'inline': [[500, 550]]},
             (50, 50),
             (40, 40),
             [0, 0, 75, 0],
             256,
             1.0,
             0.1)
        ]
    )
    def test_tfrecord_dumper(self,
                             train_slices: dict,
                             tile_shape: Tuple[int, int],
                             strides: Tuple[int, int],
                             gray_levels: int,
                             crop: Union[Tuple[int, int, int, int], List[int]],
                             percentile: float,
                             valid_ratio: float, ):
        def serialize(datum):
            d = dict()
            d[FEATURE_NAME] = DataDumper.bytes_feature([datum.features.flatten().tostring()])
            d[LABEL_NAME] = DataDumper.bytes_feature([datum.label.flatten().tostring()])
            d[PostStackDatum.names.DIRECTION.value] = DataDumper.bytes_feature([datum.direction.value.encode()])
            d[PostStackDatum.names.LINE_NUMBER.value] = DataDumper.int64_feature([datum.line_number])
            d[PostStackDatum.names.PIXEL_DEPTH.value] = DataDumper.int64_feature([datum.pixel_depth])
            d[PostStackDatum.names.COLUMN.value] = DataDumper.int64_feature([datum.column])
            d['features_shape'] = DataDumper.int64_feature(datum.features.shape)
            return d

        composer = Composer(
            transformations=[
                image.Crop2D(
                    crop_left=crop[0],
                    crop_right=crop[1],
                    crop_top=crop[2],
                    crop_bottom=crop[3]
                ),
                image.ScaleIntensity(
                    gray_levels=gray_levels,
                    percentile=percentile
                ),
                image.FillSegmentationMask(),
                image.ViewAsWindows(
                    tile_shape=tile_shape,
                    stride_shape=strides
                ),
                MinimumTextureFilter(min_texture_in_features=0.9)
            ]
        )
        adapter = PostStackAdapter2D(
            segy_path=self.segy_path,
            horizons_path_list=[d['path'] for d in self.horizon_path_list],
            data_dict=train_slices
        )
        segy_info = adapter.initial_scan()

        train_tiles = composer.apply(list(adapter))
        train_tiles, valid_tiles = split_dataset(train_tiles, valid_ratio=valid_ratio)
        train_tiles = RandomSampler().sample(dataset=train_tiles)

        DataDumper.to_tfrecords(train_tiles, self.save_path, serialize)
