""" Copyright 2019 IBM Research. All Rights Reserved. """

import os
from typing import Union, List, Tuple

import numpy as np
import pytest

from rockml.data.adapter import FEATURE_NAME, LABEL_NAME, DataDumper
from rockml.data.adapter.seismic.segy.poststack import PostStackAdapter2D, PostStackDatum
from rockml.data.sampling import RandomSampler, split_dataset
from rockml.data.transformations import Composer
from rockml.data.transformations.seismic import image
from rockml.data.transformations.seismic.filters import ClassificationFilter


class TestPostStackDataPipeline:
    def setup_method(self):
        self.segy_path = os.path.abspath('segy_test_data/cropped_netherlands.sgy')
        self.horizon_path_list = [
            {'path': os.path.abspath('segy_test_data/North_Sea_Group.xyz'),
             'range': [112, 116]},
            {'path': os.path.abspath('segy_test_data/SSN_Group.xyz'),
             'range': [258, 263]},
            {'path': os.path.abspath('segy_test_data/Germanic_Group.xyz'),
             'range': [346, 358]}
        ]

    @pytest.mark.parametrize(
        "train_slices, tile_shape, strides, noise,"
        "crop, gray_levels, percentile, valid_ratio",
        [
            ({'inline': [[500, 550]]},
             (50, 50),
             (25, 25),
             0.3,
             [0, 0, 75, 0],
             256,
             1.0,
             0.1)
        ]
    )
    def test_cls_pipeline(self,
                          train_slices: dict,
                          tile_shape: Union[Tuple[int, ...], List[int]],
                          strides: Union[Tuple[int, ...], List[int]],
                          noise: int,
                          gray_levels: int,
                          crop: Union[Tuple[int, int, int, int], List[int]],
                          percentile: float,
                          valid_ratio: float):
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

        horizons_path_list = [d['path'] for d in self.horizon_path_list]

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
                ClassificationFilter(noise=noise)]
        )
        adapter = PostStackAdapter2D(segy_path=self.segy_path,
                                     horizons_path_list=horizons_path_list,
                                     data_dict=train_slices)
        segy_info = adapter.initial_scan()

        train_tiles = composer.apply(list(adapter))
        # Number of classes (we have  3 horizons, so 4 classes
        assert len(np.unique([t.label for t in train_tiles])) == 4

        train_tiles, valid_tiles = split_dataset(train_tiles, valid_ratio=valid_ratio)
        train_tiles = RandomSampler().sample(dataset=train_tiles)

        # DataDumper.to_tfrecords(datum_list=train_tiles,
        #                         path=os.path.join('/home/sallesd/projects/rockml/seisfast/segy_test_data', 'train.tfr'),
        #                         serialize=serialize)
