""" Copyright 2023 IBM Research. All Rights Reserved.
"""

import os
from typing import Union, List, Tuple

import numpy as np
import pytest
from rockml.data.adapter.seismic.segy.poststack import PostStackAdapter2D
from rockml.data.transformations import Composer
from rockml.data.transformations.seismic import image, horizon
from rockml.data.transformations.seismic.filters import MinimumTextureFilter
from seisfast.io import PostStackSEGY


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
        "segy_header",
        [
            ([189, 193, 181, 185])
        ]
    )
    def test_analyse(self, segy_header: List[np.uint8]):
        # def time_depth_to_pixel(segy_info: dict, time_depth: float) -> int:
        #     td_res = (segy_info["range_time_depth"][1] - segy_info["range_time_depth"][0]) / (
        #             segy_info["num_time_depth"] - 1)
        #
        #     return int(round((time_depth - segy_info["range_time_depth"][0]) / td_res))
        #
        # def order_horizons(segy_info: dict, horizons_paths: List[str]) -> Tuple[List, List]:
        #     from seisfast.io import Reader
        #     horizons = [None] * len(horizons_paths)
        #     ranges = [None] * len(horizons_paths)
        #
        #     for idx, horizon in enumerate(horizons_paths):
        #         horizons[idx] = Reader(horizon)
        #         range_temp = (time_depth_to_pixel(segy_info, horizons[idx].get_values()[:, -1].min()),
        #                       time_depth_to_pixel(segy_info, horizons[idx].get_values()[:, -1].max()))
        #         ranges[idx] = [horizons_paths[idx], range_temp]
        #
        #     horizons, ranges = (list(t) for t in zip(*sorted(zip(horizons, ranges), key=lambda pair: pair[1][1][0])))
        #
        #     ranges = [{'path': r[0], 'range': r[1]} for r in ranges]
        #
        #     return horizons, ranges

        segy = PostStackSEGY(self.segy_path)
        mapping = segy.trace_header_mapping()
        segy_info = segy.scan(save_scan=True,
                              finline=mapping[segy_header[0]],
                              fcrossline=mapping[segy_header[1]],
                              fx=mapping[segy_header[2]],
                              fy=mapping[segy_header[3]])
        # horizons, horizons_ranges = order_horizons(segy_info, [d['path'] for d in self.horizon_path_list])

        assert segy_info['range_inlines'] == [500, 550]
        assert segy_info['range_crosslines'] == [550, 650]
        assert segy_info['range_time_depth'] == [4, 1848.0]

    @pytest.mark.parametrize(
        "train_slices, tile_shape, strides,"
        "crop, gray_levels, percentile, valid_ratio",
        [
            ({'inline': [[500, 550]]},
             (50, 50),
             (5, 5),
             [0, 0, 75, 0],
             256,
             1.0,
             0.0)
        ]
    )
    def test_seg_pipeline(self,
                          train_slices: dict,
                          tile_shape: Tuple[int, int],
                          strides: Tuple[int, int],
                          gray_levels: int,
                          crop: Union[Tuple[int, int, int, int], List[int]],
                          percentile: float,
                          valid_ratio: float, ):
        horizons_path_list = [d['path'] for d in self.horizon_path_list]

        adapter = PostStackAdapter2D(segy_path=self.segy_path,
                                     horizons_path_list=horizons_path_list,
                                     data_dict=train_slices)
        segy_info = adapter.initial_scan()

        pre_proc = Composer(
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
                    stride_shape=strides,
                    auto_pad=True
                ),
                MinimumTextureFilter(min_texture_in_features=0.9),
                image.ScaleByteFeatures()
            ]
        )
        pos_proc = Composer(
            transformations=[
                image.ReconstructFromWindows(
                    inline_shape=(
                        segy_info['num_time_depth'],
                        segy_info['range_crosslines'][1] - segy_info['range_crosslines'][0] + 1
                    ),
                    crossline_shape=(
                        segy_info['num_time_depth'],
                        segy_info['range_inlines'][1] - segy_info['range_inlines'][0] + 1
                    ),
                    strides=tile_shape
                ),
                horizon.ConvertHorizon(
                    horizon_names=[
                        'hrz_1',
                        'hrz_2',
                        'hrz_3',
                        'hrz_4'
                    ],
                    crop_left=crop[0],
                    crop_top=crop[2],
                    inline_resolution=adapter.segy_raw_data.get_inline_resolution(),
                    crossline_resolution=adapter.segy_raw_data.get_crossline_resolution()
                )
            ]
        )
        dataset = pre_proc.apply(adapter[0:2])
        dataset = pos_proc.apply([dataset])
        dataset = horizon.ConcatenateHorizon()(dataset)

        # Number of classes (we have  3 horizons, so 4 classes
        assert len(dataset) == 4
        # PostStackDataDumper.to_tfrecords(datum_list=train_tiles,
        #                                  path=os.path.join('/Users/sallesd/Projects/f3_db', f'train.tfr'))
        #
        # report_dict = {'segy_info_file': '',
        #                'data_type': 'segmentation',
        #                'output_path': '',
        #                'train_slices': train_slices,
        #                'test_slices': 0,
        #                'tile_shape': list(train_tiles[0].features.shape),
        #                'strides': list(strides),
        #                'gray_levels': gray_levels,
        #                'crop': list(crop),
        #                'percentile': percentile,
        #                'valid_ratio': valid_ratio,
        #                'limit_train': None,
        #                'cores': 1,
        #                'num_blocks': 1,
        #                'log_level': 'INFO',
        #                'num_classes': len(np.unique([t.label for t in train_tiles])),
        #                'train_records': len(train_tiles),
        #                'valid_records': len(valid_tiles),
        #                'test_records': 0,
        #                'runtime_seconds': 0}
        # with open(os.path.join('/Users/sallesd/Projects/f3_db', 'dataset_log.yml'), 'w') as f:
        #     safe_dump(report_dict, f)
