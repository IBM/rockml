""" Copyright 2023 IBM Research. All Rights Reserved.

    - Test functions for seisfast.adapter.seismic.segy.
"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pytest
from rockml.data.adapter import FEATURE_NAME, LABEL_NAME
from rockml.data.adapter.seismic.segy.prestack import CDPGatherAdapter, CDPGatherDatum, _CDPProps as cdp_props, \
    CDPGatherDataDumper
from seisfast.io.seismic import PreStackSEGY


class TestCDPGatherDatum:
    @pytest.mark.parametrize(
        "features, label, offsets, inline, crossline, pixel_depth, coherence, velocities",
        [
            (np.ones((3000, 144)), None,
             np.ndarray([105, 140, 175]),
             980, 185, 0, np.random.uniform(low=0.0005, high=0.63, size=(200, 66)),
             [[336, 1711], [702, 1845], [945, 425]])
        ]
    )
    def test_cdp_gather_datum_init(self, features, label, offsets, inline, crossline, pixel_depth, coherence,
                                   velocities):
        # Arrange
        gather_datum = CDPGatherDatum(features, offsets, label, inline, crossline, pixel_depth, coherence, velocities)

        expected_features = features
        expected_label = label
        expected_inline = inline
        expected_crossline = crossline
        expected_pixel_depth = pixel_depth
        expected_coherence = coherence
        expected_velocities = velocities

        # Act:
        actual_features = gather_datum.features
        actual_label = gather_datum.label
        actual_inline = gather_datum.inline
        actual_crossline = gather_datum.crossline
        actual_pixel_depth = gather_datum.pixel_depth
        actual_coherence = gather_datum.coherence
        actual_velocities = gather_datum.velocities

        # Assert
        assert actual_features.shape == expected_features.shape
        assert actual_label == expected_label
        assert actual_inline == expected_inline
        assert actual_crossline == expected_crossline
        assert actual_pixel_depth == expected_pixel_depth
        assert actual_coherence.all() == expected_coherence.all()
        assert actual_velocities == expected_velocities

    @pytest.mark.parametrize(
        "features, label, offsets, inline, crossline, pixel_depth, coherence, velocities",
        [
            (np.ones((3000, 144)), None,
             np.ndarray([105, 140, 175]),
             980, 185, 0, np.random.uniform(low=0.0005, high=0.63, size=(200, 66)),
             [[336, 1711], [702, 1845], [945, 425]])
        ]
    )
    def test_cdp_gather_datum_str(self, features, label, offsets, inline, crossline, pixel_depth, coherence,
                                  velocities):
        gather_datum = CDPGatherDatum(features, offsets, label, inline, crossline, pixel_depth, coherence, velocities)

        expected_string = gather_datum.__str__()

        actual_string = f"<CDPGatherDatum ({inline},{crossline}), pixel_depth: {pixel_depth}>"

        assert actual_string == expected_string


def _get_stub_cdp_gather_datum(features, offsets, label, inline, crossline, pixel_depth, coherence, velocities):
    seismic_stub = Mock()
    seismic_stub.features = features
    seismic_stub.offsets = offsets
    seismic_stub.label = label
    seismic_stub.inline = inline
    seismic_stub.crossline = crossline
    seismic_stub.pixel_depth = pixel_depth
    seismic_stub.coherence = coherence
    seismic_stub.velocities = velocities

    return seismic_stub


class TestCDPGatherDataDumper:
    def setup_method(self):
        self.velocity_path = os.path.abspath('segy_test_prestack_data/velocity_file.txt')

    @pytest.mark.parametrize(
        "datum_settings, path",
        [
            ({FEATURE_NAME: np.ones((3000, 144)),
              LABEL_NAME: None,
              cdp_props.OFFSETS.value: [105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455, 490, 525, 560,
                                        595, 630, 665, 700, 735, 770, 805, 840, 875, 910, 945, 980, 1015, 1050,
                                        1085, 1120, 1155, 1190, 1225, 1260, 1295, 1330, 1365, 1400, 1435, 1470, 1505],
              cdp_props.INLINE.value: 980, cdp_props.CROSSLINE.value: 185, cdp_props.PIXEL_DEPTH.value: 0,
              cdp_props.COHERENCE.value: np.random.uniform(low=0.0005, high=0.63, size=(200, 66)),
              cdp_props.VELOCITIES.value: [[336, 1711], [702, 1845], [945, 425]]
              }, "/some_path")
        ]
    )
    def test_to_hdf(self, datum_settings, path):
        # Arrange:
        fake_seismic_datum = _get_stub_cdp_gather_datum(**datum_settings)

        # Act:
        with patch('h5py.File') as h5:
            h5f = Mock()
            h5.return_value = h5f
            h5f.create_dataset.return_value = None
            CDPGatherDataDumper.to_hdf([fake_seismic_datum], path)

        # Assert:
        h5.assert_called_once_with(path, 'w')
        assert h5f.create_dataset.call_count == 8

    @pytest.mark.parametrize(
        "inline, crossline, velocity",
        [
            (980, 185, [[336, 1711], [702, 1845], [945, 425]])
        ]
    )
    def test_write_velocity_function(self, inline, crossline, velocity):
        space = "     "
        with open(self.velocity_path, 'w+') as file:
            CDPGatherDataDumper.write_velocity_function(file, inline, crossline, velocity)
        file.close()

        with open(self.velocity_path, 'r+') as file:
            assert file.readline() == f"VFUNC {space}{inline}{space}{crossline}\n"


class TestCDPGatherAdapter:
    def setup_method(self):
        self.segy_prestack_path = os.path.abspath('segy_test_prestack_data/prestack.sgy')
        self.velocity_data_file = os.path.abspath('segy_test_prestack_data/velocity')

    @pytest.mark.parametrize(
        "segy_path, gather_list, velocity_file_path",
        [
            ("path/to/segy.sgy",
             [[980, 185], [980, 225], [980, 265], [980, 305], [980, 345]],
             None),
        ]
    )
    def test_cdp_gather_adapter_init(self, segy_path, gather_list, velocity_file_path):
        # Arrange:
        segy = CDPGatherAdapter(segy_path=segy_path, gather_list=gather_list, velocity_file_path=velocity_file_path)

        # Act:
        actual_segy_path = segy.segy_path
        actual_segy_raw_data = segy._segy_raw_data
        actual_segy_info = segy.segy_info

        # Assert:
        assert actual_segy_path == segy_path
        assert actual_segy_raw_data is None
        assert actual_segy_info is None

    @pytest.mark.parametrize(
        "segy_path, gather_list, velocity_file_path, inline_byte, crossline_byte, "
        "x_byte, y_byte, source_byte, recx_byte, recy_byte",
        [
            ("path/to/segy.sgy",
             None,
             None, np.uint8(189), np.uint8(193), np.uint8(181), np.uint8(185),
             np.uint8(17), np.uint8(81), np.uint8(85)),
        ]
    )
    def test_cdp_gather_cdp_list_none(self, segy_path, gather_list, velocity_file_path, inline_byte, crossline_byte,
                                      x_byte, y_byte, source_byte, recx_byte, recy_byte):
        # Arrange:
        segy = CDPGatherAdapter(segy_path=segy_path, gather_list=gather_list, velocity_file_path=velocity_file_path,
                                inline_byte=inline_byte, crossline_byte=crossline_byte, x_byte=x_byte, y_byte=y_byte,
                                source_byte=source_byte, recx_byte=recx_byte, recy_byte=recy_byte)

        # Act:
        actual_cdp_gather_list = segy.gather_list

        # Assert:
        assert actual_cdp_gather_list == list()

    @pytest.mark.parametrize(
        "gather_list, velocity_file_path, expected_prestack_segy_info",
        [
            ([[980, 185], [980, 225], [980, 265], [980, 305], [980, 345]],
             None,
             {'range_inlines': [980, 980], 'range_xlines': [185, 545], 'num_ilines': 1, 'num_xlines': 10,
              'range_time_depth': [0, 5998.0], 'num_time_depth': 3000}
             )
        ]
    )
    def test_cdp_gather_adapter_segy_raw_data(self, gather_list, velocity_file_path, expected_prestack_segy_info):
        # Arrange:
        segy = CDPGatherAdapter(segy_path=self.segy_prestack_path, gather_list=gather_list, velocity_file_path=None)

        # Act:

        actual_segy_raw_data = segy.segy_raw_data
        actual_segy_info = segy.segy_info

        # Assert:
        assert actual_segy_info['range_inlines'] == expected_prestack_segy_info['range_inlines']

        assert isinstance(actual_segy_raw_data, PreStackSEGY)

    @pytest.mark.parametrize(
        "gather_list, velocity_file_path, expected_prestack_segy_info",
        [
            ([[980, 185], [980, 225], [980, 265], [980, 305], [980, 345]],
             None,
             {'range_inlines': [980, 980], 'range_xlines': [185, 545], 'num_ilines': 1, 'num_xlines': 10,
              'range_time_depth': [0, 5998.0], 'num_time_depth': 3000}
             )
        ]
    )
    def test_cdp_gather_adapter_initial_scan(self, gather_list, velocity_file_path, expected_prestack_segy_info):
        # Arrange:
        segy = CDPGatherAdapter(segy_path=self.segy_prestack_path, gather_list=gather_list, velocity_file_path=None)

        # Act:
        actual_segy_info = segy.initial_scan()

        # Assert:
        assert actual_segy_info['range_inlines'] == expected_prestack_segy_info['range_inlines']

    @pytest.mark.parametrize(
        "gather_list, velocity_file_path",
        [
            ([[980, 185], [980, 225], [980, 265], [980, 305], [980, 345]],
             None
             )
        ]
    )
    def test_cdp_gather_datum_instance(self, gather_list, velocity_file_path):
        # Arrange:
        segy = CDPGatherAdapter(segy_path=self.segy_prestack_path, gather_list=gather_list,
                                velocity_file_path=velocity_file_path)

        # Act:
        actual_datum = next(iter(segy))

        # Assert:
        assert isinstance(actual_datum, CDPGatherDatum)

    @pytest.mark.parametrize(
        "gather_list, velocity_file_path",
        [
            ([[980, 185], [980, 225], [980, 265], [980, 305], [980, 345]],
             None
             )
        ]
    )
    def test_cdp_gather_adapter2d_getitem_key_error(self, gather_list, velocity_file_path):
        # Arrange:
        segy = CDPGatherAdapter(segy_path=self.segy_prestack_path, gather_list=gather_list,
                                velocity_file_path=velocity_file_path)
        # Assert:
        with pytest.raises(KeyError):
            # Act:
            actual_datum_list = segy[""]
