""" Copyright 2023 IBM Research. All Rights Reserved.

    - Test functions for seisfast.transformations.seismic.gather.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from rockml.data.adapter import FEATURE_NAME, LABEL_NAME
from rockml.data.adapter.seismic.segy.prestack import _CDPProps as props
from rockml.data.transformations.seismic import gather as gat


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


class TestComputeCoherence:
    @pytest.mark.parametrize(
        "coherence_settings",
        [
            ({"time_gate_ms": 30, "time_range_ms": [0, 5998.0], "sample_rate_ms": 2.0, "velocity_range": [1250, 4500],
              "velocity_step": 50}),
        ]
    )
    def test_compute_coherence_init(self, coherence_settings):
        # Arrange
        coherence = gat.ComputeCoherence(**coherence_settings)

        # Act
        time_gate_ms = coherence.time_gate_ms
        time_range_ms = coherence.time_range_ms
        sample_rate_ms = coherence.sample_rate_ms
        velocity_range = coherence.velocity_range
        velocity_step = coherence.velocity_step

        # Assert
        assert time_range_ms == coherence_settings["time_range_ms"]
        assert time_gate_ms == coherence_settings["time_gate_ms"]
        assert sample_rate_ms == coherence_settings["sample_rate_ms"]
        assert velocity_step == coherence_settings["velocity_step"]
        assert velocity_range == coherence_settings["velocity_range"]

    @pytest.mark.parametrize(
        "datum_settings, coherence_settings",
        [
            ({FEATURE_NAME: np.ones((3000, 144)),
              LABEL_NAME: None,
              props.OFFSETS.value: [105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455, 490, 525, 560,
                                    595, 630, 665, 700, 735, 770, 805, 840, 875, 910, 945, 980, 1015, 1050,
                                    1085, 1120, 1155, 1190, 1225, 1260, 1295, 1330, 1365, 1400, 1435, 1470, 1505, 1540,
                                    1575, 1610, 1645, 1680, 1715, 1750, 1785, 1820, 1855, 1890, 1925, 1960, 1995, 2030,
                                    2065, 2100, 2135, 2170, 2205, 2240, 2275, 2310, 2345, 2380, 2415, 2450, 2485, 2520,
                                    2555, 2590, 2625, 2660, 2695, 2730, 2765, 2800, 2835, 2870, 2905, 2940, 2975, 3010,
                                    3045, 3080, 3115, 3150, 3185, 3220, 3255, 3290, 3325, 3360, 3395, 3430, 3465, 3500,
                                    3535, 3570, 3605, 3640, 3675, 3710, 3745, 3780, 3815, 3850, 3885, 3920, 3955, 3990,
                                    4025, 4060, 4095, 4130, 4165, 4200, 4235, 4270, 4305, 4340, 4375, 4410, 4445, 4480,
                                    4515, 4550, 4585, 4620, 4655, 4690, 4725, 4760, 4795, 4830, 4865, 4900, 4935, 4970,
                                    5005, 5040, 5075, 5110],
              props.INLINE.value: 980, props.CROSSLINE.value: 185, props.PIXEL_DEPTH.value: 0,
              props.COHERENCE.value: None,
              props.VELOCITIES.value: None
              },
             {"time_gate_ms": 30, "time_range_ms": [0, 5998.0], "sample_rate_ms": 2.0, "velocity_range": [1250, 4500],
              "velocity_step": 50})
        ]
    )
    def test_compute_coherence_call(self, coherence_settings, datum_settings):
        # Arrange
        coherence = gat.ComputeCoherence(**coherence_settings)
        fake_datum = _get_stub_cdp_gather_datum(**datum_settings)

        # Act
        actual_datum = coherence(fake_datum)

        # Assert
        assert actual_datum.coherence is not None
        assert actual_datum.coherence.shape == (200, 66)

    @pytest.mark.parametrize(
        "time_gate_ms, time_range_ms, sample_rate_ms, velocity_range, velocity_step",
        [
            (0, [0, 5998.0], 0.0, [1250, 4500], 0)
        ]
    )
    def test_compute_coherence_str(self, time_gate_ms, time_range_ms, sample_rate_ms, velocity_range, velocity_step):
        # Arrange:
        coherence = gat.ComputeCoherence(time_gate_ms, time_range_ms, sample_rate_ms, velocity_range, velocity_step)
        expected_output_string = f"<ComputeCoherence time gate: {time_gate_ms}, time_range: {time_range_ms}, " \
                                 f"sample rate: {sample_rate_ms}, velocity range: {velocity_range}, " \
                                 f"velocity step: {velocity_step}>"

        # Act:
        actual_output_string = coherence.__str__()

        # Assert:
        assert actual_output_string == expected_output_string


class TestComputeVelocityFunction:
    @pytest.mark.parametrize(
        "velocity_settings",
        [
            ({"time_gate_ms": 30, "time_range_ms": [0, 5998.0], "sample_rate_ms": 2.0, "velocity_range": [1250, 4500],
              "velocity_step": 50, "handle_sample_zero": True, "initial_velocity": None,
              "initial_analysis_time_ms": None, "spl_order": 1, "spl_smooth": 0.1, "spl_ext": 0,
              "savgol_window": 51, "savgol_order": 1}),
        ]
    )
    def test_compute_velocity_function_init(self, velocity_settings):
        # Arrange
        velocity = gat.ComputeVelocityFunction(**velocity_settings)

        # Act
        time_gate_ms = velocity.time_gate_ms
        time_range_ms = velocity.time_range_ms
        sample_rate_ms = velocity.sample_rate_ms
        velocity_range = velocity.velocity_range
        velocity_step = velocity.velocity_step
        handle_sample_zero = velocity.handle_sample_zero
        initial_velocity = velocity.initial_velocity
        initial_analysis_time_ms = velocity.initial_analysis_time
        spl_order = velocity.spl_order
        spl_smooth = velocity.spl_smooth
        spl_ext = velocity.spl_ext
        savgol_window = velocity.savgol_window
        savgol_order = velocity.savgol_order

        # Assert
        assert time_range_ms == velocity_settings["time_range_ms"]
        assert time_gate_ms == velocity_settings["time_gate_ms"]
        assert sample_rate_ms == velocity_settings["sample_rate_ms"]
        assert velocity_step == velocity_settings["velocity_step"]
        assert velocity_range == velocity_settings["velocity_range"]
        assert handle_sample_zero == velocity_settings["handle_sample_zero"]
        assert initial_velocity == velocity_settings["initial_velocity"]
        assert initial_analysis_time_ms == velocity_settings["initial_analysis_time_ms"]
        assert spl_order == velocity_settings["spl_order"]
        assert spl_smooth == velocity_settings["spl_smooth"]
        assert spl_ext == velocity_settings["spl_ext"]
        assert savgol_window == velocity_settings["savgol_window"]
        assert savgol_order == velocity_settings["savgol_order"]

    @pytest.mark.parametrize(
        "datum_settings, velocity_settings",
        [
            ({FEATURE_NAME: np.ones((3000, 144)),
              LABEL_NAME: None,
              props.OFFSETS.value: [105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455, 490, 525, 560,
                                    595, 630, 665, 700, 735, 770, 805, 840, 875, 910, 945, 980, 1015, 1050,
                                    1085, 1120, 1155, 1190, 1225, 1260, 1295, 1330, 1365, 1400, 1435, 1470, 1505, 1540,
                                    1575, 1610, 1645, 1680, 1715, 1750, 1785, 1820, 1855, 1890, 1925, 1960, 1995, 2030,
                                    2065, 2100, 2135, 2170, 2205, 2240, 2275, 2310, 2345, 2380, 2415, 2450, 2485, 2520,
                                    2555, 2590, 2625, 2660, 2695, 2730, 2765, 2800, 2835, 2870, 2905, 2940, 2975, 3010,
                                    3045, 3080, 3115, 3150, 3185, 3220, 3255, 3290, 3325, 3360, 3395, 3430, 3465, 3500,
                                    3535, 3570, 3605, 3640, 3675, 3710, 3745, 3780, 3815, 3850, 3885, 3920, 3955, 3990,
                                    4025, 4060, 4095, 4130, 4165, 4200, 4235, 4270, 4305, 4340, 4375, 4410, 4445, 4480,
                                    4515, 4550, 4585, 4620, 4655, 4690, 4725, 4760, 4795, 4830, 4865, 4900, 4935, 4970,
                                    5005, 5040, 5075, 5110],
              props.INLINE.value: 980, props.CROSSLINE.value: 185, props.PIXEL_DEPTH.value: 0,
              props.COHERENCE.value: np.random.uniform(low=0.0005, high=0.63, size=(200, 66)),
              props.VELOCITIES.value: None
              },
             {"time_gate_ms": 30, "time_range_ms": [0, 5998.0], "sample_rate_ms": 2.0, "velocity_range": [1250, 4500],
              "velocity_step": 50, "handle_sample_zero": True, "initial_velocity": None,
              "initial_analysis_time_ms": None, "spl_order": 1, "spl_smooth": 0.1, "spl_ext": 0,
              "savgol_window": 51, "savgol_order": 1})
        ]
    )
    def test_compute_velocity_function_call(self, velocity_settings, datum_settings):
        velocity = gat.ComputeVelocityFunction(**velocity_settings)
        fake_datum = _get_stub_cdp_gather_datum(**datum_settings)

        # Act
        actual_datum = velocity(fake_datum)

        # Assert
        assert actual_datum.velocities is not None
        assert len(actual_datum.velocities[0]) == 2

    @pytest.mark.parametrize(
        "time_gate_ms, time_range_ms, sample_rate_ms, velocity_range, velocity_step, handle_sample_zero, "
        "initial_velocity, initial_analysis_time_ms",
        [
            (0, [0, 5998.0], 0.0, [1250, 4500], 0, True, None, None)
        ]
    )
    def test_compute_velocity_function_str(self, time_gate_ms, time_range_ms, sample_rate_ms, velocity_range,
                                           velocity_step, handle_sample_zero, initial_velocity,
                                           initial_analysis_time_ms):
        # Arrange:
        velocity = gat.ComputeVelocityFunction(time_gate_ms, time_range_ms, sample_rate_ms, velocity_range,
                                               velocity_step, initial_velocity, initial_analysis_time_ms)
        expected_output_string = f"<ComputeVelocityFunction time gate: {time_gate_ms}, time_range: {time_range_ms}, " \
                                 f"sample rate: {sample_rate_ms}, velocity range: {velocity_range}, " \
                                 f"velocity step: {velocity_step}>"

        # Act:
        actual_output_string = velocity.__str__()

        # Assert:
        assert actual_output_string == expected_output_string


class TestGenerateGatherWindows:
    @pytest.mark.parametrize(
        "generate_gather_settings",
        [
            ({"time_range_ms": [0, 5998.0], "sample_rate_ms": 2.0, "num_samples": 3000, "velocity_range": [1250, 4500],
              "window_size": 64, "stride": 64,
              "velocity_deltas": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
                                  750, 800, 850, 900, 950, 1000]}),
        ]
    )
    def test_generate_gather_init(self, generate_gather_settings):
        # Arrange
        generate_gather = gat.GenerateGatherWindows(**generate_gather_settings)

        # Act
        time_range_ms = generate_gather.time_range_ms
        sample_rate_ms = generate_gather.sample_rate_ms
        num_samples = generate_gather.num_samples
        velocity_range = generate_gather.velocity_range
        window_size = generate_gather.window_size
        stride = generate_gather.stride
        velocity_deltas = generate_gather.velocity_deltas

        # Assert
        assert time_range_ms == generate_gather_settings["time_range_ms"]
        assert sample_rate_ms == generate_gather_settings["sample_rate_ms"]
        assert num_samples == generate_gather_settings["num_samples"]
        assert velocity_range == generate_gather_settings["velocity_range"]
        assert window_size == generate_gather_settings["window_size"]
        assert stride == generate_gather_settings["stride"]
        assert velocity_deltas == generate_gather_settings["velocity_deltas"]

    @pytest.mark.parametrize(
        "datum_settings, generate_gather_settings",
        [
            ({FEATURE_NAME: np.ones((3000, 144)),
              LABEL_NAME: None,
              props.OFFSETS.value: [105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455, 490, 525, 560,
                                    595, 630, 665, 700, 735, 770, 805, 840, 875, 910, 945, 980, 1015, 1050,
                                    1085, 1120, 1155, 1190, 1225, 1260, 1295, 1330, 1365, 1400, 1435, 1470, 1505, 1540,
                                    1575, 1610, 1645, 1680, 1715, 1750, 1785, 1820, 1855, 1890, 1925, 1960, 1995, 2030,
                                    2065, 2100, 2135, 2170, 2205, 2240, 2275, 2310, 2345, 2380, 2415, 2450, 2485, 2520,
                                    2555, 2590, 2625, 2660, 2695, 2730, 2765, 2800, 2835, 2870, 2905, 2940, 2975, 3010,
                                    3045, 3080, 3115, 3150, 3185, 3220, 3255, 3290, 3325, 3360, 3395, 3430, 3465, 3500,
                                    3535, 3570, 3605, 3640, 3675, 3710, 3745, 3780, 3815, 3850, 3885, 3920, 3955, 3990,
                                    4025, 4060, 4095, 4130, 4165, 4200, 4235, 4270, 4305, 4340, 4375, 4410, 4445, 4480,
                                    4515, 4550, 4585, 4620, 4655, 4690, 4725, 4760, 4795, 4830, 4865, 4900, 4935, 4970,
                                    5005, 5040, 5075, 5110],
              props.INLINE.value: 980, props.CROSSLINE.value: 185, props.PIXEL_DEPTH.value: 0,
              props.COHERENCE.value: np.random.uniform(low=0.0005, high=0.63, size=(200, 66)),
              props.VELOCITIES.value: [[336, 1711], [702, 1845]]
              },
             {"time_range_ms": [0, 5998.0], "sample_rate_ms": 2.0, "num_samples": 3000, "velocity_range": [1250, 4500],
              "window_size": 64, "stride": 64,
              "velocity_deltas": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
                                  750, 800, 850, 900, 950, 1000]})
        ]
    )
    def test_generate_gather_call(self, generate_gather_settings, datum_settings):
        generate_gather_windows = gat.GenerateGatherWindows(**generate_gather_settings)
        fake_datum = _get_stub_cdp_gather_datum(**datum_settings)

        # Act
        output = generate_gather_windows(fake_datum)

        # Assert
        assert output is not None
        assert len(output) == 1886

    @pytest.mark.parametrize(
        "time_range_ms, sample_rate_ms, num_samples, velocity_range, window_size, stride, velocity_deltas",
        [
            ([0, 5998.0], 2.0, 3000, [1250, 4500], 64, 64,
             [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
              750, 800, 850, 900, 950, 1000])
        ]
    )
    def test_generate_gather_str(self, time_range_ms, sample_rate_ms, num_samples, velocity_range, window_size, stride,
                                 velocity_deltas):
        # Arrange:
        generate_gather = gat.GenerateGatherWindows(time_range_ms, sample_rate_ms, num_samples, velocity_range,
                                                    window_size, stride,
                                                    velocity_deltas)

        expected_output_string = f"<GenerateGatherWindows window size: {window_size}, stride: {stride}, " \
                                 f"velocity deltas: " f"{True if len(velocity_deltas) > 0 else False}, " \
                                 f"time_range: {time_range_ms}, " f"sample rate: {sample_rate_ms}, " \
                                 f"velocity range: {velocity_range}> "

        # Act:
        actual_output_string = generate_gather.__str__()

        # Assert:
        assert actual_output_string == expected_output_string


class TestConcatenateVelocities:
    @pytest.mark.parametrize(
        "datum_settings, other_datum",
        [
            ({FEATURE_NAME: np.ones((3000, 144)),
              LABEL_NAME: None,
              props.OFFSETS.value: [140],
              props.INLINE.value: 980, props.CROSSLINE.value: 185, props.PIXEL_DEPTH.value: 0,
              props.COHERENCE.value: np.random.uniform(low=0.0005, high=0.63, size=(200, 66)),
              props.VELOCITIES.value: [[336, 1711]]
              },
             {FEATURE_NAME: np.ones((3000, 144)),
              LABEL_NAME: None,
              props.OFFSETS.value: [140],
              props.INLINE.value: 980, props.CROSSLINE.value: 185, props.PIXEL_DEPTH.value: 0,
              props.COHERENCE.value: np.random.uniform(low=0.0005, high=0.63, size=(200, 66)),
              props.VELOCITIES.value: [[702, 1845]]}
             )
        ]
    )
    def test_concatenate_velocities_call(self, datum_settings, other_datum):
        datum_list = []
        fake_datum = _get_stub_cdp_gather_datum(**datum_settings)
        other_fake_datum = _get_stub_cdp_gather_datum(**other_datum)
        velocities_concatenated = gat.ConcatenateVelocities()
        datum_list.append(fake_datum)
        datum_list.append(other_fake_datum)
        new_datum = velocities_concatenated(datum_list)

        assert len(new_datum.velocities) == 2

    def test_concatenate_velocities_str(self):
        velocities_concatenated = gat.ConcatenateVelocities()
        expected_string = velocities_concatenated.__str__()

        assert expected_string == f"<ConcatenateVelocities>>"


class TestFilterVelocities:
    @pytest.mark.parametrize(
        "savgol_settings",
        [
            ({"savgol_window": 3, "savgol_order": 1, "initial_velocity": None, "initial_analysis_time_ms": None})
        ]
    )
    def test_filter_velocities_init(self, savgol_settings):
        filter_velocities = gat.FilterVelocities()

        # Act
        savgol_window = filter_velocities.savgol_window
        savgol_order = filter_velocities.savgol_order
        initial_velocity = filter_velocities.initial_velocity
        initial_analysis = filter_velocities.initial_time

        # Assert
        assert savgol_window == savgol_settings["savgol_window"]
        assert savgol_order == savgol_settings["savgol_order"]
        assert initial_velocity == savgol_settings["initial_velocity"]
        assert initial_analysis == savgol_settings["initial_analysis_time_ms"]

    @pytest.mark.parametrize(
        "datum_settings",
        [
            ({FEATURE_NAME: np.ones((3000, 144)),
              LABEL_NAME: None,
              props.OFFSETS.value: [105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455, 490, 525, 560,
                                    595, 630, 665, 700, 735, 770, 805, 840, 875, 910, 945, 980, 1015, 1050,
                                    1085, 1120, 1155, 1190, 1225, 1260, 1295, 1330, 1365, 1400, 1435, 1470, 1505, 1540,
                                    1575, 1610, 1645, 1680, 1715, 1750, 1785, 1820, 1855, 1890, 1925, 1960, 1995, 2030,
                                    2065, 2100, 2135, 2170, 2205, 2240, 2275, 2310, 2345, 2380, 2415, 2450, 2485, 2520,
                                    2555, 2590, 2625, 2660, 2695, 2730, 2765, 2800, 2835, 2870, 2905, 2940, 2975, 3010,
                                    3045, 3080, 3115, 3150, 3185, 3220, 3255, 3290, 3325, 3360, 3395, 3430, 3465, 3500,
                                    3535, 3570, 3605, 3640, 3675, 3710, 3745, 3780, 3815, 3850, 3885, 3920, 3955, 3990,
                                    4025, 4060, 4095, 4130, 4165, 4200, 4235, 4270, 4305, 4340, 4375, 4410, 4445, 4480,
                                    4515, 4550, 4585, 4620, 4655, 4690, 4725, 4760, 4795, 4830, 4865, 4900, 4935, 4970,
                                    5005, 5040, 5075, 5110],
              props.INLINE.value: 980, props.CROSSLINE.value: 185, props.PIXEL_DEPTH.value: 0,
              props.COHERENCE.value: np.random.uniform(low=0.0005, high=0.63, size=(200, 66)),
              props.VELOCITIES.value: [[336, 1711], [702, 1845], [945, 425]]
              })
        ]
    )
    def test_filter_velocities_call(self, datum_settings):
        filter_velocities = gat.FilterVelocities()
        fake_datum = _get_stub_cdp_gather_datum(**datum_settings)

        output = filter_velocities(fake_datum)

        # assert output.velocities[0][1] is None
        assert round(output.velocities[0][1]) == 1970
        assert round(output.velocities[1][1]) == 1327

    @pytest.mark.parametrize(
        "savgol_window, savgol_order",
        [
            (3, 1)
        ]
    )
    def test_filter_velocities_str(self, savgol_window, savgol_order):
        filter_velocities = gat.FilterVelocities()

        expected_string = filter_velocities.__str__()

        actual_string = f"<FilterVelocities savgol window: {savgol_window}, savgol order: {savgol_order} "

        assert actual_string == expected_string


class TestSmoothVelocities:
    @pytest.mark.parametrize(
        "smooth_settings",
        [
            ({"time_range_ms": [0, 2000], "sample_rate_ms": 10, "savgol_window": 5, "savgol_order": 1})
        ]
    )
    def test_filter_velocities_init(self, smooth_settings):
        smooth_velocities = gat.SmoothVelocities(**smooth_settings)

        # Act
        time_range_ms = smooth_velocities.time_range
        sample_rate_ms = smooth_velocities.sample_rate
        savgol_window = smooth_velocities.savgol_window
        savgol_order = smooth_velocities.savgol_order

        # Assert
        assert time_range_ms == smooth_settings["time_range_ms"]
        assert sample_rate_ms == smooth_settings["sample_rate_ms"]
        assert savgol_window == smooth_settings["savgol_window"]
        assert savgol_order == smooth_settings["savgol_order"]

    @pytest.mark.parametrize(
        "datum_settings, smooth_settings",
        [
            ({FEATURE_NAME: np.ones((3000, 144)),
              LABEL_NAME: None,
              props.OFFSETS.value: [105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455, 490, 525, 560,
                                    595, 630, 665, 700, 735, 770, 805, 840, 875, 910, 945, 980, 1015, 1050,
                                    1085, 1120, 1155, 1190, 1225, 1260, 1295, 1330, 1365, 1400, 1435, 1470, 1505, 1540,
                                    1575, 1610, 1645, 1680, 1715, 1750, 1785, 1820, 1855, 1890, 1925, 1960, 1995, 2030,
                                    2065, 2100, 2135, 2170, 2205, 2240, 2275, 2310, 2345, 2380, 2415, 2450, 2485, 2520,
                                    2555, 2590, 2625, 2660, 2695, 2730, 2765, 2800, 2835, 2870, 2905, 2940, 2975, 3010,
                                    3045, 3080, 3115, 3150, 3185, 3220, 3255, 3290, 3325, 3360, 3395, 3430, 3465, 3500,
                                    3535, 3570, 3605, 3640, 3675, 3710, 3745, 3780, 3815, 3850, 3885, 3920, 3955, 3990,
                                    4025, 4060, 4095, 4130, 4165, 4200, 4235, 4270, 4305, 4340, 4375, 4410, 4445, 4480,
                                    4515, 4550, 4585, 4620, 4655, 4690, 4725, 4760, 4795, 4830, 4865, 4900, 4935, 4970,
                                    5005, 5040, 5075, 5110],
              props.INLINE.value: 2, props.CROSSLINE.value: 3, props.PIXEL_DEPTH.value: 0,
              props.COHERENCE.value: None,
              props.VELOCITIES.value: [[336, 1711], [702, 1845], [945, 425]]
              },
             {"time_range_ms": [0, 2000], "sample_rate_ms": 100, "savgol_window": 5, "savgol_order": 1})
        ]
    )
    def test_filter_velocities_call(self, datum_settings, smooth_settings):
        smooth_velocities = gat.SmoothVelocities(**smooth_settings)
        fake_datum = _get_stub_cdp_gather_datum(**datum_settings)

        output = smooth_velocities(fake_datum)

        assert output.velocities[1][0] == 100
        assert round(output.velocities[1][1]) == 1625
        assert output.velocities[2][0] == 200
        assert round(output.velocities[2][1]) == 1661

    @pytest.mark.parametrize(
        "smooth_settings",
        [
            ({"time_range_ms": [0, 2000], "sample_rate_ms": 10, "savgol_window": 3, "savgol_order": 2})
        ]
    )
    def test_filter_velocities_str(self, smooth_settings):
        smooth_velocities = gat.SmoothVelocities(**smooth_settings)

        expected_string = smooth_velocities.__str__()

        actual_string = f"<SmoothVelocities savgol window: {smooth_settings['savgol_window']}, " \
                        f"savgol order: {smooth_settings['savgol_order']} "

        assert actual_string == expected_string
