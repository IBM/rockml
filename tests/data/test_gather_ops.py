""" Copyright 2023 IBM Research. All Rights Reserved.

    - Test functions for seisfast.gather_ops.
"""

import numpy as np
import pytest

from rockml.data import gather_ops as ops


@pytest.mark.parametrize(
    "array_shape_after_nmo, offset_value, initial_time_s, sample_rate_s",
    [
        ((3000,), 105, 0.0, 0.002),
    ]
)
def test_nmo(array_shape_after_nmo, offset_value, initial_time_s, sample_rate_s):
    # Arrange
    trace = np.ones((3000,))
    velocities = np.ones((3000,))

    # Act
    actual = ops.nmo(trace, initial_time_s, velocities, sample_rate_s)
    print(actual.shape)
    # Assert
    assert np.array_equal(array_shape_after_nmo, actual.shape)


@pytest.mark.parametrize(
    "array_shape_after_correct, velocity, time_range_ms, sample_rate_ms",
    [
        ((3000, 144), 1250, [0, 5998.0], 2.0),
    ]
)
def test_correct_gather(array_shape_after_correct, velocity, time_range_ms, sample_rate_ms):
    # Arrange
    offsets = np.ones(144)
    gather = np.ones((3000, 144))

    # Act
    actual = ops.correct_gather(gather, offsets, velocity, time_range_ms, sample_rate_ms)

    # Assert
    assert np.array_equal(array_shape_after_correct, actual.shape)


@pytest.mark.parametrize(
    "array_shape_after_time_windows, time_gate_ms, sample_rate_ms",
    [
        ((200,), 30, 2.0),
    ]
)
def test_semblance_in_time_windows(array_shape_after_time_windows, time_gate_ms, sample_rate_ms):
    # Arrange
    gather = np.ones((3000, 144))

    # Act
    actual = ops.semblance_in_time_windows(gather, time_gate_ms, sample_rate_ms)

    # Assert
    assert np.array_equal(array_shape_after_time_windows, actual.shape)


@pytest.mark.parametrize(
    "array_shape_after_evaluate",
    [
        ((46,)),
    ]
)
def test_evaluate_velocity_function(array_shape_after_evaluate):
    # Arrange
    velocity_function = [[14, 1603], [252, 1659], [496, 1791], [722, 1882], [890, 1924], [1106, 2041], [1252, 2088],
                         [1420, 2189], [1586, 2279], [1772, 2406], [1960, 2517], [2082, 2655], [2258, 2740],
                         [2498, 2851], [2632, 2957], [3526, 3244], [5972, 4202]]
    time_values = np.ones(46)

    # Act
    actual = ops.evaluate_velocity_function(velocity_function, time_values)

    # Assert
    assert np.array_equal(array_shape_after_evaluate, actual.shape)
