""" Copyright 2023 IBM Research. All Rights Reserved.

    - Functions to modify gather seisfast as numpy arrays.
"""

from typing import Union, List, Tuple

import numpy as np
from rockml.data.array_ops import view_as_windows
from scipy.interpolate import splrep, splev


def nmo(trace: np.ndarray, offset_value: float, velocities: np.ndarray, initial_time_s: float = 0.0,
        sample_rate_s: float = 0.002) -> np.ndarray:
    """ Applies Normal Move Out correction on *trace* using numpy native operations in arrays.
        Notice that units related to time should be in seconds.

    Args:
        trace: 1D numpy array containing a single seismic trace.
        offset_value: (float) offset value in meters.
        velocities: 1D numpy array containing velocity values in meters/second.
        initial_time_s: (float) initial time in seconds.
        sample_rate_s: (float) sample rate in seconds.

    Returns:
        np.ndarray with corrected trace
    """

    times = np.arange(initial_time_s, len(trace) * sample_rate_s + initial_time_s, sample_rate_s)[:len(trace)]

    t = (np.sqrt(np.square(times) + (offset_value ** 2) / np.square(velocities)) / sample_rate_s).astype(np.int32)

    valid_idx = np.where(t < len(trace))

    output = np.zeros(trace.shape)
    output[valid_idx] = trace[t[valid_idx]]

    return output


def correct_gather(gather: np.ndarray, offsets: np.ndarray, velocity: Union[float, int, np.ndarray],
                   time_range_ms: List[float], sample_rate_ms: float) -> np.ndarray:
    """ Corrects a gather using NMO correction. Attention to units, as described below! The milliseconds units are
        converted to seconds units.

    Args:
        gather: 2D numpy array representing gather seisfast; shape = (num_points_per_trace, num_traces).
        offsets: 1D numpy array with offset values; shape = (num_traces,).
        velocity: (float, np.ndarray) velocity value or values in meters/second.
        time_range_ms: list containing the time range in milliseconds; format = [initial_time, final_time]
        sample_rate_ms: (float) sample rate in milliseconds.

    Returns:
        np.ndarray with the corrected gather.
    """

    corrected_gather = np.zeros(gather.shape, dtype=gather.dtype)

    if type(velocity) in [int, float]:
        velocity = np.full(shape=gather.shape[0], fill_value=velocity)

    for trace in range(gather.shape[1]):
        corrected_gather[:, trace] = nmo(trace=gather[:, trace], offset_value=offsets[trace],
                                         velocities=velocity,
                                         initial_time_s=time_range_ms[0] / 1000, sample_rate_s=sample_rate_ms / 1000)

    return corrected_gather


def divide_in_time_windows(gather: np.ndarray, time_gate_ms: float, sample_rate_ms: float) -> np.ndarray:
    """ Divide *gather* in non overlapping windows of size *time_gate* / *sample_rate*.

    Args:
        gather: 2D numpy array representing gather seisfast; shape = (num_points_per_trace, num_traces).
        time_gate_ms: (float) time gate in milliseconds.
        sample_rate_ms: (float) sample rate in milliseconds.

    Returns:
        ndarray of shape (num_windows, window_size, num_traces)
    """

    window_size = round(time_gate_ms / sample_rate_ms)
    num_traces = gather.shape[1]

    blocks = view_as_windows(gather, window_shape=(window_size, num_traces), stride=(window_size, num_traces))

    blocks = blocks.reshape((-1, window_size, num_traces))

    return blocks


def _semblance(gather_block: np.ndarray) -> float:
    """ Calculate semblance for a gather window *gather_block*.

    Args:
        gather_block: 2D numpy array with a single gather block; shape = (time_block_size, num_velocities)

    Returns:
        (float) semblance value
    """

    num = np.sum(np.sum(gather_block, axis=1) ** 2)
    den = np.sum(np.sum(gather_block ** 2, axis=1))
    if den > 0:
        ne = num / (den * gather_block.shape[1])
    else:
        ne = 0.0

    return ne


def semblance_in_time_windows(gather: np.ndarray, time_gate_ms: float, sample_rate_ms: float) -> np.ndarray:
    """ Compute semblance for *gather* divided in non overlapping windows of size *time_gate* / *sample_rate*.

    Args:
        gather: 2D numpy array representing gather seisfast; shape = (num_points_per_trace, num_traces).
        time_gate_ms: (float) time gate in milliseconds.
        sample_rate_ms: (float) sample rate in milliseconds.

    Returns:
        ndarray of semblance values for each time window; shape = (num_windows,)
    """

    blocks = divide_in_time_windows(gather, time_gate_ms, sample_rate_ms)

    semblance_values = np.asarray([_semblance(blocks[i, :, :]) for i in range(blocks.shape[0])])

    return semblance_values


def evaluate_velocity_function(velocity_function: List[List[int]], time_values: np.ndarray) -> np.ndarray:
    """ Get the velocity values for the specified time values *time_values*, using linear interpolation.

    Args:
        velocity_function: list of pairs in the form [[time_value, velocity_value], ...].
        time_values: np.ndarray of time values.

    Returns:
        np.ndarray of velocity values.
    """

    velocity_fn_array = np.asarray(velocity_function).transpose()

    spline = splrep(velocity_fn_array[0], velocity_fn_array[1], s=0, k=1)
    velocity_values_new = splev(time_values, spline, ext=0)

    return velocity_values_new


def resample_velocity_function(velocity_function: List[List[int]], time_range_ms: List[float],
                               sample_rate_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Wrapper of the function evaluate_velocity_function() to get new velocity and time values based on the time range
        and sample rate provided by the user (*time_range_ms* and *sample_rate_ms*).

    Args:
        velocity_function: list of pairs in the form [[time_value, velocity_value], ...].
        time_range_ms: (list) contains the time range in milliseconds; format = [initial_time, final_time].
        sample_rate_ms: (float) sample rate in milliseconds.

    Returns:
        np.ndarray of time values.
        np.ndarray of velocity values.
    """

    times = np.arange(time_range_ms[0], time_range_ms[1] + 1, sample_rate_ms)
    velocities = evaluate_velocity_function(velocity_function, times)

    return times, velocities
