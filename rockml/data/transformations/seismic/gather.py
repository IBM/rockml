""" Copyright 2023 IBM Research. All Rights Reserved.
"""

from typing import List, Tuple, Union

import numpy as np
from rockml.data import gather_ops as ops
from rockml.data.adapter.seismic.segy.prestack import VELOCITY_FORMAT, CDPGatherDatum
from rockml.data.array_ops import view_as_windows
from rockml.data.transformations import Transformation
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter


class ComputeCoherence(Transformation):
    def __init__(self, time_gate_ms: float, time_range_ms: List[float], sample_rate_ms: float,
                 velocity_range: List[int], velocity_step: int):
        """  Initialize ComputeCoherence.

        Args:
            time_gate_ms: (float) time gate in milliseconds.
            time_range_ms: (list) contains the time range in milliseconds; format = [initial_time, final_time].
            sample_rate_ms: (float) sample rate in milliseconds.
            velocity_range: (list) velocity range in m/s; format = [velocity_first_value, velocity_last_value].
            velocity_step: (int) velocity step in m/s.
        """

        self.time_gate_ms = time_gate_ms
        self.time_range_ms = time_range_ms
        self.sample_rate_ms = sample_rate_ms
        self.velocity_range = velocity_range
        self.velocity_step = velocity_step

    def __call__(self, datum: CDPGatherDatum) -> CDPGatherDatum:
        """ Compute coherence for each velocity in *self.velocity_range* in 2 steps:
            1) correct gather using NMO.
            2) calculate coherence.

            OBS1: This transformation is assuming that the datum features were NOT corrected before!

            OBS2: The corrected gather is not saved in *datum*, as this would be a hidden side effect. To correct the
            gather and save the results in *datum*, use a specific Transformation to do it.

        Args:
            datum: CDPGatherDatum containing traces as features and offsets.

        Returns:
            same CDPGatherDatum with the added coherence.
        """

        v_range = range(self.velocity_range[0], self.velocity_range[1] + 1, self.velocity_step)

        coherence = [None] * len(v_range)

        for count, v in enumerate(v_range):
            corrected_gather = ops.correct_gather(datum.features, datum.offsets,
                                                  v, self.time_range_ms, self.sample_rate_ms)

            coherence[count] = ops.semblance_in_time_windows(corrected_gather, self.time_gate_ms, self.sample_rate_ms)

        datum.coherence = np.stack(coherence, axis=1)

        return datum

    def __str__(self):
        return f"<ComputeCoherence time gate: {self.time_gate_ms}, time_range: {self.time_range_ms}, " \
               f"sample rate: {self.sample_rate_ms}, velocity range: {self.velocity_range}, " \
               f"velocity step: {self.velocity_step}>"


class ComputeVelocityFunction(Transformation):
    def __init__(self, time_gate_ms: float, time_range_ms: List[float], sample_rate_ms: float,
                 velocity_range: List[int], velocity_step: int, initial_velocity: Union[float, None],
                 initial_analysis_time_ms: float = None, handle_sample_zero: bool = True,
                 spl_order: int = 1, spl_smooth: float = 0.1, spl_ext: int = 0,
                 savgol_window: int = 51, savgol_order: int = 1):
        """  Initialize ComputeCoherence.

        Args:
            time_gate_ms: (float) time gate in milliseconds.
            time_range_ms: (list) contains the time range in milliseconds; format = [initial_time, final_time].
            sample_rate_ms: (float) sample rate in milliseconds.
            velocity_range: (list) velocity range in m/s; format = [velocity_first_value, velocity_last_value].
            velocity_step: (int) velocity step in m/s.
            initial_velocity: initial velocity value in m/s to appear in the velocity functions. If None, no
                restriction is applied to the interpolation. Ignored if initial_analysis_time_ms is None.
            initial_analysis_time_ms: (float) time value in ms to start the analysis, that is, the interpolation and
                smoothing will only be carried out from this value forward. If None, all time values are considered.
            handle_sample_zero: (boolean) if one wants to handle initial gather samples with zero values.
            spl_order: (int) the degree of the spline fit (k in splrep).
            spl_smooth: (float) the user can use s to control the trade-off between closeness and smoothness of fit.
                Larger s means more smoothing while smaller values of s indicate less smoothing (s in splrep).
            spl_ext: (int) controls the value returned for elements of not in the interval defined by the knot sequence.
                It is the `ext` parameter in splev; the default value is 0.

                * if spl_ext=0, return the extrapolated value.
                * if spl_ext=1, return 0
                * if spl_ext=2, raise a ValueError
                * if spl_ext=3, return the boundary value.

            savgol_window: (int) the length of the Savitzky-Golay filter window (window_length in savgol_filter).
            savgol_order: (int) the order of the polynomial used to fit the samples (polyorder in savgol_filter).
        """

        self.time_gate_ms = time_gate_ms
        self.time_range_ms = time_range_ms
        self.sample_rate_ms = sample_rate_ms
        self.velocity_range = velocity_range
        self.velocity_step = velocity_step
        self.initial_velocity = initial_velocity
        self.handle_sample_zero = handle_sample_zero
        self.spl_order = spl_order
        self.spl_smooth = spl_smooth
        self.spl_ext = spl_ext
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order

        # Changing time from milliseconds to semblance pixel coordinate:
        if initial_analysis_time_ms is not None:
            self.initial_analysis_time = int((initial_analysis_time_ms - self.time_range_ms[0]) / self.time_gate_ms)
        else:
            self.initial_analysis_time = None
            self.initial_velocity = None

    def __call__(self, datum: CDPGatherDatum) -> CDPGatherDatum:
        """ Compute velocity function based on calculated semblance values. The method uses interpolation and smoothing.

        Args:
            datum: CDPGatherDatum containing traces (features), offsets and semblance.

        Returns:
            CDPGatherDatum with the added velocities.
        """

        if self.handle_sample_zero:
            semblance_array = self._modify_semblance(datum)
        else:
            semblance_array = datum.coherence

        velocity_ax, time_ax, semblance_values = self._process_semblance(semblance_array)

        # Start the interpolation/smoothing from self.initial_analysis_time forward.
        if self.initial_analysis_time:
            velocity_ax = velocity_ax[self.initial_analysis_time:]
            time_ax = time_ax[self.initial_analysis_time:]
            semblance_values = semblance_values[self.initial_analysis_time:]

        times, filtered_velocities = self._compute_velocity_fn(velocity_values=velocity_ax, time_values=time_ax,
                                                               semblance_values=semblance_values,
                                                               num_points=semblance_array.shape[0])
        # TODO: check if this necessary
        if filtered_velocities is None:
            velocity_fn = None
        else:
            # Correcting units and range for times and velocities
            times = times * self.time_gate_ms + self.time_range_ms[0]
            filtered_velocities = filtered_velocities * self.velocity_step + self.velocity_range[0]

            # Adding the first (t, v) point, if self.initial_velocity and self.initial_analysis_time are available
            if self.initial_velocity is not None:
                times = np.concatenate([[self.time_range_ms[0]], times])
                filtered_velocities = np.concatenate([[self.initial_velocity], filtered_velocities])

            velocity_fn = self._create_velocity_fn(times, filtered_velocities)

        datum.velocities = velocity_fn

        return datum

    def __str__(self):
        return f"<ComputeVelocityFunction time gate: {self.time_gate_ms}, time_range: {self.time_range_ms}, " \
               f"sample rate: {self.sample_rate_ms}, velocity range: {self.velocity_range}, " \
               f"velocity step: {self.velocity_step}>"

    @staticmethod
    def _process_semblance(semblance_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Process semblance plot with the following steps:
           1) Normalize semblance;
           2) Get maximum semblance values for each line (time axis).

        Args:
            semblance_array: 2D np.ndarray with semblance plot; shape = (num_time_windows, num_velocities).

        Returns:
            1D ndarray of velocity axis values.
            1D ndarray of time axis values.
            1D ndarray of semblance values for respective velocity and time points.
        """

        # semblance normalization
        semblance_array = (semblance_array - semblance_array.min()) / (semblance_array.max() - semblance_array.min())

        # x is the velocity axis, y is the time axis and z is the array of processed semblance values
        velocity_ax = np.argmax(semblance_array, axis=1)  # maximum semblance values per line
        time_ax = np.arange(semblance_array.shape[0])
        semblance_values = semblance_array[time_ax, velocity_ax]

        return velocity_ax, time_ax, semblance_values

    def _modify_semblance(self, datum: CDPGatherDatum) -> np.ndarray:
        """ Copy and modify *datum.coherence* array to handle initial gather samples with zero values.

            1) Break datum traces into time windows using divide_in_time_windows() function;
            2) Loop through windows looking for gather blocks which sum 0.
              - for these blocks, the semblance value is set to zero for the whole line and 1 only in the position
                 (block_number, block_number)
              - the loop stops in the first valid gather line (non-zero)

            OBS: *datum* is NOT modified; the function returns a new semblance array.

        Args:
            datum: CDPGatherDatum containing traces (features) and offsets.

        Returns:
            modified semblance
        """

        gather_blocks = ops.divide_in_time_windows(datum.features, self.time_gate_ms, self.sample_rate_ms)
        semblance = datum.coherence.copy()

        # TODO: is this k variable necessary ??
        k = 0
        for b in range(gather_blocks.shape[0]):

            if abs(gather_blocks[b, :, :].sum() - 0) < 1e-5:
                semblance[k, :] = 0
                if k < semblance.shape[1]:
                    semblance[k, k] = 1
            else:
                break
            k += 1

        return semblance

    def _compute_velocity_fn(self, velocity_values: np.ndarray, time_values: np.ndarray, semblance_values: np.ndarray,
                             num_points: int) -> Union[Tuple[None, None], Tuple[np.ndarray, np.ndarray]]:
        """ Computes velocity function using interpolation and smoothing.

        Args:
            velocity_values: 1D ndarray with velocity values.
            time_values: 1D ndarray with time values.
            semblance_values: 1D ndarray with semblance values.
            num_points: (int) number of points to interpolate.

        Returns:
            Tuple (new_time_values, new_velocity_values).
        """

        # TODO: add the specific exception to catch
        try:

            initial_point = self.initial_analysis_time if self.initial_analysis_time else 0

            time_values_new = np.arange(initial_point, num_points)

            weights = semblance_values
            weights = weights / np.sqrt(weights.sum())

            spline = splrep(time_values, velocity_values, w=weights, s=self.spl_smooth, k=self.spl_order)
            velocity_values_new = splev(time_values_new, spline, ext=self.spl_ext)

            velocity_values_filtered = savgol_filter(velocity_values_new, self.savgol_window, self.savgol_order)

            # avoid negative values
            mask = velocity_values_filtered < 0
            velocity_values_filtered[mask] = 0

        except:
            time_values_new = None
            velocity_values_filtered = None

        return time_values_new, velocity_values_filtered

    @staticmethod
    def _create_velocity_fn(times: np.ndarray, velocities: np.ndarray) -> VELOCITY_FORMAT:
        """ Creates list of pairs in the form [time_value, velocity_value] from *times* and *velocities*.

        Args:
            times: 1D ndarray containing time values.
            velocities: 1D ndarray with the velocity values.

        Returns:
            list of velocity function points.
        """

        last_v = None
        velocity_fn = []

        for t, v in zip(times, velocities):

            if last_v is None:
                last_v = int(v)
            else:
                if int(v) == last_v:
                    continue

            last_v = int(v)

            velocity_fn.append([int(t), int(v)])

        return velocity_fn


class GenerateGatherWindows(Transformation):
    def __init__(self, time_range_ms: List[float], sample_rate_ms: float, num_samples: int,
                 velocity_range: List[int], window_size: int, stride: int, velocity_deltas: Union[np.ndarray, List],
                 ignore_velocites: bool = False):
        """  Initialize GenerateGatherWindows.

        Args:
            time_range_ms: (list) contains the time range in milliseconds; format = [initial_time, final_time].
            sample_rate_ms: (float) sample rate in milliseconds.
            num_samples: (int) number of samples in a trace.
            velocity_range: (list) velocity range in m/s; format = [velocity_first_value, velocity_last_value].
            window_size: (int) size of the (square) output gather windows.
            stride: (int) stride to be used in the sliding window that generates de gather tiles; the stride is only
                valid for the height direction.
            velocity_deltas: 1D ndarray containing the velocity delta values that will be used to correct.
            ignore_velocites: (bool) if True, the datum.velocities are ignored, and treated as an array of ones; this
                also means that the gather will not be corrected before generating the windows.
        """

        self.time_range_ms = time_range_ms
        self.sample_rate_ms = sample_rate_ms
        self.velocity_range = velocity_range
        self.velocity_deltas = velocity_deltas
        self.window_size = window_size
        self.stride = stride
        self.num_samples = num_samples
        self.ignore_velocities = ignore_velocites

        if ignore_velocites and len(velocity_deltas) > 0:
            raise ValueError(f"ignore_velocities cannot be set to True when velocity_deltas is not an empty list!")

        # time_windows_idx saves an array of indexes with shape = (num_windows, window_size). As we need to cut gathers
        # in the same way several times, we create the windows indexes once and then use it to get the gather windows.
        self.time_windows_idx = view_as_windows(np.arange(num_samples, dtype=int),
                                                window_shape=self.window_size, stride=stride)

        # Get the time indexes in the center of each window in time_windows_idx and convert to millisecond unit
        self.times = (self.time_windows_idx[:, int(self.time_windows_idx.shape[1] / 2)] * self.sample_rate_ms).astype(
            int)

        # Generate all time images
        self.times_imgs = self._generate_time_images()

    def __call__(self, datum: CDPGatherDatum) -> List[CDPGatherDatum]:
        """ This Transformation generates gather windows with additional information.

            A velocity value is calculated for each time in *self.times*, based on the datum.velocities function. For
            each (time, velocity) pair, the traces in datum.features are corrected with NMO. A window of size
            *self.window_size* centered in the gather pixel corresponding to the time value is collected and rescaled
            to 255 range. A 3-channel image is created by stacking this gather window with 2 additional images: a time
            image, informing the ranges of time the window was extracted from, and the velocity image, containing the
            single value of velocity of the center pixel.

            If velocity_deltas are given, the function also generates these 3D images for (v + delta) and (v - delta).

            The output is a list of CDPGatherDatum containing the 3D images as features and the velocity values
            (considering deltas if given) as labels.

        Args:
            datum: CDPGatherDatum containing traces (features) and offsets.

        Returns:
            list of CDPGatherDatum.
        """

        if not self.ignore_velocities:
            velocity_values = ops.evaluate_velocity_function(datum.velocities, time_values=self.times)
        else:
            velocity_values = np.ones(shape=self.times.shape)

        output_list = [None] * (len(self.times) + 2 * len(self.velocity_deltas) * len(self.times))

        count = 0

        for idx, (v, t) in enumerate(zip(velocity_values, self.times)):

            # original image
            image = self._get_gather_window(datum, idx, v)
            velocity_image = self._get_velocity_image(v)
            times_image = self.times_imgs[idx, :, :]

            # image = np.stack((image, velocity_image, times_image), axis=-1)
            image = np.stack((image, velocity_image[:, :image.shape[1]], times_image[:, :image.shape[1]]), axis=-1)

            # Correcting pixel depth according to the time window
            pixel_depth = datum.pixel_depth + self.time_windows_idx[idx, 0]

            output_list[count] = CDPGatherDatum(features=image,
                                                label=0,
                                                offsets=None, inline=datum.inline, crossline=datum.crossline,
                                                velocities=[[int(t), int(v)]], pixel_depth=pixel_depth)
            count += 1

            for velocity_delta in self.velocity_deltas:
                # minus velocity delta
                image = self._get_gather_window(datum, idx, v - velocity_delta)
                # image = np.stack((image, velocity_image, times_image), axis=-1)
                image = np.stack((image, velocity_image[:, :image.shape[1]], times_image[:, :image.shape[1]]), axis=-1)

                output_list[count] = CDPGatherDatum(features=image,
                                                    label=int(-velocity_delta),
                                                    offsets=None, inline=datum.inline, crossline=datum.crossline,
                                                    velocities=[[int(t), int(v)]], pixel_depth=pixel_depth)
                count += 1

                # plus velocity delta
                image = self._get_gather_window(datum, idx, v + velocity_delta)
                # image = np.stack((image, velocity_image, times_image), axis=-1)
                image = np.stack((image, velocity_image[:, :image.shape[1]], times_image[:, :image.shape[1]]), axis=-1)

                output_list[count] = CDPGatherDatum(features=image,
                                                    label=int(velocity_delta),
                                                    offsets=None, inline=datum.inline, crossline=datum.crossline,
                                                    velocities=[[int(t), int(v)]], pixel_depth=pixel_depth)
                count += 1

        return output_list

    def __str__(self):
        return f"<GenerateGatherWindows window size: {self.window_size}, stride: {self.stride}, velocity deltas: " \
               f"{True if len(self.velocity_deltas) > 0 else False}, time_range: {self.time_range_ms}, " \
               f"sample rate: {self.sample_rate_ms}, velocity range: {self.velocity_range}> "

    def _generate_time_images(self) -> np.ndarray:
        """ Generate time images from the window indexes *self.time_window_idx*. The values only vary in the height
            direction. It also rescales the indexes to the uint8 range (0 .. 255).

        Returns:
            3D array of time images; shape = shape = [num_windows, window_size, window_size]
        """

        times_imgs = np.repeat(np.expand_dims(self.time_windows_idx, axis=-1), self.time_windows_idx.shape[1], axis=-1)
        times_imgs = (times_imgs / (self.num_samples - 1) * 255).astype(np.uint8)

        return times_imgs

    def _get_velocity_image(self, velocity_value: float) -> np.ndarray:
        """ Generate a 2D velocity image with a unique velocity value. This value is the encoding of *velocity_value* in
            the uint8 range (0 .. 255). The normalization is done using the *self.velocity_range*.

        Args:
            velocity_value: (float) velocity value.

        Returns:
            2D array with velocity image; shape = (size, size).
        """

        velocity_image = np.full((self.window_size, self.window_size), ((velocity_value - self.velocity_range[0]) / (
                self.velocity_range[1] - self.velocity_range[0])) * 255, dtype=np.uint8)

        return velocity_image

    def _get_gather_window(self, datum: CDPGatherDatum, idx: int, velocity_value: float) -> np.ndarray:
        """ The output gather window is generated with the following steps:

            1. correct gather (datum.features) using *velocity_value* (if self.ignore_velocity is False);
            2. get the corrected gather window for the corresponding *velocity_value* (indicated by idx);
            3. normalize and rescale the window to the 255 uint8 range.
            4. cut the right side of the window so it has no more than *self.window_size*.

        Args:
            datum: CDPGatherDatum with features and offsets.
            idx: (int) an index indicating which range of from the *self.time_windows_idx* should be used.
            velocity_value: (float) velocity to be used in gather correction.

        Returns:
            2D uint8 ndarray with the gather window.
        """

        if not self.ignore_velocities:
            corrected_gather = ops.correct_gather(datum.features, datum.offsets, velocity_value,
                                                  self.time_range_ms, self.sample_rate_ms)
        else:
            corrected_gather = np.copy(datum.features)

        gather_window = corrected_gather[self.time_windows_idx[idx, :], :]

        gather_window = (gather_window - gather_window.min()) / (gather_window.max() - gather_window.min())
        gather_window *= 255
        gather_window = gather_window[:, :self.window_size].astype(np.uint8)

        return gather_window


class ConcatenateVelocities(Transformation):

    def __call__(self, datum_list: List[CDPGatherDatum]) -> CDPGatherDatum:
        """ Concatenate the velocities of each datum in *datum_list* in a single velocity. The output is an empty datum,
            only with the velocities, inline and crossline information.

            OBS 1: This transformation assumes that all tiles in datum list comes from the same gather.

            OBS 2: This transformation assumes that each datum in datum_list contains a single velocity point.

        Args:
            datum_list: CDPGatherDatum list containing pieces velocities as single pairs: [[time, velocity]].

        Returns:
            new CDPGatherDatum with the concatenated velocity.
        """

        # check if datum_list contains only tiles from a single gather
        self._check_datum_list(datum_list)

        velocities = [None] * len(datum_list)

        for i, datum in enumerate(datum_list):
            velocities[i] = datum.velocities[0]

        new_datum = CDPGatherDatum(features=np.empty(0), label=None, offsets=None,
                                   inline=datum_list[0].inline, crossline=datum_list[0].crossline,
                                   velocities=velocities, pixel_depth=0)

        return new_datum

    def __str__(self):
        return f"<ConcatenateVelocities>>"

    @staticmethod
    def _check_datum_list(datum_list: List[CDPGatherDatum]):
        assert len(set([(datum.inline, datum.crossline) for datum in datum_list])) <= 1, \
            f"This datum list contains multiple gathers! Cannot concatenate velocities!"


class FilterVelocities(Transformation):
    def __init__(self, savgol_window: int = 3, savgol_order: int = 1, initial_velocity: Union[float, None] = None,
                 initial_analysis_time_ms: float = None):
        """  Initialize FilterVelocities.

        Args:
            savgol_window: (int) the length of the Savitzky-Golay filter window (window_length in savgol_filter).
            savgol_order: (int) the order of the polynomial used to fit the samples (polyorder in savgol_filter).
            initial_velocity: initial velocity value in m/s to appear in the velocity functions. If None, no
                restriction is applied. Ignored if initial_analysis_time_ms is None.
            initial_analysis_time_ms: (float) time value in ms to consider the analysis, that is, only the values
                adjusted from this point forward will be saved. If None, all time values are considered.
        """

        self.savgol_window = savgol_window
        self.savgol_order = savgol_order
        self.initial_time = initial_analysis_time_ms
        self.initial_velocity = initial_velocity

        if initial_analysis_time_ms is None:
            self.initial_velocity = None

    def __call__(self, datum: CDPGatherDatum) -> CDPGatherDatum:
        """ Filter datum.velocities using savgol algorithm (inplace).

        Args:
            datum: CDPGatherDatum with velocities.

        Returns:
            datum with filtered velocity_fn
        """

        velocities_array = np.stack(datum.velocities)

        new_velocities = savgol_filter(velocities_array[:, 1], self.savgol_window, self.savgol_order)

        initial_time = 0 if self.initial_time is None else self.initial_time

        velocity_fn = [[datum.velocities[i][0], new_velocities[i]] for i in range(len(datum.velocities))
                       if datum.velocities[i][0] > initial_time]

        if self.initial_velocity is not None:
            velocity_fn = [[0, self.initial_velocity]] + velocity_fn

        datum.velocities = velocity_fn

        return datum

    def __str__(self):
        return f"<FilterVelocities savgol window: {self.savgol_window}, savgol order: {self.savgol_order} "


class SmoothVelocities(Transformation):
    def __init__(self, time_range_ms: List[float], sample_rate_ms: int, savgol_window: int, savgol_order: int = 1):
        """  Initialize SmoothVelocities.

        Args:
            time_range_ms: (list) contains the time range in milliseconds to resample the velocity function before
                smoothing; format = [initial_time, final_time].
            sample_rate_ms: (int) sample rate in milliseconds to resample the velocity function before smoothing.
            savgol_window: (int) the length of the Savitzky-Golay filter window (window_length in savgol_filter).
            savgol_order: (int) the order of the polynomial used to fit the samples (polyorder in savgol_filter).
        """

        self.savgol_window = savgol_window
        self.savgol_order = savgol_order
        self.time_range = time_range_ms
        self.sample_rate = sample_rate_ms

    def __call__(self, datum: CDPGatherDatum) -> CDPGatherDatum:
        """ Filter datum.velocities using savgol algorithm (inplace). Before smoothing, resample the velocity function
            considering *self.sample_rate* points in the range *self.time_range*.

        Args:
            datum: CDPGatherDatum with velocities.

        Returns:
            datum with filtered velocity_fn.
        """

        times, velocities = ops.resample_velocity_function(datum.velocities, self.time_range, self.sample_rate)
        new_velocities = savgol_filter(velocities, self.savgol_window, self.savgol_order)

        velocity_fn = [[times[i], new_velocities[i]] for i in range(len(times))]

        datum.velocities = velocity_fn

        return datum

    def __str__(self):
        return f"<SmoothVelocities savgol window: {self.savgol_window}, savgol order: {self.savgol_order} "
