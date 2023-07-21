from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from rockml.data.adapter.seismic.horizon import HorizonDatum
from rockml.data.adapter.seismic.segy.poststack import Direction, Phase, PostStackDatum
from rockml.data.transformations import Transformation


class ConvertHorizon(Transformation):
    def __init__(self, horizon_names: List[str],
                 crop_top: int, crop_left: int,
                 inline_resolution: int, crossline_resolution: int,
                 correction: int = 10):
        """ Initialize ConvertHorizon.

        Args:
            horizon_names: list of horizon names.
            crop_top: (int) how many pixels were cropped on top of all the PostStackDatum.features to be converted.
            crop_left: (int) how many pixels were cropped on top of all the PostStackDatum.features to be converted.
            inline_resolution: (int) separation between inline slices, i.e., difference between adjacent slice numbers.
            crossline_resolution: (int) separation between crossline slices, i.e., difference between adjacent slice
                numbers.
            correction: (int)
        """
        self.horizon_names = horizon_names
        self.crop_top = crop_top
        self.crop_left = crop_left
        self.inline_resolution = inline_resolution
        self.crossline_resolution = crossline_resolution
        self.correction = correction
        self.start_point = [self.correction] * len(self.horizon_names)

    def __call__(self, datum: PostStackDatum) -> List[HorizonDatum]:
        """ Convert the segmentation mask in *datum.label* to a list of HorizonDatum (one for each horizon in the mask).

        Args:
            datum: a PostStackDatum with label field containing a segmentation mask.

        Returns:
            list of HorizonDatum.
        """

        horizons = self._update_point_maps(datum)

        if datum.direction == Direction.INLINE:
            columns = [HorizonDatum.columns.INLINE.value, HorizonDatum.columns.CROSSLINE.value,
                       HorizonDatum.columns.PIXEL_DEPTH.value]
        elif datum.direction == Direction.CROSSLINE:
            columns = [HorizonDatum.columns.CROSSLINE.value, HorizonDatum.columns.INLINE.value,
                       HorizonDatum.columns.PIXEL_DEPTH.value]
        else:
            raise AttributeError('Currently, ConvertHorizon only works with 2D lines.')

        datum_list = []
        for key in horizons:
            df = pd.DataFrame(
                horizons[key],
                columns=columns,
            )
            df = df.astype({columns[0]: 'int32', columns[1]: 'int32', columns[2]: 'int32'})

            # Set the index always as inline/crossline, no matter the original datum direction
            df.set_index([HorizonDatum.columns.INLINE.value, HorizonDatum.columns.CROSSLINE.value], inplace=True)

            datum_list.append(HorizonDatum(df, key))

        return datum_list

    def _update_point_maps(self, datum: PostStackDatum):
        """ This function loops through *datum*.label columns, for each class, to find the points where the category
            changes.

            For example, considering class 0, the function finds the points in each column where there is a
            transition to class 1:

                column = [0, 0, 0, 1, 1, 2, 2, 2] --> transition from 0 to 1 in position 3.

            The result is a list of points of transition for each class, defining the point map for each horizon. A
            point map is a list in the format: [[datum.direction, real_column, pixel depth of the transition], ...].
            If datum.direction = INLINE, datum.line_number = 400, and real_column = 200 (image column, considering the
            original crossline range), for the above column example, the point map for horizon 1 would be [400, 300, 3].

            The output is a dict in the form: {horizon_name: point_map list}

        Args:
            datum: a PostStackDatum with label.

        Returns:
            dict in the form {horizon_name: point_map}
        """

        segy_line = 0
        if datum.direction == Direction.INLINE:
            # start = self.segy_adapter.segy_raw_data.get_range_crosslines()[0]
            segy_line = round(datum.column * self.crossline_resolution)  # + start
        elif datum.direction == Direction.CROSSLINE:
            # start = self.segy_adapter.segy_raw_data.get_range_inlines()[0]
            segy_line = round(datum.column * self.inline_resolution)  # + start

        horizons = {name: [] for name in self.horizon_names}
        mask_t = np.transpose(datum.label)

        for cls, horizon in enumerate(self.horizon_names):
            point_map = [None] * mask_t.shape[0]
            for col, col_data in enumerate(mask_t):
                hrz = np.argmax(col_data[self.start_point[cls] - self.correction:] > cls)

                self.start_point[cls] = max(self.correction,
                                            hrz + self.start_point[cls] - self.correction)
                # Add top crop into consideration
                hrz_depth = self.start_point[cls] + self.crop_top
                real_col = col + segy_line
                point_map[col] = [datum.line_number, real_col, hrz_depth]

            horizons[horizon].extend(point_map)

        return horizons


class ConcatenateHorizon(Transformation):
    def __call__(self, dataset: List[HorizonDatum]) -> List[HorizonDatum]:
        """ Concatenate a list of HorizonDatum that contains several point maps for the same horizon. This function is
            useful when one wants to join the point maps extracted from many PostStackDatum lines in only one
            HorizonDatum per horizon.

        Args:
            dataset: list of HorizonDatum, with repeated horizon names.

        Returns:
            new list HorizonDatum, with only one HorizonDatum per horizon name.
        """
        new_horizons = {
            e.horizon_name: HorizonDatum(
                pd.DataFrame(),
                e.horizon_name,
            ) for e in dataset
        }

        for datum in dataset:
            new_horizons[datum.horizon_name].point_map = pd.concat([new_horizons[datum.horizon_name].point_map,
                                                                    datum.point_map])
        # Order dataframe indexes (by inline, then crossline) for each horizon
        for horizon in new_horizons.values():
            horizon.point_map.sort_index(axis=0, inplace=True)

        return list(new_horizons.values())


class RemoveMutedTraces(Transformation):
    def __init__(self, valid_mask: np.ndarray, segy_info: dict):
        """ Initialize RemoveMutedTraces.

        Args:
            valid_mask: ndarray of booleans with shape = (num_inlines, num_crosslines), indicating the valid traces.
            segy_info: dict with the segy file related to the HorizonDatums to be processed.
        """
        self.valid_mask = valid_mask
        self.segy_info = segy_info

    def __call__(self, dataset: HorizonDatum) -> HorizonDatum:
        """ Remove the (inline, crossline) points where there is no trace available (mute traces). This is done inplace
            in the *dataset*.point_map DataFrame.

        Args:
            dataset: an initialized HorizonDatum.

        Returns:
            modified HorizonDatum.
        """

        mute_indexes = np.argwhere(self.valid_mask == False)
        mute_indexes[:, 0] += self.segy_info['range_inlines'][0]
        mute_indexes[:, 1] += self.segy_info['range_crosslines'][0]
        mute_indexes = [tuple(el) for el in mute_indexes]
        dataset.point_map.drop(mute_indexes, inplace=True, errors='ignore')

        return dataset


class PhaseCorrection(Transformation):
    def __init__(self, segy_info: dict, seismic_lines: List[PostStackDatum],
                 amp_factor: int = 10, mode: Phase = Phase.MAX):
        super(PhaseCorrection, self).__init__()

        self.mode = mode
        self.segy_info = segy_info
        self.seismic_lines = seismic_lines
        self.amp_factor = amp_factor

    def __call__(self, dataset):
        for line in self.seismic_lines:
            expanded_line = self.interpolate_data(line.features)
            x = self.correct_depth(expanded_line, dataset, line.direction, line.line_number)

        return dataset

    def correct_depth(self, poststack: np.ndarray, dataset: List[HorizonDatum],
                      direction: Direction, line_number: int):
        """ Sweep the 2D array *amplitudes* in positions defined by *idx_xline* and *idx_depth* to correct the depth value
            based on the maximum (mode = Phase.MAX) or minimum (mode = Phase.MIN) surrounding amplitude values.

            We iterate across the width direction (in *idx_xline* points) to search for amplitude values in the depth
            surroundings defined by *depth_range*. Example:

            In iteration 0:
              idx_xline[0] = [0]
              idx_depth[0] = [105]
              depth_range = 1

            We go to the position [105, 0] in *amplitudes*. Then we also get the values of positions [104, 0] and [106, 0]
            to get the depth value of the maximum amplitude in these indexes.


        Args:
            poststack: np.ndarray with the raw seismic amplitudes.
            dataset: List[HorizonDatum]with the horizons to be corrected.
            direction: Direction inline or crossline.
            line_number: (int) line number related to the cube.

        Returns:
            np.ndarray of corrected depth values.
        """

        depth_range = round(10 * 1.5)
        new_dataset = [None] * len(dataset)

        for idx, horizon in enumerate(dataset):
            # TODO: why not start an empty DF instead of copying?
            new_dataset[idx] = HorizonDatum(horizon.point_map.copy(), horizon.horizon_name)
            sub_hrz = horizon.point_map.iloc[horizon.point_map.index.get_level_values(str(direction)) == line_number]

            for _, value in sub_hrz.items():
                for (_, xline), depth in value.items():
                    depth = round(depth * self.amp_factor)
                    amp_top = round(max(0, depth - depth_range))
                    amp_bottom = round(min(poststack.shape[0] - 1, depth + depth_range + 1))
                    amps = poststack[amp_top:amp_bottom, xline - self.segy_info['range_crosslines'][0]]

                    if self.mode == Phase.MAX:
                        new_depth = depth + np.argmax(amps) - depth_range
                    elif self.mode == Phase.MIN:
                        new_depth = depth + np.argmin(amps) - depth_range
                    else:
                        raise ValueError('Wrong mode! should be either max or min.')

                    new_dataset[idx].point_map.loc[(line_number, xline), :] = new_depth / self.amp_factor

        return new_dataset

    def interpolate_data(self, seismic_line: np.ndarray) -> np.ndarray:
        """ Resample *image* in height direction using linear interpolation.
            Final shape = (image.shape[0]*AMP_FACTOR, image.shape[1], 1)

        Args:
            seismic_line: np.ndarray representing a seismic image.

        Returns:
            resampled np.ndarray.
        """

        width, height = seismic_line.shape[1], seismic_line.shape[0]
        seismic_line = Image.fromarray(seismic_line[:, :, 0])
        seismic_line = seismic_line.resize(size=(width, height * self.amp_factor), resample=Image.BICUBIC)
        seismic_line = np.array(seismic_line)[:, :, np.newaxis]

        return seismic_line
