""" Copyright 2023 IBM Research. All Rights Reserved.
"""

import os
from enum import Enum
from functools import partial
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from rockml.data.adapter import Datum, DataDumper, BaseDataAdapter
from seisfast.io import PostStackSEGY


class _Columns(Enum):
    """Define HorizonDatum.point_map column names. """

    INLINE = 'inline'
    CROSSLINE = 'crossline'
    PIXEL_DEPTH = 'pixel_depth'
    X = 'x'
    Y = 'y'
    Z = 'z'


class HorizonDatum(Datum):
    """ Establish the structure of a datum for horizon seisfast, which includes horizon point map, direction and
        line_number.
    """

    columns = _Columns

    def __init__(self, point_map: pd.DataFrame, horizon_name: str):
        """ Initialize HorizonDatum.

        Args:
            point_map: pd.DataFrame indexed by inline and crossline, with z as depth information.
            horizon_name: (str) name of the horizon.
        """

        self.point_map = point_map
        self.horizon_name = horizon_name

    def __str__(self):
        return f"<HorizonDatum {self.horizon_name}>"


class HorizonDataDumper(DataDumper):
    """ Class for dumping HorizonDatum list to disk. """

    @staticmethod
    def _pixel_depth_to_z(data_point: pd.Series, time_depth_resolution: float, initial_time: float) -> float:
        """ Converts the pixel depth information in *data_point* to real depth 'z' in milliseconds.

        Args:
            data_point: pd.Series with a single point of the horizon point map (pixel depth information).
            time_depth_resolution: (float) sample rate in milliseconds.
            initial_time: (int) initial time value (in milliseconds).

        Returns:
            the real time/depth value in milliseconds.
        """

        return round(data_point.values[0] * time_depth_resolution + initial_time)

    @staticmethod
    def _compute_xyz(data_point: pd.Series, segy_raw_data: PostStackSEGY,
                     fx: str = "72 x coordinate of ensemble position (cdp)",
                     fy: str = "73 y coordinate of ensemble position (cdp)") -> Tuple[float, float, float]:
        """ Compute the real values for x, y and z for the *data_point*. It converts the inline and crossline numbers to
            the coordinates x and y, and also the pixel depth to the real depth in milliseconds z.

        Args:
            data_point: pd.Series with a single point of the horizon point map.
            segy_raw_data: initialized PostStackSEGY object.
            fx: (str) field name in SEGY header for cdp geo-x; default: "72 x coordinate of ensemble position (cdp)".
            fy: (str) field name in SEGY header for cdp geo-y; default: "72 x coordinate of ensemble position (cdp)".

        Returns:
            (x, y, z)
        """

        initial_inline = segy_raw_data.get_range_inlines()[0]
        initial_crossline = segy_raw_data.get_range_crosslines()[0]
        inline_resolution = segy_raw_data.get_inline_resolution()
        crossline_resolution = segy_raw_data.get_crossline_resolution()
        initial_time_depth = segy_raw_data.get_range_time_depth()[0]
        time_depth_resolution = segy_raw_data.get_time_depth_resolution()

        trace_map = segy_raw_data.get_trace_map()
        file_idx = trace_map[
            int(round((data_point.name[0] - initial_inline) / inline_resolution)),
            int(round((data_point.name[1] - initial_crossline) / crossline_resolution))
        ]
        x, y = segy_raw_data.get_corrected_geo_coordinates(file_idx, fx, fy)
        z = HorizonDataDumper._pixel_depth_to_z(data_point, time_depth_resolution, initial_time_depth)

        return x, y, z

    @staticmethod
    def _convert(segy_raw_data: PostStackSEGY, point_map: pd.DataFrame, include_xy: bool = True) -> pd.DataFrame:
        """ Convert the DataFrame *point_map*

        Args:
            segy_raw_data: initialized PostStackSEGY object.
            point_map: pd.DataFrame with horizon point map (depth information should be in pixel_depth)
            include_xy: (bool) whether x and y columns should be included in the output file.

        Returns:
            processed pd.DataFrame.
        """

        if include_xy:
            columns = {_Columns.X.value: np.float32, _Columns.Y.value: np.float32, _Columns.Z.value: np.float32}
        else:
            columns = {_Columns.Z.value: np.float32}

        if include_xy:
            # If it was already processed
            if _Columns.X.value in point_map.columns:
                return point_map
            fn = partial(HorizonDataDumper._compute_xyz, segy_raw_data=segy_raw_data)
        else:
            # If it was already processed
            if _Columns.Z.value in point_map.columns:
                return point_map

            range_time_depth = segy_raw_data.get_range_time_depth()
            time_depth_resolution = segy_raw_data.get_time_depth_resolution()

            fn = partial(
                HorizonDataDumper._pixel_depth_to_z,
                time_depth_resolution=time_depth_resolution,
                initial_time=range_time_depth[0]
            )

        xyz_df = point_map.apply(fn, axis=1, result_type='expand')

        # When apply generates only one column, the result is a pd.Series; we need to convert it back to pd.DataFrame
        if type(xyz_df) == pd.Series:
            xyz_df = xyz_df.to_frame()

        xyz_df.columns = columns.keys()
        xyz_df = xyz_df.astype(columns)

        return xyz_df

    @staticmethod
    def to_text_file(datum_list: List[HorizonDatum],
                     path: str,
                     segy_path: str,
                     include_xy: bool = True,
                     mode: str = 'a',
                     inline_byte: np.uint8 = 189, crossline_byte: np.uint8 = 193,
                     x_byte: np.uint8 = 181, y_byte: np.uint8 = 185) -> None:
        """ Saves the horizons in the list of HorizonDatum in a text file (xyz). The format is similar to the seisfast
            library: the spaces between columns are smaller.

        Args:
            datum_list: list of HorizonDatum.
            path: (str) path to the directory where to save the output file.
            segy_path: path to the SEGY file related to the horizons.
            include_xy: (bool) whether x and y should be included in the file or not; default = True.
            mode: (str) Python write mode ('w' or 'a'); default = 'a'.
            inline_byte: (np.uint8) SEGY header byte for inlines; field name = "74 3d PostStack inline number"
            crossline_byte: (np.uint8) SEGY header byte for crosslines; field name = "75 3d PostStack crossline number"
            x_byte: (np.uint8) SEGY header byte for cdp geo-x; field name = "72 x coordinate of ensemble position (cdp)"
            y_byte: (np.uint8) SEGY header byte for cdp geo-y; field name = "73 y coordinate of ensemble position (cdp)"
        """

        columns = [_Columns.X.value, _Columns.Y.value, _Columns.Z.value] if include_xy else [_Columns.Z.value]

        # Initialize PostStackSEGY and scan file
        segy_raw_data = PostStackSEGY(segy_path)
        mapping = segy_raw_data.trace_header_mapping()

        segy_raw_data.scan(finline=mapping[inline_byte], fcrossline=mapping[crossline_byte],
                           fx=mapping[x_byte], fy=mapping[y_byte], save_scan=True)

        for datum in datum_list:
            # Copy point map so that original information is kept.
            df = datum.point_map.copy()
            # Convert information to dump the df correctly
            df = HorizonDataDumper._convert(segy_raw_data, df, include_xy)
            df.to_csv(os.path.join(path, f'{datum.horizon_name}.xyz'),
                      sep=' ',
                      columns=columns,
                      float_format='%.2f',
                      mode=mode,
                      header=False)


class HorizonAdapter(BaseDataAdapter):
    """ Specific class for reading and iterating horizon files. A HorizonDatum is the
        iteration element; it is associated with a single horizon file.
    """

    def __init__(self, horizons_path_list: List[str], time_depth_resolution: float, initial_time: int,
                 column_dict: dict = None, separator: str = '\s+'):
        """ Initialize HorizonAdapter class.

        Args:
            horizons_path_list: (list) list containing the paths of the horizons.
            time_depth_resolution: (float) sample rate in milliseconds, as given by the SEGY related to the horizons.
            initial_time: (int) initial time value (in milliseconds), as given by the SEGY related to the horizons.
            column_dict: dict with column names in the horizon file as keys and types as values. If None, the dict is
                set to {'inline': np.int32, 'crossline': np.int32, 'x': np.float32, 'y': np.float32, 'z': np.float32}.
                The columns 'inline', 'crossline' and 'z' are mandatory.
            separator: (str) column separator in the horizon file.
        """

        super(HorizonAdapter, self).__init__()

        self.horizons_paths = horizons_path_list
        self.time_depth_resolution = time_depth_resolution
        self.initial_time = initial_time

        if column_dict is None:
            self.column_dict = {_Columns.INLINE.value: np.int32, _Columns.CROSSLINE.value: np.int32,
                                _Columns.X.value: np.float32, _Columns.Y.value: np.float32,
                                _Columns.Z.value: np.float32}
        else:
            # Checking if the inline, crossline and z columns are in the dictionary.
            if {_Columns.INLINE.value, _Columns.CROSSLINE.value, _Columns.Z.value} not in set(column_dict.keys()):
                raise ValueError(f"Your column_dict should contain at least il, xl, and z, but is has {column_dict}")
            else:
                self.column_dict = column_dict

        self.column_names = list(self.column_dict.keys())
        self.separator = separator

    @staticmethod
    def _get_horizon_file_name(horizon_path: str):
        """  Get the horizon file name from path.

        Args:
            horizon_path: (str) path to a horizon file.

        Returns:
            horizon file name.
        """

        return os.path.splitext(os.path.basename(horizon_path))[0]

    @staticmethod
    def z_to_pixel_depth(data_point: pd.Series, time_depth_resolution: float, initial_time: int) -> int:
        """ Transform a millisecond depth value to a pixel depth value.

        Args:
            data_point: pd.Series with a single point of the horizon point map (real depth information).
            time_depth_resolution: (float) sample rate in milliseconds.
            initial_time: (int) initial time value (in milliseconds).

        Returns:
            the pixel depth value.
        """

        z_px = np.round(((data_point.values[0] - initial_time) / time_depth_resolution)).astype(np.int32)

        return z_px

    def load_horizon(self, horizon_path: str) -> pd.DataFrame:
        """ Load the horizon in *horizon_path* into a pd.DataFrame.

        Args:
            horizon_path: (str) path to the horizon file.

        Returns:
            pd.DataFrame containing the horizon seisfast.
        """

        horizon_df = pd.read_csv(
            horizon_path,
            sep=self.separator, header=None, engine='python',
            names=self.column_names,
            dtype=self.column_dict,
            # error_bad_lines=False,
            # warn_bad_lines=True
        )

        horizon_df.set_index([_Columns.INLINE.value, _Columns.CROSSLINE.value], inplace=True, drop=True)

        # Removing x and y columns if they are present
        horizon_df.drop(columns=[_Columns.X.value, _Columns.Y.value], inplace=True, errors='ignore')

        # Convert Z to pixel depth
        fn = partial(HorizonAdapter.z_to_pixel_depth,
                     time_depth_resolution=self.time_depth_resolution, initial_time=self.initial_time)
        # In this case, the apply returns a Series and not a DataFrame, so we need to convert it back.
        horizon_df = horizon_df.apply(fn, axis=1).to_frame()

        horizon_df.columns = [_Columns.PIXEL_DEPTH.value]

        return horizon_df

    def __len__(self):
        return len(self.horizons_paths)

    def __iter__(self) -> HorizonDatum:
        for horizon_path in self.horizons_paths:
            horizon_df = self.load_horizon(horizon_path)
            yield HorizonDatum(horizon_df, self._get_horizon_file_name(horizon_path))

    def __getitem__(self, key: Union[int, slice, list]) -> Union[HorizonDatum, List[HorizonDatum]]:
        """ Returns a HorizonDatum if *key* is int and a HorizonDatum list if *key* is either a slice or a list.

        Args:
            key: int, slice or list.

        Returns:
            HorizonDatum or list of HorizonDatum.
        """
        if type(key) == int:
            horizon_path = self.horizons_paths[key]
            horizon_df = self.load_horizon(horizon_path)
            return HorizonDatum(horizon_df, self._get_horizon_file_name(horizon_path))
        elif type(key) == list:
            temp = []
            for idx in key:
                temp += [self[idx]]
            return temp
        elif type(key) == slice:
            temp = []
            start = key.start if key.start else 0
            stop = key.stop if key.stop is not None else len(self)
            step = key.step if key.step is not None else 1
            for idx in range(start, stop, step):
                temp += [self[idx]]
            return temp
        else:
            raise KeyError("This method only supports integer, list or slice.")
