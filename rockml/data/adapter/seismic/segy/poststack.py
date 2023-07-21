""" Copyright 2019 IBM Research. All Rights Reserved.

    - PostStack segy adapter and datum definition.
"""

from enum import Enum
from typing import List, Tuple, Union

import h5py
import numpy as np
from rockml.data.adapter import BaseDataAdapter, Datum, DataDumper, FEATURE_NAME, LABEL_NAME
from seisfast.io.horizon import Reader
from seisfast.io.seismic import PostStackSEGY


class Direction(Enum):
    INLINE = 'inline'
    CROSSLINE = 'crossline'
    BOTH = 'both'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)


class Phase(Enum):
    MIN = 'min'
    MAX = 'max'
    CROSSUP = 'cross_up'
    CROSSDOWN = 'cross_down'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)


class _PostStackProps(Enum):
    """Define PostStack exclusive properties names. """

    DIRECTION = 'direction'
    LINE_NUMBER = 'line_number'
    PIXEL_DEPTH = 'pixel_depth'
    COLUMN = 'column'


class PostStackDatum(Datum):
    """ Establish the structure of a datum for seismic seisfast, which includes features, label and info. """

    # PostStack exclusive names
    names = _PostStackProps

    def __init__(self, features: np.ndarray, label: Union[np.ndarray, int, float], direction: Direction,
                 line_number: int, pixel_depth: int, column: int):
        """ Initialize PostStackDatum class.

        Args:
            features: ndarray containing the features (seismic line seisfast).
            label: label that characterizes this set of features (can be a single float, an int or a ndarray in the
                same shape of features).
            direction: one of Direction.INLINE or Direction.CROSSLINE.
            line_number: (int) number of the seismic slice.
            pixel_depth: (int) idx [0] (height direction) of the upper left pixel of *features* in seismic slice *line*.
            column: (int) idx [1] (width direction) of the upper left pixel of *features* in seismic slice *line*. This
                value is expected to take into consideration the geometry of the cube, e.g. if the line range is
                [300 500] it should start in 300 not zero.
        """

        self.features = features
        self.label = label
        self.direction = direction
        self.line_number = int(line_number)
        self.pixel_depth = int(pixel_depth)
        self.column = int(column)

    # TODO add test here
    def __delete__(self, instance):
        del self.features
        del self.label
        del self.direction
        del self.line_number
        del self.pixel_depth
        del self.column

    def __str__(self):
        return f"<PostStackDatum {self.direction}, " \
               f"line number: {self.line_number}, " \
               f"pixel depth: {self.pixel_depth}, column: {self.column}>"


class PostStackDataDumper(DataDumper):
    """ Class for dumping PostStackDatum list to disk. """

    @staticmethod
    def to_hdf(datum_list: List[PostStackDatum], path: str):
        """ Wrapper to save the list of PostStackDatum in a hdf5 file.

        Chunked Storage

            What if there were some way to express this in advance? Isn’t there a way to
            preserve the shape of the seisfast, which is semantically important, but tell HDF5
            to optimize the seisfast for access in 64×64 pixel blocks?

            That’s what chunking does in HDF5. It lets you specify the N-dimensional “shape”
            that best fits your access pattern. When the time comes to write seisfast to disk,
            HDF5 splits the seisfast into “chunks” of the specified shape, flattens them, and
            writes them to disk. The chunks are stored in various places in the file and
            their coordinates are indexed by a B-tree.

            Here’s an example. Let’s take the (100, 480, 640)-shape seisfast just shown and
            tell HDF5 to store it in chunked format. We do this by providing a new keyword,
            chunks, to the create_dataset method:

            ``` python
            dset = f.create_dataset('chunked', (100,480,640), dtype='i1', chunks=(1,64,64))
            ```

            https://www.oreilly.com/library/view/python-and-hdf5/9781491944981/ch04.html

        Args:
            datum_list: list of PostStackDatum.
            path: (str) path where to save the hdf5 file.
        """

        h5f = h5py.File(path, 'w')
        h5f.create_dataset(FEATURE_NAME,
                           data=np.stack([datum.features for datum in datum_list], axis=0), chunks=True)
        h5f.create_dataset(LABEL_NAME,
                           data=np.stack([datum.label for datum in datum_list], axis=0), chunks=True)
        h5f.create_dataset(PostStackDatum.names.DIRECTION.value,
                           data=np.asarray([datum.direction for datum in datum_list], dtype=np.string_), chunks=True)
        h5f.create_dataset(PostStackDatum.names.LINE_NUMBER.value,
                           data=np.stack([datum.line_number for datum in datum_list], axis=0), chunks=True)
        h5f.create_dataset(PostStackDatum.names.PIXEL_DEPTH.value,
                           data=np.stack([datum.pixel_depth for datum in datum_list], axis=0), chunks=True)
        h5f.create_dataset(PostStackDatum.names.COLUMN.value,
                           data=np.stack([datum.column for datum in datum_list], axis=0), chunks=True)
        h5f.close()

    @staticmethod
    def to_dict(datum_list: List[PostStackDatum]) -> dict:
        dataset = dict()
        dataset[FEATURE_NAME] = np.stack([datum.features for datum in datum_list], axis=0)
        dataset[LABEL_NAME] = np.stack([datum.label for datum in datum_list], axis=0)
        dataset[PostStackDatum.names.DIRECTION.value] = np.asarray(
            [datum.direction for datum in datum_list])
        dataset[PostStackDatum.names.LINE_NUMBER.value] = np.stack(
            [datum.line_number for datum in datum_list], axis=0)
        dataset[PostStackDatum.names.PIXEL_DEPTH.value] = np.stack(
            [datum.pixel_depth for datum in datum_list], axis=0)
        dataset[PostStackDatum.names.COLUMN.value] = np.stack(
            [datum.column for datum in datum_list], axis=0)

        return dataset


class PostStackAdapter2D(BaseDataAdapter):
    """ Specific class for reading and iterating seismic seisfast in segy files and horizon files. A PostStackDatum is the
        iteration element; it is associated with a single seismic image and its corresponding mask.
    """

    def __init__(self, segy_path: str, horizons_path_list: List[str], data_dict: Union[dict, None],
                 inline_byte: np.uint8 = 189, crossline_byte: np.uint8 = 193,
                 x_byte: np.uint8 = 181, y_byte: np.uint8 = 185):
        """ Initialize PostStackAdapter2D class.

        Args:
            segy_path: (str) path to the SEGY file.
            horizons_path_list: (list) list containing the paths of the horizons. Notice that
                the elements MUST be depth-wise sorted, i.e., youngest to oldest.
            data_dict: dict containing the ranges of seismic lines to be included.
            inline_byte: (np.uint8) SEGY header byte for inlines; field name = "74 3d PostStack inline number"
            crossline_byte: (np.uint8) SEGY header byte for crosslines; field name = "75 3d PostStack crossline number"
            x_byte: (np.uint8) SEGY header byte for cdp geo-x; field name = "72 x coordinate of ensemble position (cdp)"
            y_byte: (np.uint8) SEGY header byte for cdp geo-y; field name = "73 y coordinate of ensemble position (cdp)"
        """
        super(PostStackAdapter2D, self).__init__()
        self.data_dict = data_dict if data_dict is not None else dict()
        self.expanded_list = self._create_lists()
        self.segy_path = segy_path
        self._segy_raw_data = None
        self.segy_info = None
        self.horizons_data = horizons_path_list

        self.inline_byte = inline_byte
        self.crossline_byte = crossline_byte
        self.x_byte = x_byte
        self.y_byte = y_byte

    @property
    def segy_raw_data(self) -> PostStackSEGY:
        if self._segy_raw_data is None:
            self._segy_raw_data = PostStackSEGY(self.segy_path)
            self.segy_info = self._segy_raw_data.scan(save_scan=True)
        return self._segy_raw_data

    def initial_scan(self) -> dict:
        """ Perform a scan without initializing self._segy_raw_data and self.segy_info. This is
            important if the adapter is going to be used in parallel later.

        Returns:
            segy_info dict.
        """
        segy_raw_data = PostStackSEGY(self.segy_path)
        mapping = segy_raw_data.trace_header_mapping()

        return segy_raw_data.scan(finline=mapping[self.inline_byte],
                                  fcrossline=mapping[self.crossline_byte],
                                  fx=mapping[self.x_byte],
                                  fy=mapping[self.y_byte],
                                  save_scan=True)

    def _create_lists(self) -> List[Tuple[Direction, int]]:
        """ Create list of tuples containing the direction name and the slice number, according
            to the ranges specified in *self.data_dict*.

        Returns:
            list of tuples in the form (direction_name, slice_number).
        """
        expanded_list = []

        # Checking if data_dict keys are correct before creating the expanded list.
        assert set(self.data_dict.keys()) <= {Direction.INLINE.value, Direction.CROSSLINE.value}, \
            f"Invalid key in data_dict! Please use the values {Direction.INLINE.value} or {Direction.CROSSLINE.value}"

        for key, row_list in self.data_dict.items():
            for row in row_list:
                expanded_list.extend([(Direction(key), e) for e in list(range(row[0], row[1]))])

        assert len(expanded_list) == len(set(expanded_list)), f'data_dict yaml file contains overlapping ranges.'

        return expanded_list

    def _get_raw_line(self, direction: Direction, line_number: int) -> np.ndarray:
        """ Get inline or crossline slice defined by *line_number* as a 3D numpy array (with channel axis added on the
            last dimension).

        Args:
            direction: one of Direction.INLINE or Direction.CROSSLINE.
            line_number: int indicating slice number.

        Returns:
            3D np.ndarray.
        """
        if direction == Direction.INLINE:
            return self.segy_raw_data.get_inline_section(line_number)[..., np.newaxis].astype(np.float32)
        if direction == Direction.CROSSLINE:
            return self.segy_raw_data.get_crossline_section(line_number)[..., np.newaxis].astype(np.float32)

        raise ValueError(f'Invalid direction option: {direction}!')

    def _get_horizons_for_line(self, direction: Direction, line_number: int) -> np.ndarray:
        """ Construct a 2D numpy array mask from the *self.horizons_data* for line
            *line_number* of *direction*. The mask contains the horizon lines (with one pixel
            of thickness) ordered by depth. The other pixels of the mask are all set to 0.

        Args:
            direction: one of Direction.INLINE or Direction.CROSSLINE.
            line_number: int indicating slice number.

        Returns:
            2D uint8 array containing the mask of line *line_number* of *direction*.
        """
        width = 0
        other_resolution = 0
        if direction == Direction.INLINE:
            width = self.segy_raw_data.get_num_crosslines()
            other_resolution = self.segy_raw_data.get_crossline_resolution()
        elif direction == Direction.CROSSLINE:
            width = self.segy_raw_data.get_num_inlines()
            other_resolution = self.segy_raw_data.get_inline_resolution()

        mask = np.zeros((self.segy_raw_data.get_num_samples(), width), dtype=np.uint8)
        hor_label = 0
        for horizon in self.horizons_data:
            horizon_reader = Reader(horizon)
            point_map = horizon_reader.get_point_map(save=True)

            if direction == Direction.INLINE:
                horizon_range = horizon_reader.get_range_inlines()
                horizon_other_range = horizon_reader.get_range_crosslines()
            elif direction == Direction.CROSSLINE:
                horizon_range = horizon_reader.get_range_crosslines()
                horizon_other_range = horizon_reader.get_range_inlines()

            hor_label += 1

            il_hor_idx = int(line_number - horizon_range[0])

            if il_hor_idx < 0:
                continue

            if direction == Direction.INLINE:
                if il_hor_idx > (point_map.shape[0] - 1):
                    continue
                line = point_map[il_hor_idx, :]
            elif direction == Direction.CROSSLINE:
                if il_hor_idx > (point_map.shape[1] - 1):
                    continue
                line = point_map[:, il_hor_idx]

            for l in range(line.shape[0]):

                if line[l] < 0:
                    continue

                # translates from horizon coordinates to seismic coordinates
                # this is necessary since the horizon may not defined for the whole seismic
                xl_hor_idx = int(l + horizon_other_range[0])
                xl_seis_idx = int((xl_hor_idx - horizon_other_range[0]) / other_resolution)

                # Protection when annotation goeas outside of the cube
                if xl_seis_idx >= mask.shape[1]:
                    break

                h = int((line[l] - self.segy_raw_data.get_range_time_depth()[
                    0]) / self.segy_raw_data.get_time_depth_resolution())

                mask[h:h + 1, xl_seis_idx] = hor_label

        return mask

    def get_line(self, direction: Direction, line_number: int) -> PostStackDatum:
        """ Create a PostStackDatum with a seismic line, its corresponding mask and info.

        Args:
            direction: one of Direction.INLINE or Direction.CROSSLINE.
            line_number: int indicating slice number.

        Returns:
            PostStackDatum for line *line_number* of *direction*.
        """
        image = self._get_raw_line(direction, line_number)
        mask = self._get_horizons_for_line(direction, line_number)
        return PostStackDatum(image, mask, **self.parse_info(direction, line_number))

    def parse_info(self, direction: Direction, line_number: int) -> dict:
        """ Create a dict with the info of the direction (*key*), line (*line_number*), pixel depth and pixel column.
            Both pixel depth and column are = 0 in the case of raw seismic images.

        Args:
            direction: one of Direction.INLINE or Direction.CROSSLINE.
            line_number: int indicating slice number.

        Returns:
            dict in the form {'direction': *key*, 'line_number': *line_number*, 'pixel_depth': 0, 'pixel_column': 0}.
        """

        if direction == Direction.INLINE:
            line_correction = self.segy_info['range_crosslines'][0]
        elif direction == Direction.CROSSLINE:
            line_correction = self.segy_info['range_inlines'][0]
        else:
            raise ValueError('Invalid key representing direction.')

        return {PostStackDatum.names.DIRECTION.value: direction,
                PostStackDatum.names.LINE_NUMBER.value: int(line_number),
                PostStackDatum.names.PIXEL_DEPTH.value: int(0),
                PostStackDatum.names.COLUMN.value: int(line_correction)}

    def __len__(self):
        return len(self.expanded_list)

    def __iter__(self) -> PostStackDatum:
        for (direction, line_number) in self.expanded_list:
            yield self.get_line(direction, line_number)

    def __getitem__(self, key: Union[int, slice, list]) -> Union[PostStackDatum, List[PostStackDatum]]:
        """ Returns a PostStackDatum if *key* is int and a PostStackDatum list if *key* is either a slice or a list.

        Args:
            key: int, slice or list.

        Returns:
            PostStackDatum or list of PostStackDatum.
        """
        if type(key) == int:
            direction, line = self.expanded_list[key]
            return self.get_line(direction, line)
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


class PostStackAdapter3D(BaseDataAdapter):
    def __len__(self):
        pass

    def __iter__(self) -> PostStackDatum:
        pass

    def __getitem__(self, key: Union[int, slice, list]) -> Union[PostStackDatum, List[PostStackDatum]]:
        pass
