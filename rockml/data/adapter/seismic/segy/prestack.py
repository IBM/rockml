""" Copyright 2023 IBM Research. All Rights Reserved.

    - PreStack segy adapter and datum definition
"""

from enum import Enum
from typing import List, Tuple, Union, TextIO

import h5py
import numpy as np
from rockml.data.adapter import BaseDataAdapter, Datum, DataDumper, FEATURE_NAME, LABEL_NAME
from seisfast.io.seismic import PreStackSEGY
from seisfast.io.velocity import read

VELOCITY_FORMAT = List[List[Union[int, float]]]


class _CDPProps(Enum):
    """Define CDP Gather exclusive properties names. """

    OFFSETS = 'offsets'
    INLINE = 'inline'
    CROSSLINE = 'crossline'
    PIXEL_DEPTH = 'pixel_depth'
    COHERENCE = 'coherence'
    VELOCITIES = 'velocities'


class CDPGatherDatum(Datum):
    # CDP Gather exclusive names
    names = _CDPProps

    def __init__(self, features: np.ndarray, offsets: Union[np.ndarray, None],
                 label: Union[np.ndarray, int, float, None], inline: int, crossline: int, pixel_depth: int,
                 coherence: Union[np.ndarray, float, None] = None, velocities: Union[VELOCITY_FORMAT, None] = None):
        """ Initialize CDPGatherDatum class.

        Args:
            features: 2D numpy array of containing traces from a CDP gather.
            offsets: 1D numpy array of offset seisfast.
            label: label that characterizes these features for a particular task (can be a float, an int or a ndarray).
            inline: (int) inline number.
            crossline: (int) crossline number.
            pixel_depth: (int) index of the upper pixel of *features* in seismic slice *line*.
            coherence: ndarray containing coherence values calculated for the datum; leave it None if this information
                is not available or not calculated.
            velocities: list of pairs with velocities associated with the gather; leave it None if this information
                is not available or not calculated.
        """

        self.features = features
        self.offsets = offsets
        self.label = label
        self.inline = inline
        self.crossline = crossline
        self.pixel_depth = pixel_depth
        self.coherence = coherence
        self.velocities = velocities

    def __str__(self):
        return f"<CDPGatherDatum ({self.inline},{self.crossline}), pixel_depth: {self.pixel_depth}>"


class CDPGatherDataDumper(DataDumper):
    """ Class for dumping CDPGatherDatum list to disk. """

    @staticmethod
    def to_hdf(datum_list: List[CDPGatherDatum], path: str, save_list: List[str] = None):
        """ Wrapper to save the list of CDPGatherDatum in a hdf5 file. The default behavior is to save all the
            information available. If the user wants to save a subset of the names, pass it to the list *save_list*
            with the correct names. For example, to save only CDPGatherDatum.coherence, pass save_list = ['coherence'].

        Args:
            datum_list: list of CDPGatherDatum.
            path: (str) path where to save the hdf5 file.
            save_list: list of strings containing CDPGatherDatum names to save. If None, all information is saved.

        """

        valid_names = [FEATURE_NAME, LABEL_NAME] + [n.value for n in CDPGatherDatum.names]

        if save_list is None:
            valid_list = valid_names
        else:
            # Check values in list
            valid_list = [name for name in save_list if name in valid_names]
            if len(valid_list) != len(save_list):
                raise ValueError(f"Invalid names were given! {[name for name in save_list if name not in valid_names]}")

        h5f = h5py.File(path, 'w')

        for name in valid_list:
            h5f.create_dataset(name, data=np.stack([getattr(datum, name) for datum in datum_list], axis=0), chunks=True)

        h5f.close()

    @staticmethod
    def write_velocity_function(file: TextIO, il: int, xl: int, velocity_fn: VELOCITY_FORMAT):
        """ Writes velocity function.

        Args:
            file: file object.
            il: (int) inline number.
            xl: (int) crossline number.
            velocity_fn: list of pairs in the form [[time_value, velocity_value], ...].
        """

        if velocity_fn is None:
            return

        space = "     "

        file.write(f"VFUNC {space}{il}{space}{xl}\n")

        count = 0

        for pair in velocity_fn:
            file.write(f"{int(pair[0])}{space}{int(pair[1])}{space}")
            count += 1
            if count == 4:
                file.write("\n")
                count = 0

        if count != 0:
            file.write("\n")

        return

    @staticmethod
    def to_velocity_file(datum_list: List[CDPGatherDatum], path: str):
        """ Saves the velocities in each datum in the list of CDPGatherDatum in a text file. The format is the same of
            the seisfast library.

        Args:
            datum_list: list of CDPGatherDatum.
            path: (str) path where to save the velocity text file.
        """

        with open(path, "w") as file:
            for datum in datum_list:
                print(datum)
                if datum is not None and datum.velocities is not None:
                    CDPGatherDataDumper.write_velocity_function(file, datum.inline, datum.crossline, datum.velocities)


class CDPGatherAdapter(BaseDataAdapter):
    """ Specific class for reading and iterating CDP gathers in Pre Stack segy files. A CDPGatherDatum is the
        iteration element; it is associated with a single CDP gather for a (inline, crossline) position.
    """

    def __init__(self, segy_path: str, gather_list: Union[List[List[int]], None], velocity_file_path: Union[str, None],
                 inline_byte: np.uint8 = 189, crossline_byte: np.uint8 = 193, x_byte: np.uint8 = 181,
                 y_byte: np.uint8 = 185, source_byte: np.uint8 = 17, recx_byte: np.uint8 = 81,
                 recy_byte: np.uint8 = 85):
        """ Initialize CDPGatherAdapter class.

        Args:
            segy_path: (str) path to the SEGY file.
            gather_list: list with gather positions in the form [[il_num1, xl_num1], [il_num2, xl_num2]].
            velocity_file_path: (str) path to the velocity file. If None, no velocity information will be loaded.
            inline_byte: (np.uint8) SEGY header byte for inline; field name = "74 3d PostStack inline number".
            crossline_byte: (np.uint8) SEGY header byte for crossline; field name: "75 3d PostStack crossline number".
            x_byte: (np.uint8) SEGY header byte for cdp geo-x; field name: "72 x coordinate of ensemble position (cdp)".
            y_byte: (np.uint8) SEGY header byte for cdp geo-y; field name: "73 y coordinate of ensemble position (cdp)".
            source_byte: (np.uint8) SEGY header byte for source id; field name: "05 energy source point number".
            recx_byte: (np.uint8) SEGY header byte for receiver geo-y; field name: "24 group coordinate x".
            recy_byte: (np.uint8) SEGY header byte for receiver geo-y; field name: "25 group coordinate y".
        """

        super(CDPGatherAdapter, self).__init__()

        self.gather_list = gather_list if gather_list is not None else list()
        self.segy_path = segy_path
        self.velocity_path = velocity_file_path
        self._segy_raw_data = None
        self.segy_info = None

        self.inline_byte = inline_byte
        self.crossline_byte = crossline_byte
        self.x_byte = x_byte
        self.y_byte = y_byte
        self.source_byte = source_byte
        self.recx_byte = recx_byte
        self.recy_byte = recy_byte

        if self.velocity_path is not None:
            self.velocities = read(self.velocity_path)
        else:
            self.velocities = None

    @property
    def segy_raw_data(self) -> PreStackSEGY:
        if self._segy_raw_data is None:
            self._segy_raw_data = PreStackSEGY(self.segy_path)
            self.segy_info = self._segy_raw_data.scan(save_scan=True)
        return self._segy_raw_data

    def initial_scan(self) -> dict:
        """ Perform a scan without initializing self._segy_raw_data and self.segy_info. This is
            important if the adapter is going to be used in parallel later.

        Returns:
            segy_info dict.
        """
        segy_raw_data = PreStackSEGY(self.segy_path)
        mapping = segy_raw_data.trace_header_mapping()

        return segy_raw_data.scan(finline=mapping[self.inline_byte],
                                  fcrossline=mapping[self.crossline_byte],
                                  fx=mapping[self.x_byte],
                                  fy=mapping[self.y_byte],
                                  fsource=mapping[self.source_byte],
                                  frecx=mapping[self.recx_byte],
                                  frecy=mapping[self.recy_byte],
                                  save_scan=True)

    def _get_ordered_cdp_gather(self, il: int, xl: int) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        """ Read cdp gather from the SEGY file *self.segy_path* in position defined by (*il*, *xl*). The collected seisfast
            is then ordered by offset (also read from the file). The output is the ordered gather array along with the
            corresponding offset values.

            Notice that if a valid (*il*, *xl*) position has an empty gather, the function returns (None, None).

        Args:
            il: (int) inline number.
            xl: (int) crossline number

        Returns:
            gather_img: 2D np.ndarray of gather seisfast ordered by offset.
            offsets: 1D np.ndarray of corresponding offsets as extrated from the SEGY file.
        """

        gather = self.segy_raw_data.get_cdp_gather(il, xl)

        # Empty gather
        if len(gather) == 0:
            return None, None

        gather_img = np.zeros((self.segy_raw_data.trace_samples(), len(gather)))

        distances = np.zeros(len(gather), dtype=int)

        for k in range(len(gather)):
            th = self.segy_raw_data.trace_header(gather[k])
            # sx = th["22 source coordinate x"]
            # sy = th["23 source coordinate y"]
            # gx = th["24 group coordinate x"]
            # gy = th["25 group coordinate y"]
            # distances.append(np.sqrt((sx-gx)**2 + (sy-gy)**2))
            # distances.append(th["02 trace sequence number within segy file"])
            distances[k] = (th["12 distance from center of source point to the center of the receiver group"])

        idx = np.argsort(distances, kind='mergesort')
        gather = np.asarray(gather)[idx]

        distances = distances[idx]

        for k in range(len(gather)):
            gather_img[:, k] = self.segy_raw_data.trace_data(gather[k])

        return gather_img, distances

    def _get_velocity_for_gather(self, il: int, xl: int) -> Union[np.ndarray, None]:
        """ Get the velocities for the gather in (*il*, *xl*) if a velocity file is available.

        Args:
            il: (int) inline number.
            xl: (int) crossline number

        Returns:
            velocities for the respective pair (*il*, *xl*).
        """

        if self.velocities is not None:
            return self.velocities.get(f"{il}_{xl}")
        else:
            return None

    def __len__(self):
        return len(self.gather_list)

    def __iter__(self) -> CDPGatherDatum:
        for (inline, crossline) in self.gather_list:
            traces, offsets = self._get_ordered_cdp_gather(inline, crossline)
            if traces is None:
                yield None
            else:
                velocity = self._get_velocity_for_gather(inline, crossline)
                yield CDPGatherDatum(traces, offsets, None, inline, crossline, pixel_depth=0,
                                     coherence=None, velocities=velocity)

    def __getitem__(self, key: Union[int, slice, list]) -> Union[CDPGatherDatum, List[CDPGatherDatum], None]:
        """ Return a CDPGatherDatum if *key* is int and a CDPGatherDatum list if *key* is either a slice or a list.

        Args:
            key: int, slice or list.

        Returns:
            CDPGatherDatum, list of CDPGatherDatum or None.
        """
        if type(key) == int:
            inline, crossline = self.gather_list[key]
            traces, offsets = self._get_ordered_cdp_gather(inline, crossline)
            if traces is None:
                return None
            else:
                velocity = self._get_velocity_for_gather(inline, crossline)
                return CDPGatherDatum(traces, offsets, None, inline, crossline, pixel_depth=0,
                                      coherence=None, velocities=velocity)
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
