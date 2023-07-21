# -*- coding: utf-8 -*-

"""
Module: RockML
author: daniela.szw@ibm.com, rosife@br.ibm.com, sallesd@br.ibm.com
copyright: IBM Confidential
copyright: OCO Source Materials
copyright: © IBM Corp. All Rights Reserved
date: 2020
IBM Certificate of Originality
"""


import codecs
import json as jsonlib
import os
import pickle
import struct
import sys

import numpy as np


def _num_bytes(format_code):
    if format_code == 3:
        return 2
    elif format_code == 8:
        return 1
    else:
        return 4


def _ibm2ieee(ibm):
    """
    Converts an IBM floating point number into IEEE format.
    :param: ibm - 32 bit unsigned integer: unpack('>L', f.read(4))
    """
    if ibm == 0:
        return 0.0
    sign = ibm >> 31 & 0x01
    exponent = ibm >> 24 & 0x7f
    mantissa = (ibm & 0x00ffffff) / float(pow(2, 24))
    return (1 - 2 * sign) * mantissa * pow(16, exponent - 64)


class BaseSEGY:
    """
    Base class to read SEGY files.
    """

    file = None
    file_obj = None
    file_mode = None
    encoding = "cp500"
    byteorder = "big"
    file_size = 0
    num_samples = 0
    num_traces = 0
    format_code = 0
    measurement_system = 1
    fixed_length_trace_flag = 1
    trace_identification_code = 1
    coordinate_units = 1

    # %% initialization

    def __init__(self, filename, num_traces=None, num_samples=None, format_code=None):
        """
        Sets the filename and initializes variables.

        An existing file is open automatically in read mode. Otherwise, it is open in write mode.
        """

        self.file = filename
        self.encoding = "cp500"
        self.byteorder = "big"
        self.file_size = 0
        self.num_samples = 0
        self.num_traces = 0
        self.format_code = 0
        self.measurement_system = 1
        self.fixed_length_trace_flag = 1
        self.trace_identification_code = 1
        self.coordinate_units = 1

        self.range_inlines = None
        self.res_inline = None
        self.range_crosslines = None
        self.res_crossline = None
        self.range_time_depth = None
        self.res_time_depth = None

        self.range_x = None
        self.range_y = None

        self.scan_info = {}

        if os.path.isfile(self.file):

            self.file_obj = open(self.file, "rb")

            self.file_mode = "read"

            statinfo = os.stat(self.file)

            self.file_size = statinfo.st_size

            self.num_samples = self.trace_samples()

            self.format_code = self.fmt_code()

            self.num_traces = int((self.file_size - 3600) / (
                    240 + self.num_samples * _num_bytes(self.format_code)))

        else:

            self.file_obj = open(self.file, "ab")

            self.file_mode = "write"

            self.num_traces = num_traces

            self.num_samples = num_samples

            self.format_code = format_code

            self.file_size = 3600 + (
                    240 + self.num_samples * _num_bytes(self.format_code)) * self.num_traces

        return

    def __del__(self):
        """
        Closes the file before destruction.
        """

        if self.file_obj is not None:
            self.file_obj.close()

    def set_encoding(self, v):
        """
        Sets the enconding. Usually 'cp500' for EBCDIC and 'ascii' for ASCII. Default: cp500

        For more info: https://docs.python.org/2/library/codecs.html
        """

        self.encoding = v

        return

    def set_byteorder(self, v):
        """
        Sets the byteorder. Should be 'big' or 'little'. Default: big
        """

        self.byteorder = v

        return

    def set_format_code(self, v):
        """
        Sets the format code. Usually 3 for integers and 1 for floats.
        """

        self.format_code = v

        return

    def set_measurement_system(self, v):
        """
        Sets the measurement system. Usually 1.
        """

        self.measurement_system = v

        return

    def set_fixed_length_trace_flag(self, v):
        """
        Sets the fixed-length trace flag. Usually 1.
        """

        self.fixed_length_trace_flag = v

        return

    def set_trace_identification_code(self, v):
        """
        Sets the trace identification code. Usually 1.
        """

        self.trace_identification_code = v

        return

    def set_coordinate_units(self, v):
        """
        Sets the coordinate units. Usually 1.
        """

        self.coordinate_units = v

        return

    def set_rangex(self, range_x):
        """
        Sets the range of geo-x coordinates.
        """
        self.range_x = range_x

        return

    def set_range_y(self, range_y):
        """
        Sets the range of geo-y coordinates.
        """
        self.range_y = range_y

        return

    def set_range_inlines(self, range_inlines):
        """
        Sets the range of inlines.
        """
        self.range_inlines = range_inlines

        return

    def set_range_crosslines(self, range_crosslines):
        """
        Sets the range of crosslines.
        """
        self.range_crosslines = range_crosslines

        return

    def set_inline_resolution(self, res_inline):
        """
        Sets inline resolution.
        """
        self.res_inline = res_inline

        return

    def set_crossline_resolution(self, res_crossline):
        """
        Sets crossline resolution.
        """
        self.res_crossline = res_crossline

        return

    # %% read

    def fmt_code(self):
        """
        Reads format code.
        """

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        f = self.file_obj

        f.seek(3224)

        fc = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)

        return fc

    def textual_header(self):
        """
        Reads textual header.
        """

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        f = self.file_obj

        f.seek(0)

        th = codecs.decode(f.read(3200), self.encoding)

        return th

    def binary_header(self):
        """
        Reads binary header.
        """

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        f = self.file_obj

        f.seek(3200)

        bh = {}

        bh["01 job identification number"] = int.from_bytes(f.read(4), byteorder=self.byteorder,
                                                            signed=True)
        bh["02 line number"] = int.from_bytes(f.read(4), byteorder=self.byteorder, signed=True)
        bh["03 reel number"] = int.from_bytes(f.read(4), byteorder=self.byteorder, signed=True)
        bh["04 number of seisfast traces per ensemble"] = int.from_bytes(f.read(2),
                                                                     byteorder=self.byteorder,
                                                                     signed=True)
        bh["05 number of auxiliary traces per ensemble"] = int.from_bytes(f.read(2),
                                                                          byteorder=self.byteorder,
                                                                          signed=True)
        bh["06 sample interval (us)"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                       signed=True)
        bh["07 sample interval of original field recording (us)"] = int.from_bytes(f.read(2),
                                                                                   byteorder=self.byteorder,
                                                                                   signed=True)
        bh["08 number of samples per seisfast trace"] = int.from_bytes(f.read(2),
                                                                   byteorder=self.byteorder,
                                                                   signed=True)
        bh["09 number of samples per seisfast trace for original field recording"] = int.from_bytes(
            f.read(2), byteorder=self.byteorder, signed=True)
        bh["10 seisfast sample format code"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                          signed=True)
        bh["11 ensemble fold"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                signed=True)
        bh["12 trace sorting code"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                     signed=True)
        bh["13 vertical sum code"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                    signed=True)
        bh["14 sweep frequency at start (Hz)"] = int.from_bytes(f.read(2),
                                                                byteorder=self.byteorder,
                                                                signed=True)
        bh["15 sweep frequency at end (Hz)"] = int.from_bytes(f.read(2),
                                                              byteorder=self.byteorder,
                                                              signed=True)
        bh["16 sweep length"] = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
        bh["17 sweep type code"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                  signed=True)
        bh["18 trace number of sweep channel"] = int.from_bytes(f.read(2),
                                                                byteorder=self.byteorder,
                                                                signed=True)
        bh["19 sweep trace taper length at start (ms)"] = int.from_bytes(f.read(2),
                                                                         byteorder=self.byteorder,
                                                                         signed=True)
        bh["20 sweep trace taper length at end (ms)"] = int.from_bytes(f.read(2),
                                                                       byteorder=self.byteorder,
                                                                       signed=True)
        bh["21 taper type"] = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
        bh["22 correlated seisfast traces"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                         signed=True)
        bh["23 binary gain recovered"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                        signed=True)
        bh["24 amplitude recovery method"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                            signed=True)
        bh["25 measurement system"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                     signed=True)
        bh["26 impulse signal polarity"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                          signed=True)
        bh["27 vibratory polarity code"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                          signed=True)

        bh["28 unassigned"] = int.from_bytes(f.read(240), byteorder=self.byteorder, signed=True)

        bh["29 segy format revision number"] = int.from_bytes(f.read(2),
                                                              byteorder=self.byteorder,
                                                              signed=True)
        bh["30 fixed length trace flag"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                          signed=True)
        bh["31 number of extended textual file header"] = int.from_bytes(f.read(2),
                                                                         byteorder=self.byteorder,
                                                                         signed=True)

        bh["32 unassigned"] = int.from_bytes(f.read(94), byteorder=self.byteorder, signed=True)

        return bh

    def trace_header_mapping(self):
        """
        Return trace header byte-key mapping.
        """

        mapping = {}

        mapping[1] = "01 trace sequence number within line"
        mapping[5] = "02 trace sequence number within segy file"
        mapping[9] = "03 original field record number"
        mapping[13] = "04 trace number within the original field record"
        mapping[17] = "05 energy source point number"
        mapping[21] = "06 ensemble number"
        mapping[25] = "07 trace number within the ensemble"
        mapping[29] = "08 trace identification code"
        mapping[31] = "09 number of vertically summed traces yielding this trace"
        mapping[33] = "10 number of horizontally stacked traces yielding this trace"

        mapping[35] = "11 seisfast use"

        mapping[
            37] = "12 distance from center of source point to the center of the receiver group"
        mapping[41] = "13 receiver group elevation"
        mapping[45] = "14 surface elevation at source"
        mapping[49] = "15 source depth below surface"
        mapping[53] = "16 datum elevation at receiver group"
        mapping[57] = "17 datum elevaton at source"
        mapping[61] = "18 water depth at source"
        mapping[65] = "19 water depth at group"

        mapping[69] = "20 scalar to be applied to all elevations and depths"
        mapping[71] = "21 scalar to be applied to all coordinates"
        mapping[73] = "22 source coordinate x"
        mapping[77] = "23 source coordinate y"
        mapping[81] = "24 group coordinate x"
        mapping[85] = "25 group coordinate y"
        mapping[89] = "26 coordinate units"

        mapping[91] = "27 weathering velocity"
        mapping[93] = "28 subweathering velocity"
        mapping[95] = "29 uphole time at source (ms)"
        mapping[97] = "30 uphole time at group (ms)"

        mapping[99] = "31 source static correction (ms)"
        mapping[101] = "32 group static correction (ms)"
        mapping[103] = "33 total static applied (ms)"
        mapping[105] = "34 lag time A"
        mapping[107] = "35 lag time B"
        mapping[109] = "36 delay recording time"
        mapping[111] = "37 mute start time (ms)"
        mapping[113] = "38 mute end time (ms)"

        mapping[115] = "39 number of samples in this trace"
        mapping[117] = "40 sample interval for this trace (us)"

        mapping[119] = "41 gain type of field instruments"
        mapping[121] = "42 instrument gain constant (dB)"
        mapping[123] = "43 instrument early or initial gain (dB)"
        mapping[125] = "44 correlated"

        mapping[127] = "45 sweep frequency at start (Hz)"
        mapping[129] = "46 sweep frequency at end (Hz)"
        mapping[131] = "47 sweep length (ms)"
        mapping[133] = "48 sweep type"
        mapping[135] = "49 sweep trace taper length at start (ms)"
        mapping[137] = "50 sweep trace taper length at end (ms)"
        mapping[139] = "51 taper type"

        mapping[141] = "52 alias filter frequency (Hz)"
        mapping[143] = "53 alias filter slope (dB/octave)"
        mapping[145] = "54 notch filter frequency (Hz)"
        mapping[147] = "55 notch filter slope (dB/octave)"
        mapping[149] = "56 low-cut frequency (Hz)"
        mapping[151] = "57 high-cut frequency (Hz)"
        mapping[153] = "58 low-cut slope (dB/octave)"
        mapping[155] = "59 high-cut slope (dB/octave)"

        mapping[157] = "60 year seisfast recorded"
        mapping[159] = "61 day of year"
        mapping[161] = "62 hour of day"
        mapping[163] = "63 minute of hour"
        mapping[165] = "64 second of minute"
        mapping[167] = "65 time basis code"

        mapping[169] = "66 trace weighting factor"

        mapping[171] = "67 geophone group number of roll switch position one"
        mapping[
            173] = "68 geophone group number of trace number one within original field record"
        mapping[175] = "69 geophone group number of last trace within original field record"

        mapping[177] = "70 gap size"
        mapping[179] = "71 over travel associated with taper"

        mapping[181] = "72 x coordinate of ensemble position (cdp)"
        mapping[185] = "73 y coordinate of ensemble position (cdp)"
        mapping[189] = "74 3d poststack inline number"
        mapping[193] = "75 3d poststack crossline number"
        mapping[197] = "76 shotpoint number"
        mapping[201] = "77 scalar to be applied to the shotpoint number"

        mapping[203] = "78 trace value measurement unit"

        mapping[205] = "79 transduction constant"
        mapping[211] = "80 transduction units"
        mapping[213] = "81 device/trace identifier"
        mapping[215] = "82 scalar to be applied to times"
        mapping[217] = "83 source type/orientation"
        mapping[219] = "84 source energy direction"
        mapping[225] = "85 source measurement"
        mapping[231] = "86 source measurement unit"
        mapping[233] = "87 unassigned"

        return mapping

    def trace_header(self, idx):
        """
        Reads trace header.
        """

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        f = self.file_obj

        f.seek(
            3200 + 400 + (240 * idx) + (self.num_samples * _num_bytes(self.format_code) * idx))

        th = {}

        th["01 trace sequence number within line"] = int.from_bytes(f.read(4),
                                                                    byteorder=self.byteorder,
                                                                    signed=True)
        th["02 trace sequence number within segy file"] = int.from_bytes(f.read(4),
                                                                         byteorder=self.byteorder,
                                                                         signed=True)
        th["03 original field record number"] = int.from_bytes(f.read(4),
                                                               byteorder=self.byteorder,
                                                               signed=True)
        th["04 trace number within the original field record"] = int.from_bytes(f.read(4),
                                                                                byteorder=self.byteorder,
                                                                                signed=True)
        th["05 energy source point number"] = int.from_bytes(f.read(4),
                                                             byteorder=self.byteorder,
                                                             signed=True)
        th["06 ensemble number"] = int.from_bytes(f.read(4), byteorder=self.byteorder,
                                                  signed=True)
        th["07 trace number within the ensemble"] = int.from_bytes(f.read(4),
                                                                   byteorder=self.byteorder,
                                                                   signed=True)
        th["08 trace identification code"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                            signed=True)
        th["09 number of vertically summed traces yielding this trace"] = int.from_bytes(
            f.read(2), byteorder=self.byteorder, signed=True)
        th["10 number of horizontally stacked traces yielding this trace"] = int.from_bytes(
            f.read(2), byteorder=self.byteorder, signed=True)

        th["11 seisfast use"] = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)

        th[
            "12 distance from center of source point to the center of the receiver group"] = int.from_bytes(
            f.read(4), byteorder=self.byteorder, signed=True)
        th["13 receiver group elevation"] = int.from_bytes(f.read(4), byteorder=self.byteorder,
                                                           signed=True)
        th["14 surface elevation at source"] = int.from_bytes(f.read(4),
                                                              byteorder=self.byteorder,
                                                              signed=True)
        th["15 source depth below surface"] = int.from_bytes(f.read(4),
                                                             byteorder=self.byteorder,
                                                             signed=True)
        th["16 datum elevation at receiver group"] = int.from_bytes(f.read(4),
                                                                    byteorder=self.byteorder,
                                                                    signed=True)
        th["17 datum elevaton at source"] = int.from_bytes(f.read(4), byteorder=self.byteorder,
                                                           signed=True)
        th["18 water depth at source"] = int.from_bytes(f.read(4), byteorder=self.byteorder,
                                                        signed=True)
        th["19 water depth at group"] = int.from_bytes(f.read(4), byteorder=self.byteorder,
                                                       signed=True)

        th["20 scalar to be applied to all elevations and depths"] = int.from_bytes(f.read(2),
                                                                                    byteorder=self.byteorder,
                                                                                    signed=True)
        th["21 scalar to be applied to all coordinates"] = int.from_bytes(f.read(2),
                                                                          byteorder=self.byteorder,
                                                                          signed=True)
        th["22 source coordinate x"] = int.from_bytes(f.read(4), byteorder=self.byteorder,
                                                      signed=True)
        th["23 source coordinate y"] = int.from_bytes(f.read(4), byteorder=self.byteorder,
                                                      signed=True)
        th["24 group coordinate x"] = int.from_bytes(f.read(4), byteorder=self.byteorder,
                                                     signed=True)
        th["25 group coordinate y"] = int.from_bytes(f.read(4), byteorder=self.byteorder,
                                                     signed=True)
        th["26 coordinate units"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                   signed=True)

        th["27 weathering velocity"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                      signed=True)
        th["28 subweathering velocity"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                         signed=True)
        th["29 uphole time at source (ms)"] = int.from_bytes(f.read(2),
                                                             byteorder=self.byteorder,
                                                             signed=True)
        th["30 uphole time at group (ms)"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                            signed=True)

        th["31 source static correction (ms)"] = int.from_bytes(f.read(2),
                                                                byteorder=self.byteorder,
                                                                signed=True)
        th["32 group static correction (ms)"] = int.from_bytes(f.read(2),
                                                               byteorder=self.byteorder,
                                                               signed=True)
        th["33 total static applied (ms)"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                            signed=True)
        th["34 lag time A"] = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
        th["35 lag time B"] = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
        th["36 delay recording time"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                       signed=True)
        th["37 mute start time (ms)"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                       signed=True)
        th["38 mute end time (ms)"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                     signed=True)

        th["39 number of samples in this trace"] = int.from_bytes(f.read(2),
                                                                  byteorder=self.byteorder,
                                                                  signed=True)  # 115
        th["40 sample interval for this trace (us)"] = int.from_bytes(f.read(2),
                                                                      byteorder=self.byteorder,
                                                                      signed=True)

        th["41 gain type of field instruments"] = int.from_bytes(f.read(2),
                                                                 byteorder=self.byteorder,
                                                                 signed=True)
        th["42 instrument gain constant (dB)"] = int.from_bytes(f.read(2),
                                                                byteorder=self.byteorder,
                                                                signed=True)
        th["43 instrument early or initial gain (dB)"] = int.from_bytes(f.read(2),
                                                                        byteorder=self.byteorder,
                                                                        signed=True)
        th["44 correlated"] = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)

        th["45 sweep frequency at start (Hz)"] = int.from_bytes(f.read(2),
                                                                byteorder=self.byteorder,
                                                                signed=True)
        th["46 sweep frequency at end (Hz)"] = int.from_bytes(f.read(2),
                                                              byteorder=self.byteorder,
                                                              signed=True)
        th["47 sweep length (ms)"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                    signed=True)
        th["48 sweep type"] = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
        th["49 sweep trace taper length at start (ms)"] = int.from_bytes(f.read(2),
                                                                         byteorder=self.byteorder,
                                                                         signed=True)
        th["50 sweep trace taper length at end (ms)"] = int.from_bytes(f.read(2),
                                                                       byteorder=self.byteorder,
                                                                       signed=True)
        th["51 taper type"] = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)

        th["52 alias filter frequency (Hz)"] = int.from_bytes(f.read(2),
                                                              byteorder=self.byteorder,
                                                              signed=True)
        th["53 alias filter slope (dB/octave)"] = int.from_bytes(f.read(2),
                                                                 byteorder=self.byteorder,
                                                                 signed=True)
        th["54 notch filter frequency (Hz)"] = int.from_bytes(f.read(2),
                                                              byteorder=self.byteorder,
                                                              signed=True)
        th["55 notch filter slope (dB/octave)"] = int.from_bytes(f.read(2),
                                                                 byteorder=self.byteorder,
                                                                 signed=True)
        th["56 low-cut frequency (Hz)"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                         signed=True)
        th["57 high-cut frequency (Hz)"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                          signed=True)
        th["58 low-cut slope (dB/octave)"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                            signed=True)
        th["59 high-cut slope (dB/octave)"] = int.from_bytes(f.read(2),
                                                             byteorder=self.byteorder,
                                                             signed=True)

        th["60 year seisfast recorded"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                     signed=True)
        th["61 day of year"] = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
        th["62 hour of day"] = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
        th["63 minute of hour"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                 signed=True)
        th["64 second of minute"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                   signed=True)
        th["65 time basis code"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                  signed=True)

        th["66 trace weighting factor"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                         signed=True)

        th["67 geophone group number of roll switch position one"] = int.from_bytes(f.read(2),
                                                                                    byteorder=self.byteorder,
                                                                                    signed=True)
        th[
            "68 geophone group number of trace number one within original field record"] = int.from_bytes(
            f.read(2), byteorder=self.byteorder, signed=True)
        th[
            "69 geophone group number of last trace within original field record"] = int.from_bytes(
            f.read(2), byteorder=self.byteorder, signed=True)

        th["70 gap size"] = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
        th["71 over travel associated with taper"] = int.from_bytes(f.read(2),
                                                                    byteorder=self.byteorder,
                                                                    signed=True)

        th["72 x coordinate of ensemble position (cdp)"] = int.from_bytes(f.read(4),
                                                                          byteorder=self.byteorder,
                                                                          signed=True)
        th["73 y coordinate of ensemble position (cdp)"] = int.from_bytes(f.read(4),
                                                                          byteorder=self.byteorder,
                                                                          signed=True)
        th["74 3d poststack inline number"] = int.from_bytes(f.read(4),
                                                             byteorder=self.byteorder,
                                                             signed=True)
        th["75 3d poststack crossline number"] = int.from_bytes(f.read(4),
                                                                byteorder=self.byteorder,
                                                                signed=True)
        th["76 shotpoint number"] = int.from_bytes(f.read(4), byteorder=self.byteorder,
                                                   signed=True)
        th["77 scalar to be applied to the shotpoint number"] = int.from_bytes(f.read(2),
                                                                               byteorder=self.byteorder,
                                                                               signed=True)

        th["78 trace value measurement unit"] = int.from_bytes(f.read(2),
                                                               byteorder=self.byteorder,
                                                               signed=True)

        th["79 transduction constant"] = int.from_bytes(f.read(6), byteorder=self.byteorder,
                                                        signed=True)  # enconded seisfast 4 + 2
        th["80 transduction units"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                     signed=True)
        th["81 device/trace identifier"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                          signed=True)
        th["82 scalar to be applied to times"] = int.from_bytes(f.read(2),
                                                                byteorder=self.byteorder,
                                                                signed=True)
        th["83 source type/orientation"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                          signed=True)
        th["84 source energy direction"] = int.from_bytes(f.read(6), byteorder=self.byteorder,
                                                          signed=True)  # ?
        th["85 source measurement"] = int.from_bytes(f.read(6), byteorder=self.byteorder,
                                                     signed=True)  # enconded seisfast 4 + 2
        th["86 source measurement unit"] = int.from_bytes(f.read(2), byteorder=self.byteorder,
                                                          signed=True)
        th["87 unassigned"] = int.from_bytes(f.read(8), byteorder=self.byteorder, signed=True)

        return th

    def trace_data(self, idx):
        """
        Reads trace seisfast.
        """

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        f = self.file_obj

        f.seek(3200 + 400 + (240 * (idx + 1)) + (
                self.num_samples * _num_bytes(self.format_code) * idx))

        samples = []

        if self.format_code == 0:
            print("error: seisfast sample format code was not set.")
            return []

        for s in range(self.num_samples):
            if self.format_code == 1:
                aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                value = _ibm2ieee(aux)
            elif self.format_code == 2:
                value = int.from_bytes(f.read(4), byteorder=self.byteorder, signed=True)
            elif self.format_code == 3:
                value = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
            elif self.format_code == 4:
                print("error: obsolete format code.")
                return []
            elif self.format_code == 5:
                aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                value = struct.unpack('f', struct.pack('I', aux))[0]
            elif self.format_code == 6:
                print("error: not used format code.")
                return []
            elif self.format_code == 7:
                print("error: not used format code.")
                return []
            elif self.format_code == 8:
                value = int.from_bytes(f.read(1), byteorder=self.byteorder, signed=True)

            samples.append(value)

        return samples

    def pick_trace_value(self, idx, k=0):
        """
        Reads trace value at index k.
        """

        if k >= self.num_samples:
            return None

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        f = self.file_obj

        f.seek(3200 + 400 + (240 * (idx + 1)) + (
                self.num_samples * _num_bytes(self.format_code) * idx))

        f.seek(_num_bytes(self.format_code) * k, 1)

        if self.format_code == 0:
            print("error: seisfast sample format code was not set.")
            return None

        value = None

        if self.format_code == 1:
            aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
            value = _ibm2ieee(aux)
        elif self.format_code == 2:
            value = int.from_bytes(f.read(4), byteorder=self.byteorder, signed=True)
        elif self.format_code == 3:
            value = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
        elif self.format_code == 4:
            print("error: obsolete format code.")
            return []
        elif self.format_code == 5:
            aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
            value = struct.unpack('f', struct.pack('I', aux))[0]
        elif self.format_code == 6:
            print("error: not used format code.")
            return []
        elif self.format_code == 7:
            print("error: not used format code.")
            return []
        elif self.format_code == 8:
            value = int.from_bytes(f.read(1), byteorder=self.byteorder, signed=True)

        return value

    def trace_samples(self):
        """
        Reads the number of trace samples.
        """

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        f = self.file_obj

        f.seek(3220)

        samples = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)

        return samples

    # %% get

    def get_filename(self):
        """
        Returns the file name.
        """
        return self.file

    def get_range_inlines(self):
        """
        Returns the range of inlines.
        """
        return self.range_inlines

    def get_range_crosslines(self):
        """
        Returns the range of crosslines.
        """
        return self.range_crosslines

    def get_range_time_depth(self):
        """
        Returns the range of time/depth.
        """
        return self.range_time_depth

    def get_inline_resolution(self):
        """
        Returns inline resolution.
        """
        return self.res_inline

    def get_crossline_resolution(self):
        """
        Returns crossline resolution.
        """
        return self.res_crossline

    def get_time_depth_resolution(self):
        """
        Returns time/depth resolution.
        """
        return self.res_time_depth

    def get_num_inlines(self):
        """
        Returns the number of inlines.
        """
        return int(((self.range_inlines[1] - self.range_inlines[0]) / self.res_inline) + 1)

    def get_num_crosslines(self):
        """
        Returns the number of crosslines.
        """
        return int(
            ((self.range_crosslines[1] - self.range_crosslines[0]) / self.res_crossline) + 1)

    def get_num_samples(self):
        """
        Returns the number of samples.
        """
        return self.num_samples

    def get_range_x(self):
        """
        Returns the range of x coordinates.
        """
        return self.range_x

    def get_range_y(self):
        """
        Returns the range of y coordinates.
        """
        return self.range_y

    def get_encoding(self):
        """
        Returns character encoding.
        """
        return self.encoding

    def get_byteorder(self):
        """
        Returns byte order.
        """
        return self.byteorder

    def get_num_traces(self):
        """
        Returns the number of traces.
        """
        return self.num_traces

    def get_format_code(self):
        """
        Returns the format code.
        """
        return self.format_code

    def get_prov_info(self):
        prov_info = {
            "path": self.file,
            "weakHash": self.get_weak_hash(),
            "sizeInBytes": self.file_size,
            "modifiedDate": int(os.path.getmtime(self.file))
        }
        return prov_info

    def get_weak_hash(self):
        '''
        Uses a simple heuristic to return a 'weak' hash of the file based on parts of its content.
        Parts of the content in use: trace[0] + filesize
        :return: an integer number (including < 0) for the Murmur3 hash 32-bit, using fixed seed
        :see_also: https://en.wikipedia.org/wiki/MurmurHash
        '''
        from murmurhash import hash
        return hash(str(self.file_size) + str(self.trace_data(0)), seed=42)

    def info(self):
        """
        Prints information on the seismic dataset.
        """

        print("\nfile: " + self.file)

        print("encoding: " + self.encoding)

        print("byte order: " + self.byteorder)

        size = self.file_size / (1024 * 1024)

        if size > 1024:
            print("file size: " + str(round(size / 1024)) + " GB")
        elif size > 1:
            print("file size: " + str(round(size)) + " MB")
        else:
            print("file size: " + str(round(size * 1024)) + " KB")

        print("traces: " + str(self.num_traces))

        print("samples: " + str(self.num_samples))

        print("format code: " + str(self.format_code))

        return

    def get_crs(self):
        """
        Returns CRS in textual header if available.
        Looks for UTM codes and assumes WGS84.

        If available, returns (crs number, textual section where the value was found)
        Otherwise returns (None, None)

        Use with caution.
        """

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        f = self.file_obj

        f.seek(0)

        th = codecs.decode(f.read(3200), self.encoding)

        idx = th.lower().find("utm")

        section = None
        crs = None

        if idx != -1:  # there is crs information, looking for zones

            section = th[max(idx - 100, 0):min(idx + 100, len(th))]

            for k in range(60, 0, -1):

                idx = th.lower().find(str(k) + "n")  # north zones

                if idx > -1:
                    crs = 32600 + k
                    break
                else:
                    idx = th.lower().find(str(k) + " n")
                    if idx > -1:
                        crs = 32600 + k
                        break

                idx = th.lower().find(str(k) + "s")

                if idx > -1:
                    crs = 32700 + k
                    break
                else:
                    idx = th.lower().find(str(k) + " s")
                    if idx > -1:
                        crs = 32700 + k
                        break

        return crs, section

    def get_corrected_geo_coordinates(self, idx, fx, fy):
        """
        Process geo coordinates.
        """

        th = self.trace_header(idx)

        scalar = th["21 scalar to be applied to all coordinates"]

        x = th[fx]
        y = th[fy]

        if scalar > 0:
            x = x * scalar
            y = y * scalar
        else:
            x = x / abs(scalar)
            y = y / abs(scalar)

        return x, y

    def get_geo_coord_fields(self, idx=0):
        """
        Returns the names of the fields that are likely to store geographic information.

        If available returns (x_field_name, y_field_name)
        Otherwise returns (None, None).

        Use with caution.
        """

        fx = []
        fy = []

        th = self.trace_header(idx)

        scalar = th["21 scalar to be applied to all coordinates"]

        for key, value in th.items():

            if scalar > 0:
                v = value * scalar
            else:
                v = value / abs(scalar)

            # easting
            if len(str(int(v))) == 6:
                fx.append({key: v})

            # northing
            if len(str(int(v))) == 7:
                fy.append({key: v})

        if len(fx) == 0:
            fx.append("72 x coordinate of ensemble position (cdp)")

        if len(fy) == 0:
            fx.append("73 y coordinate of ensemble position (cdp)")

        return fx, fy

    def get_il_xl_fields(self, idx=0):
        """
        Returns the names of the fields that are likely to store inline/crossline information.

        If available returns (il_field_name, xl_field_name)
        Otherwise returns (None, None).

        Use with caution.
        """

        il = []
        xl = []

        th = self.trace_header(idx)

        for key, value in th.items():

            if "inline" in key:
                il.append({key: value})

            if "crossline" in key:
                xl.append({key: value})

        if len(il) == 0:
            il.append("74 3d poststack inline number")

        if len(xl) == 0:
            xl.append("75 3d poststack crossline number")

        return il, xl

    # %% write

    # def start_write_mode(self):
    #     """
    #     Starts write mode.
    #     """
    #
    #     if self.file_obj is not None:
    #         self.file_obj.close()
    #
    #     self.file_obj = open(self.file, "ab")
    #     self.file_mode = "write"
    #
    #     return

    def write_textual_header(self, th=None):
        """
        Writes textual header.
        """

        if th is None:
            th = " " * 3200

        if self.file_mode != "write":
            print("error: write mode should be on.")
            return

        f = self.file_obj

        f.write(codecs.encode(th, self.encoding))

        return

    def write_binary_header(self, bh=None, sample_interval=None):
        """
        Writes binary header.
        """

        if self.format_code == 0:
            print("error: format code was not set.")
            return

        if self.file_mode != "write":
            print("error: write mode should be on.")
            return

        f = self.file_obj

        sizes = [4] * 3 + [2] * 24 + [240] + [2] * 3 + [94]

        if bh is None:
            bh = [0] * 32

            bh[0] = 1
            bh[5] = int(sample_interval)
            bh[7] = self.num_samples
            bh[9] = self.format_code
            bh[24] = self.measurement_system
            bh[29] = self.fixed_length_trace_flag

            c = 0
            for k in range(len(bh)):
                f.write(bh[k].to_bytes(sizes[c], self.byteorder, signed=True))
                c += 1

        else:

            c = 0
            for k in sorted(bh):
                if c == 9:
                    f.write(self.format_code.to_bytes(sizes[c], self.byteorder, signed=True))
                else:
                    f.write(bh[k].to_bytes(sizes[c], self.byteorder, signed=True))
                c += 1

        return

    def write_trace_header(self, th, geox=None, geoy=None, sample_interval=None, inline=None,
                           crossline=None, wline=None, wsegy=None, delay=0):
        """
        Writes trace header.
        """

        if self.file_mode != "write":
            print("error: write mode should be on.")
            return

        f = self.file_obj

        sizes = [4] * 7 + [2] * 4 + [4] * 8 + [2] * 2 + [4] * 4 + [2] * 46 + [4] * 5 + [
            2] * 2 + [6] + [2] * 4 + [6] * 2 + [2, 8]

        if th is None:
            th = [0] * 87

            th[0] = int(wline)
            th[1] = int(wsegy)
            th[7] = self.trace_identification_code
            th[25] = self.coordinate_units
            th[35] = int(delay)
            th[38] = self.num_samples
            th[39] = int(sample_interval)
            th[71] = int(geox)
            th[72] = int(geoy)
            th[73] = int(inline)
            th[74] = int(crossline)

            c = 0
            for k in range(len(th)):
                f.write(th[k].to_bytes(sizes[c], self.byteorder, signed=True))
                c += 1

        else:

            c = 0
            for k in sorted(th):
                f.write(th[k].to_bytes(sizes[c], self.byteorder, signed=True))
                c += 1

        return

    def write_trace_data(self, td):
        """
        Writes trace seisfast.
        """

        if self.format_code == 0:
            print("error: seisfast sample format code was not set.")
            return

        if self.file_mode != "write":
            print("error: write mode should be on.")
            return

        f = self.file_obj

        for k in range(len(td)):
            if self.format_code == 1:
                print("error: format code not supported for writing.")
                return
            elif self.format_code == 2:
                f.write(td[k].to_bytes(4, byteorder=self.byteorder, signed=True))
            elif self.format_code == 3:
                f.write(td[k].to_bytes(2, byteorder=self.byteorder, signed=True))
            elif self.format_code == 4:
                print("error: obsolete format code.")
                return
            elif self.format_code == 5:
                aux = struct.unpack('I', struct.pack("f", td[k]))[0]
                f.write(aux.to_bytes(4, self.byteorder))
            elif self.format_code == 6:
                print("error: not used format code.")
                return
            elif self.format_code == 7:
                print("error: not used format code.")
                return
            elif self.format_code == 8:
                f.write(td[k].to_bytes(1, byteorder=self.byteorder, signed=True))

        return

    # def end_write_mode(self):
    #     """
    #     Ends write mode and switches back to read mode.
    #     """
    #
    #     if self.file_obj is not None:
    #         self.file_obj.close()
    #
    #         if os.path.isfile(self.file):
    #             self.file_obj = open(self.file, "rb")
    #             self.file_mode = "read"
    #         else:
    #             self.file_obj = None
    #             self.file_mode = None
    #
    #     return


class PostStackSEGY(BaseSEGY):
    """
    Class to read post-stack segy. Extends BaseSEGY.
    """

    # %% initialization

    def __init__(self, filename, num_traces=None, num_samples=None, format_code=None):
        """
        Constructor.
        """

        BaseSEGY.__init__(self, filename, num_traces, num_samples, format_code)

        self.trace_map = None

    def set_trace_map(self, trace_map):
        """
        Sets the trace map, a np.int array.
        """
        self.trace_map = trace_map

        return

    def get_trace_map(self):
        """
        Returns the trace map, a np.int array.
        """
        return self.trace_map

    def get_inline_section(self, inline):
        """
        Returns an inline section.
        """

        idx = (inline - self.get_range_inlines()[0]) // self.get_inline_resolution()

        return self.get_inline(idx)

    def get_inline(self, idx, num_xlines=None):
        """
        Returns an inline slice (zero-based index).
        """

        if self.format_code == 0:
            print("error: seisfast sample format code was not set.")
            return []

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        f = self.file_obj

        if self.trace_map is None:

            f.seek(3200 + 400 + (240 * (idx * num_xlines + 1)) + (
                    self.num_samples * _num_bytes(self.format_code) * idx * num_xlines))

            slice = np.zeros((self.num_samples, num_xlines))

            for col in range(num_xlines):

                samples = []

                for s in range(self.num_samples):
                    if self.format_code == 1:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = _ibm2ieee(aux)
                    elif self.format_code == 2:
                        value = int.from_bytes(f.read(4), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 3:
                        value = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 4:
                        print("error: obsolete format code.")
                        return []
                    elif self.format_code == 5:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = struct.unpack('f', struct.pack('I', aux))[0]
                    elif self.format_code == 6:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 7:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 8:
                        value = int.from_bytes(f.read(1), byteorder=self.byteorder, signed=True)

                    samples.append(value)

                slice[:, col] = samples

                f.seek(240, 1)

        else:

            num_xlines = self.trace_map.shape[1]

            slice = np.zeros((self.num_samples, num_xlines))

            for col in range(num_xlines):

                samples = []

                if self.trace_map[idx, col] == -1:
                    slice[:, col] = [0] * self.num_samples
                    continue

                # int32 was causing an overflow in seek function
                offset = int(self.trace_map[idx, col])

                f.seek(3200 + 400 + (240 * (offset + 1)) + (
                        self.num_samples * _num_bytes(self.format_code) * offset))

                for s in range(self.num_samples):
                    if self.format_code == 1:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = _ibm2ieee(aux)
                    elif self.format_code == 2:
                        value = int.from_bytes(f.read(4), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 3:
                        value = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 4:
                        print("error: obsolete format code.")
                        return []
                    elif self.format_code == 5:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = struct.unpack('f', struct.pack('I', aux))[0]
                    elif self.format_code == 6:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 7:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 8:
                        value = int.from_bytes(f.read(1), byteorder=self.byteorder, signed=True)

                    samples.append(value)

                slice[:, col] = samples

        return slice

    def get_crossline_section(self, crossline):
        """
        Returns a crossline slice.
        """

        idx = (crossline - self.get_range_crosslines()[0]) // self.get_crossline_resolution()

        return self.get_crossline(idx)

    def get_crossline(self, idx, num_ilines=None, num_xlines=None):
        """
        Returns a crossline slice (zero-based index).
        """

        if self.format_code == 0:
            print("error: seisfast sample format code was not set.")
            return []

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        f = self.file_obj

        if self.trace_map is None:

            f.seek(3200 + 400 + (240 * (idx + 1)) + (
                    self.num_samples * _num_bytes(self.format_code) * idx))

            slice = np.zeros((self.num_samples, num_ilines))

            for col in range(num_ilines):

                samples = []

                for s in range(self.num_samples):
                    if self.format_code == 1:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = _ibm2ieee(aux)
                    elif self.format_code == 2:
                        value = int.from_bytes(f.read(4), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 3:
                        value = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 4:
                        print("error: obsolete format code.")
                        return []
                    elif self.format_code == 5:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = struct.unpack('f', struct.pack('I', aux))[0]
                    elif self.format_code == 6:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 7:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 8:
                        value = int.from_bytes(f.read(1), byteorder=self.byteorder, signed=True)

                    samples.append(value)

                slice[:, col] = samples

                f.seek((240 * num_xlines) + (
                        self.num_samples * _num_bytes(self.format_code) * (num_xlines - 1)),
                       1)

        else:

            num_ilines = self.trace_map.shape[0]

            slice = np.zeros((self.num_samples, num_ilines))

            for col in range(num_ilines):

                samples = []

                if self.trace_map[col, idx] == -1:
                    slice[:, col] = [0] * self.num_samples
                    continue

                # int32 was causing an overflow in seek function
                offset = int(self.trace_map[col, idx])

                f.seek(3200 + 400 + (240 * (offset + 1)) + (
                        self.num_samples * _num_bytes(self.format_code) * offset))

                for s in range(self.num_samples):
                    if self.format_code == 1:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = _ibm2ieee(aux)
                    elif self.format_code == 2:
                        value = int.from_bytes(f.read(4), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 3:
                        value = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 4:
                        print("error: obsolete format code.")
                        return []
                    elif self.format_code == 5:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = struct.unpack('f', struct.pack('I', aux))[0]
                    elif self.format_code == 6:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 7:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 8:
                        value = int.from_bytes(f.read(1), byteorder=self.byteorder, signed=True)

                    samples.append(value)

                slice[:, col] = samples

        return slice

    def get_time_depth(self, idx, num_ilines=None, num_xlines=None):
        """
        Returns a time depth slice.
        """

        if self.format_code == 0:
            print("error: seisfast sample format code was not set.")
            return []

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        f = self.file_obj

        if self.trace_map is None:

            f.seek(3200 + 400 + 240)

            slice = np.zeros((num_ilines, num_xlines))

            for il in range(num_ilines):
                for xl in range(num_xlines):

                    f.seek(_num_bytes(self.format_code) * idx, 1)

                    if self.format_code == 1:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = _ibm2ieee(aux)
                    elif self.format_code == 2:
                        value = int.from_bytes(f.read(4), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 3:
                        value = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 4:
                        print("error: obsolete format code.")
                        return []
                    elif self.format_code == 5:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = struct.unpack('f', struct.pack('I', aux))[0]
                    elif self.format_code == 6:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 7:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 8:
                        value = int.from_bytes(f.read(1), byteorder=self.byteorder, signed=True)

                    slice[il, xl] = value

                    f.seek(_num_bytes(self.format_code) * (self.num_samples - idx - 1), 1)

                    f.seek(240, 1)

        else:

            num_ilines, num_xlines = self.trace_map.shape

            slice = np.zeros((num_ilines, num_xlines))

            for il in range(num_ilines):
                for xl in range(num_xlines):

                    if self.trace_map[il, xl] == -1:
                        slice[il, xl] = 0
                        continue

                    f.seek(3200 + 400 + (240 * (self.trace_map[il, xl] + 1)) + (
                            self.num_samples * _num_bytes(self.format_code) *
                            self.trace_map[il, xl]))

                    f.seek(_num_bytes(self.format_code) * idx, 1)

                    if self.format_code == 1:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = _ibm2ieee(aux)
                    elif self.format_code == 2:
                        value = int.from_bytes(f.read(4), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 3:
                        value = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 4:
                        print("error: obsolete format code.")
                        return []
                    elif self.format_code == 5:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = struct.unpack('f', struct.pack('I', aux))[0]
                    elif self.format_code == 6:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 7:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 8:
                        value = int.from_bytes(f.read(1), byteorder=self.byteorder, signed=True)

                    slice[il, xl] = value

        return slice

    def field_sample(self, field, size=10000):
        """
        Returns a sample of field with the given size.
        """

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        f = self.file_obj

        f.seek(3600)

        values = []

        for idx in range(0, size):
            f.seek(self.num_samples * _num_bytes(self.format_code) * int(idx > 0), 1)
            th = self.trace_header(idx)
            values.append(th[field])

        return values

    def get_valid_trace_mask(self):

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        f = self.file_obj
        f.seek(3600)

        mask = np.zeros(self.trace_map.shape)

        begin = 0
        size = 10

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):

                offset = self.trace_map[i,j]

                if offset == -1:
                    continue

                f.seek(3200 + 400 + (240 * (offset + 1)) + (
                        self.num_samples * _num_bytes(self.format_code) *
                        offset))

                total = 0

                for k in range(size):

                    f.seek(_num_bytes(self.format_code) * (begin+k), 1)

                    if self.format_code == 1:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = _ibm2ieee(aux)
                    elif self.format_code == 2:
                        value = int.from_bytes(f.read(4), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 3:
                        value = int.from_bytes(f.read(2), byteorder=self.byteorder, signed=True)
                    elif self.format_code == 4:
                        print("error: obsolete format code.")
                        return []
                    elif self.format_code == 5:
                        aux = int.from_bytes(f.read(4), byteorder=self.byteorder)
                        value = struct.unpack('f', struct.pack('I', aux))[0]
                    elif self.format_code == 6:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 7:
                        print("error: not used format code.")
                        return []
                    elif self.format_code == 8:
                        value = int.from_bytes(f.read(1), byteorder=self.byteorder, signed=True)

                    total += abs(value)

                if total > 1e-5:
                    mask[i, j] = 1

                # seisfast = self.trace_data(offset)
                # if np.abs(seisfast).sum() > 0:
                #     mask[i,j] = 1

        f.close()

        return mask

    def scan(self, finline="74 3d poststack inline number",
             fcrossline="75 3d poststack crossline number",
             fx="72 x coordinate of ensemble position (cdp)",
             fy="73 y coordinate of ensemble position (cdp)",
             save_scan=False,
             name_in_db=None):
        """
        Scans the cube and learns its geometry.

        If auxiliary files are available, they are loaded and scan is skipped.

        Parameters
        ----------

        finline: str
            field name for inline
        fcrossline: str
            field name for crosslines
        fx: str
            field name for cdp geo-x
        fy : str
            field name for cdp geo-y
        save_scan: bool
            whether the result of the scan should be saved on disk
        name_in_db: str
            unique name in db for storing trace_map files

        Returns
        -------

        dict
            dictionary with meta information.

            If `save_scan` is true, json and npy auxiliary files should be created

        """
        if len(self.scan_info) > 0:
            return self.scan_info

        # TODO: please move the prints to log
        # print("\nscanning seismic...")

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        if name_in_db:
            path_to_save_trace_map = self.file + "_" + name_in_db + ".trace_map.npy"
            path_to_save_json = self.file + "_" + name_in_db + ".json"
        else:
            path_to_save_trace_map = self.file + ".trace_map.npy"
            path_to_save_json = self.file + ".json"

        if os.path.exists(path_to_save_trace_map):
            # TODO: please move the prints to log
            # print("\nfound meta files. loading...")

            self.trace_map = np.load(path_to_save_trace_map)

            json = open(path_to_save_json)
            obj = jsonlib.load(json)
            self.range_inlines = obj["range_inlines"]
            self.range_crosslines = obj["range_crosslines"]
            self.range_time_depth = obj["range_time_depth"]
            self.res_inline = obj["res_inline"]
            self.res_crossline = obj["res_crossline"]
            self.res_time_depth = obj["res_time_depth"]
            self.range_x = obj["range_x"]
            self.range_y = obj["range_y"]

            json.close()

            self.scan_info = {
                "range_inlines": self.range_inlines,
                "range_crosslines": self.range_crosslines,
                "num_inlines": int(((self.range_inlines[1] - self.range_inlines[0]) / self.res_inline) + 1),
                "num_crosslines": int(((self.range_crosslines[1] - self.range_crosslines[0]) / self.res_crossline) + 1),
                "range_time_depth": self.range_time_depth,
                "num_time_depth": self.num_samples,
                "res_inline": self.res_inline,
                "res_crossline": self.res_crossline,
                "res_time_depth": self.res_time_depth,
                "range_x": self.range_x,
                "range_y": self.range_y
            }

            return self.scan_info

        bh = self.binary_header()

        range_time_depth = []
        num_time_depth = bh["08 number of samples per seisfast trace"]

        f = self.file_obj

        f.seek(3600)

        range_inlines = [sys.maxsize, -sys.maxsize]
        range_crosslines = [sys.maxsize, -sys.maxsize]

        range_x = [sys.maxsize, -sys.maxsize]
        range_y = [sys.maxsize, -sys.maxsize]

        first_il = None
        first_xl = None
        res_il = sys.maxsize
        res_xl = sys.maxsize

        trace_dict = {}

        for idx in range(0, self.num_traces):
            f.seek(self.num_samples * _num_bytes(self.format_code) * int(idx > 0), 1)
            th = self.trace_header(idx)

            if idx == 0:
                time_res = th["40 sample interval for this trace (us)"] / 1e3
                delay = th["36 delay recording time"]
                range_time_depth = [delay, delay + time_res * (num_time_depth - 1)]

                first_il = th[finline]
                first_xl = th[fcrossline]

            delta_il = abs(th[finline] - first_il)
            res_il = max(min(delta_il, res_il), 1)

            delta_xl = abs(th[fcrossline] - first_xl)
            res_xl = max(min(delta_xl, res_xl), 1)

            if th[finline] < range_inlines[0]:
                range_inlines[0] = th[finline]
            if th[finline] > range_inlines[1]:
                range_inlines[1] = th[finline]

            if th[fcrossline] < range_crosslines[0]:
                range_crosslines[0] = th[fcrossline]
            if th[fcrossline] > range_crosslines[1]:
                range_crosslines[1] = th[fcrossline]

            if fx is not None:
                if th[fx] < range_x[0]:
                    range_x[0] = th[fx]
                if th[fx] > range_x[1]:
                    range_x[1] = th[fx]

            if fy is not None:
                if th[fy] < range_y[0]:
                    range_y[0] = th[fy]
                if th[fy] > range_y[1]:
                    range_y[1] = th[fy]

            if trace_dict.get(int(th[finline])) is not None:
                trace_dict[int(th[finline])].append({int(th[fcrossline]): idx})
            else:
                trace_dict[int(th[finline])] = []
                trace_dict[int(th[finline])].append({int(th[fcrossline]): idx})

            if (idx % int(self.num_traces * 0.01) == 0):
                print("{:5.1f}%".format(idx / self.num_traces * 100), end="\r", flush=True)

        if res_il is None:
            res_il = 1
        if res_xl is None:
            res_xl = 1

        self.range_inlines = range_inlines
        self.range_crosslines = range_crosslines
        self.res_inline = res_il
        self.res_crossline = res_xl
        self.range_time_depth = range_time_depth
        self.res_time_depth = time_res

        self.scan_info = {
            "range_inlines": self.range_inlines,
            "range_crosslines": self.range_crosslines,
            "range_time_depth": self.range_time_depth,
            "num_inlines": int(((range_inlines[1] - range_inlines[0]) / res_il) + 1),
            "num_crosslines": int(((range_crosslines[1] - range_crosslines[0]) / res_xl) + 1),
            "num_time_depth": num_time_depth,
            "res_inline": self.res_inline,
            "res_crossline": self.res_crossline,
            "res_time_depth": self.res_time_depth,
        }

        self.range_x = range_x
        self.range_y = range_y

        if fx is not None:
            self.scan_info["range_x"] = range_x
        if fy is not None:
            self.scan_info["range_y"] = range_y

        self.trace_map = np.ones((self.scan_info["num_inlines"], self.scan_info["num_crosslines"]), dtype=np.int32) * -1

        for il, value in trace_dict.items():
            for k in value:
                for xl, trace in k.items():
                    self.trace_map[int((il - range_inlines[0]) / res_il), int(
                        (xl - range_crosslines[0]) / res_xl)] = trace

        if save_scan:
            print("\nsaving meta files...")

            np.save(path_to_save_trace_map, self.trace_map)

            json = open(path_to_save_json, "w")

            # obj = {"range_inlines": range_inlines,
            #        "range_crosslines": range_crosslines,
            #        "res_inline": res_il,
            #        "res_crossline": res_xl,
            #        "range_x": range_x,
            #        "range_y": range_y,
            #        "res_time_depth": time_res,q
            #        "range_time_depth": range_time_depth}

            jsonlib.dump(self.scan_info, json)
            json.close()

        return self.scan_info

    def get_prov_info(self):
        provinfo = super().get_prov_info()
        provinfo.update(self.scan())
        return provinfo

class PreStackSEGY(BaseSEGY):
    """
    Class to read pre-stack segy. Extends BaseSEGY.
    """

    # %% initialization

    def __init__(self, filename, num_traces=None, num_samples=None, format_code=None):
        """
        Constructor.
        """

        BaseSEGY.__init__(self, filename, num_traces, num_samples, format_code)

        self.cdp_map = None
        self.source_map = None
        self.receiver_map = None

    def scan(self, finline="74 3d poststack inline number",
             fcrossline="75 3d poststack crossline number",
             fx="72 x coordinate of ensemble position (cdp)",
             fy="73 y coordinate of ensemble position (cdp)",
             save_scan=False,
             fsource="05 energy source point number",
             frecx="24 group coordinate x",
             frecy="25 group coordinate y",
             name_in_db=None):
        """
        Scans the cube and learns its geometry.

        If auxiliary files are available, they are loaded and scan is skipped.

        Parameters
        ----------

        finline: str
            field name for inline
        fcrossline: str
            field name for crosslines
        fx: str
            field name for cdp geo-x
        fy : str
            field name for cdp geo-y
        save_scan: bool
            whether the result of the scan should be saved on disk
        fsource: str
            field name for source id
        frecx: str
            field name for receiver geo-x
        frecy: str
            field name for receiver geo-y
        name_in_db: str
            unique name_in_db for storing trace_map

        Returns
        -------

        dict
            dictionary with meta information.

            If `save_scan` is true, json and pickle auxiliary files should be created

        """
        if len(self.scan_info) > 0:
            return self.scan_info

        # TODO: please move the prints to log
        # print("\nscanning seismic...")

        if self.file_mode != "read":
            print("read mode should be on.")
            return

        if name_in_db:
            path_to_save_trace_map = self.file + "_" + name_in_db + ".trace_map.pickle"
            path_to_save_json = self.file + "_" + name_in_db + ".json"
        else:
            path_to_save_trace_map = self.file + ".trace_map.pickle"
            path_to_save_json = self.file + ".json"

        if os.path.exists(path_to_save_trace_map):
            # TODO: please move the prints to log
            # print("\nfound meta files. loading...")

            trace_map_file = open(path_to_save_trace_map, "rb")
            trace_map = pickle.load(trace_map_file)
            trace_map_file.close()

            self.cdp_map = trace_map["cdp"]
            self.source_map = trace_map["source"]
            self.receiver_map = trace_map["receiver"]

            json = open(path_to_save_json)
            obj = jsonlib.load(json)
            self.range_inlines = obj["range_inlines"]
            self.range_crosslines = obj["range_crosslines"]
            self.range_time_depth = obj["range_time_depth"]
            self.res_inline = obj["res_inline"]
            self.res_crossline = obj["res_crossline"]
            self.res_time_depth = obj["res_time_depth"]
            self.range_x = obj["range_x"]
            self.range_y = obj["range_y"]

            json.close()

            self.scan_info = {
                "range_inlines": self.range_inlines,
                "range_crosslines": self.range_crosslines,
                "num_inlines": int(((self.range_inlines[1] - self.range_inlines[0]) / self.res_inline) + 1),
                "num_crosslines": int(((self.range_crosslines[1] - self.range_crosslines[0]) / self.res_crossline) + 1),
                "range_time_depth": self.range_time_depth,
                "num_time_depth": self.num_samples,
                "res_inline": self.res_inline,
                "res_crossline": self.res_crossline,
                "res_time_depth": self.res_time_depth
            }

            return self.scan_info

        bh = self.binary_header()

        range_time_depth = []
        num_time_depth = bh["08 number of samples per seisfast trace"]

        f = self.file_obj

        f.seek(3600)

        range_inlines = [sys.maxsize, -sys.maxsize]
        range_crosslines = [sys.maxsize, -sys.maxsize]

        range_x = [sys.maxsize, -sys.maxsize]
        range_y = [sys.maxsize, -sys.maxsize]

        first_il = None
        first_xl = None
        res_il = sys.maxsize
        res_xl = sys.maxsize

        trace_dict = {}
        self.source_map = {}
        self.receiver_map = {}

        for idx in range(0, self.num_traces):
            f.seek(self.num_samples * _num_bytes(self.format_code) * int(idx > 0), 1)
            th = self.trace_header(idx)

            if idx == 0:
                time_res = th["40 sample interval for this trace (us)"] / 1e3
                delay = max(th["36 delay recording time"], -th["34 lag time A"])
                range_time_depth = [delay, delay + time_res * (num_time_depth - 1)]

                first_il = th[finline]
                first_xl = th[fcrossline]

            delta_il = abs(th[finline] - first_il)
            res_il = max(min(delta_il, res_il), 1)

            delta_xl = abs(th[fcrossline] - first_xl)
            res_xl = max(min(delta_xl, res_xl), 1)

            if th[finline] < range_inlines[0]:
                range_inlines[0] = th[finline]
            if th[finline] > range_inlines[1]:
                range_inlines[1] = th[finline]

            if th[fcrossline] < range_crosslines[0]:
                range_crosslines[0] = th[fcrossline]
            if th[fcrossline] > range_crosslines[1]:
                range_crosslines[1] = th[fcrossline]

            if fx is not None:
                if th[fx] < range_x[0]:
                    range_x[0] = th[fx]
                if th[fx] > range_x[1]:
                    range_x[1] = th[fx]

            if fy is not None:
                if th[fy] < range_y[0]:
                    range_y[0] = th[fy]
                if th[fy] > range_y[1]:
                    range_y[1] = th[fy]

            if trace_dict.get(int(th[finline])) is not None:
                trace_dict[int(th[finline])].append({int(th[fcrossline]): idx})
            else:
                trace_dict[int(th[finline])] = []
                trace_dict[int(th[finline])].append({int(th[fcrossline]): idx})

            # TODO: is this reliable or should we use source x and y?
            if self.source_map.get(th[fsource]) is None:
                self.source_map[th[fsource]] = [idx]
            else:
                self.source_map[th[fsource]].append(idx)

            receiver_id = str(th[frecx]) + str(th[frecy])

            if self.receiver_map.get(receiver_id) is None:
                self.receiver_map[receiver_id] = [idx]
            else:
                self.receiver_map[receiver_id].append(idx)

            if (idx % int(self.num_traces * 0.01) == 0):
                print("{:5.1f}%".format(idx / self.num_traces * 100), end="\r", flush=True)

        if res_il is None:
            res_il = 1
        if res_xl is None:
            res_xl = 1

        self.range_inlines = range_inlines
        self.range_crosslines = range_crosslines
        self.res_inline = res_il
        self.res_crossline = res_xl
        self.range_time_depth = range_time_depth
        self.res_time_depth = time_res

        self.scan_info = {
            "range_inlines": range_inlines,
            "range_crosslines": range_crosslines,
            "num_inlines": int(((range_inlines[1] - range_inlines[0]) / res_il) + 1),
            "num_crosslines": int(((range_crosslines[1] - range_crosslines[0]) / res_xl) + 1),
            "range_time_depth": range_time_depth,
            "num_time_depth": num_time_depth,
            "res_inline": res_il,
            "res_crossline": res_xl,
            "res_time_depth": time_res
        }

        self.range_x = range_x
        self.range_y = range_y

        if fx is not None:
            self.scan_info["range_x"] = range_x
        if fy is not None:
            self.scan_info["range_y"] = range_y

        self.cdp_map = {}

        num_xlines = int(((range_crosslines[1] - range_crosslines[0]) / res_xl) + 1)

        for il, value in trace_dict.items():
            for k in value:
                for xl, trace in k.items():

                    cdp_id = int((il - range_inlines[0]) / res_il) * num_xlines + int(
                        (xl - range_crosslines[0]) / res_xl)

                    if self.cdp_map.get(cdp_id) is None:
                        self.cdp_map[cdp_id] = [trace]
                    else:
                        self.cdp_map[cdp_id].append(trace)

                    # self.trace_map[int((il-range_inlines[0]) / res_il), int((xl-range_crosslines[0]) / res_xl)] = trace

        if save_scan:
            print("\nsaving meta files...")

            trace_map = {}
            trace_map["cdp"] = self.cdp_map
            trace_map["source"] = self.source_map
            trace_map["receiver"] = self.receiver_map

            # np.save(self.file + ".trace_map", self.trace_map)
            trace_map_file = open(path_to_save_trace_map, 'wb')
            pickle.dump(trace_map, trace_map_file)
            trace_map_file.close()

            json = open(path_to_save_json, "w")

            # obj = {"range_inlines": range_inlines,
            #        "range_crosslines": range_crosslines,
            #        "res_inline": res_il,
            #        "res_crossline": res_xl,
            #        "range_x": range_x,
            #        "range_y": range_y,
            #        "res_time_depth": time_res,
            #        "range_time_depth": range_time_depth}

            jsonlib.dump(self.scan_info, json)
            json.close()

        return self.scan_info

    def get_prov_info(self):
        provinfo = super().get_prov_info()
        provinfo.update(self.scan())
        return provinfo

    def get_cdp_gather(self, inline, crossline=None):
        """
        Returns a cdp gather.

        Parameters
        ----------

        inline: number or array
            inline number. If only inline is passed, assumes it is a cdp id

        crossline: number or array
            crossline number (optional)

        Returns
        -------

        list
            list with the traces for a cdp

        """

        cdp_ids = []

        if crossline is not None:

            if type(inline) != list:
                inline = [inline]

            if type(crossline) != list:
                crossline = [crossline]

            num_xlines = int(((self.range_crosslines[1] - self.range_crosslines[
                0]) / self.res_crossline) + 1)

            for il in inline:
                for xl in crossline:
                    cdp_ids.append(
                        int((il - self.range_inlines[0]) / self.res_inline) * num_xlines + int(
                            (xl - self.range_crosslines[0]) / self.res_crossline))

        else:
            cdp_ids.append(inline)

        output = []

        for id in cdp_ids:

            l = self.cdp_map.get(id)

            if l is not None:
                output += l

        return output

    def get_il_xl_from_cdp_id(self, cdp_id):
        """
        Converts cdp id to inline, crossline coordinates.

        Parameters
        ----------

        cdp_id : number
            cdp id

        Returns
        -------

        tuple
            inline, crossline coordinates.

        """

        il = int(cdp_id / self.get_num_crosslines()) * self.res_inline + self.range_inlines[0]

        xl = (cdp_id % self.get_num_crosslines()) * self.res_crossline + self.range_crosslines[
            0]

        return il, xl

    def get_cdp_id_from_il_xl(self, inline, crossline):
        """
        Converts inline, crossline coordinates to cdp id.

        Parameters
        ----------

        inline : number
            inline coordinate

        crossline : number
            crossline coordinate

        Returns
        -------

        number
            cdp id.

        """

        num_xlines = int(
            ((self.range_crosslines[1] - self.range_crosslines[0]) / self.res_crossline) + 1)
        cdp_id = int((inline - self.range_inlines[0]) / self.res_inline) * num_xlines + int(
            (crossline - self.range_crosslines[0]) / self.res_crossline)

        return cdp_id

    def get_source_gather(self, source_id):
        """
        Returns a source/shot gather.

        Parameters
        ----------

        source_id : str
            source id

        Returns
        -------

        list
            list with the traces for a shot.

        """

        return self.source_map[source_id]

    def get_receiver_gather(self, x, y=None):
        """
        Returns a receiver gather.

        Parameters
        ----------

        x: number or str
            receiver geo-x. If only x is passed, assumes it is a receiver id

        y : number or str
            receiver geo-y (optional)

        Returns
        -------

        list
            list with the traces for a receiver

        """

        if y is None:  # if only x is passed we assume it is a receiver id
            receiver_id = x
        else:
            receiver_id = str(x) + str(y)

        return self.receiver_map[receiver_id]

    def get_cdps(self):
        """
        Returns a list of cdp id's.
        """
        return sorted(list(self.cdp_map.keys()))

    def get_sources(self):
        """
        Returns a list of source/shot id's.
        """
        return sorted(list(self.source_map.keys()))

    def get_receivers(self):
        """
        Returns a list of receiver id's.
        """
        return sorted(list(self.receiver_map.keys()))


class SEGY(PostStackSEGY):
    """
    Class to read post-stack segy. Extends PostStackSEGY.

    Backward compatibility.
    """

    # %% initialization

    def __init__(self, filename, num_traces=None, num_samples=None, format_code=None):
        """
        Constructor.
        """

        PostStackSEGY.__init__(self, filename, num_traces, num_samples, format_code)
