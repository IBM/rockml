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


import io
import json as jsonlib
import os.path
import sys

import numpy as np
import pandas as pd


def _detect_format(filename):
    """Detects ASCII file format.

    Parameters
    ----------

    filename : str

    Returns
    -------

    format : int

        1 - any csv-like (il xl z, x y z, il xl x y z)
        2 - inline crossline x.y.z (format with headers and EOFs) (deprecated)

    """

    file = open(filename, 'r')

    for line in file:

        l = line.strip()

        if len(l) == 0:  # skipping blank lines
            continue

        if l.startswith("*") or l.startswith("#"):  # skipping the header
            continue

        terms = l.split()

        if terms[0].replace(".", "").isnumeric():
            frmt = 1
        else:
            frmt = 2  # deprecated

    file.close()

    return frmt


class Reader:
    """
    Base class to read horizon files.
    """

    file = None
    file_obj = None
    format = "icxyz"
    separator = "\s+"
    comment = "*"

    # %% initialization

    def __init__(self, filename, format="icxyz", sep="\s+", comment="*"):
        """
        Sets the filename and initializes variables.
        """

        self.file = filename
        self.format = format
        self.separator = sep
        self.comment = comment

        self.data = None
        self.point_map = None

        self.range_inlines = []
        self.range_crosslines = []
        self.range_time_depth = []

        assert os.path.isfile(self.file), "Horizon file does not exist"

        self.file_obj = open(self.file, "r")

    def __del__(self):
        """
        Closes the file before destruction.
        """

        if self.file_obj is not None:
            self.file_obj.close()

    def get_point_map(self, save=False):
        """
        Returns an image of the horizon in seismic coordinates.
        """

        if os.path.exists(self.file + ".point_map.npy"):

            self.point_map = np.load(self.file + ".point_map.npy")

            json = open(self.file + ".json")
            obj = jsonlib.load(json)
            self.range_inlines = obj["range_inlines"]
            self.range_crosslines = obj["range_crosslines"]
            self.range_time_depth = obj["range_time_depth"]

            json.close()

            return self.point_map

        else:

            if "ic" not in self.format:
                print("missing inline/crossline information")
                return

            self.get_values()

            min_il = self.data[:,0].min()
            max_il = self.data[:,0].max()

            min_xl = self.data[:,1].min()
            max_xl = self.data[:,1].max()

            min_td = self.data[:,-1].min()
            max_td = self.data[:,-1].max()

            self.range_inlines = [min_il, max_il]
            self.range_crosslines = [min_xl, max_xl]
            self.range_time_depth = [min_td, max_td]

            self.point_map = np.ones((int(max_il - min_il + 1),
                                      int(max_xl - min_xl + 1))) * -1

            for i in range(self.data.shape[0]):
                self.point_map[int(self.data[i, 0] - min_il),
                               int(self.data[i, 1] - min_xl)] = self.data[
                    i, self.data.shape[1] - 1]

        if save:
            np.save(self.file + ".point_map.npy", self.point_map)

            json = open(self.file + ".json", "w")

            obj = {"range_inlines": self.range_inlines,
                   "range_crosslines": self.range_crosslines,
                   "range_time_depth": self.range_time_depth}

            jsonlib.dump(obj, json)
            json.close()

        return self.point_map

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

    def get_values(self):
        """
        Returns horizon values as an array.
        """

        if self.data is None:

            # seisfast = pd.read_csv(self.file, sep=self.separator, header=None, comment=self.comment)
            # self.seisfast = seisfast.values

            data = []

            self.file_obj.seek(0)

            for line in self.file_obj:

                l = line.strip()

                if len(l) == 0:  # skipping blank lines
                    continue

                if l.startswith(self.comment):  # skipping the header
                    continue

                # removing potential text
                l = l.replace("XLINE", "")
                l = l.replace("INLINE", "")
                l = l.replace(":", "")

                fields = l.split()

                row = []

                for f in fields:
                    value = f.strip()
                    row += [float(value)]

                data.append(row)

            self.data = np.array(data)

        return self.data


class Writer:
    """
    Base class to write horizon files.
    """

    file = None
    file_obj = None

    # %% initialization

    def __init__(self, filename):
        """
        Sets the filename and initializes variables.
        """

        self.file = filename
        self.file_obj = open(self.file, "a")

    def __del__(self):
        """
        Closes the file before destruction.
        """

        if self.file_obj is not None:
            self.file_obj.close()

    def write(self, direction, point_map, segy, fx="72 x coordinate of ensemble position (cdp)",
              fy="73 y coordinate of ensemble position (cdp)"):
        """
        Writes horizon seisfast to a file.

        Expects a point map in the form: index, column, row
        """

        #print("\nwriting horizon...\n")

        space = "          "

        trace_map = segy.get_trace_map()

        range_inlines = segy.get_range_inlines()
        range_crosslines = segy.get_range_crosslines()
        range_time_depth = segy.get_range_time_depth()

        inline_resolution = segy.get_inline_resolution()
        crossline_resolution = segy.get_crossline_resolution()
        time_depth_resolution = segy.get_time_depth_resolution()

        for k in range(point_map.shape[0]):

            if direction == "inlines":

                il = point_map[k, 0]
                xl = round(point_map[k, 1] * crossline_resolution + range_crosslines[0])
                file_idx = trace_map[int(round((il - range_inlines[0]) / inline_resolution)), int(point_map[k,1])]

            elif direction == "crosslines":

                il = round(point_map[k, 1] * inline_resolution + range_inlines[0])
                xl = point_map[k, 0]
                file_idx = trace_map[int(point_map[k,1]), int(round((xl - range_crosslines[0]) / crossline_resolution))]

            if file_idx < 0:
                continue
                
            z = round(point_map[k, 2] * time_depth_resolution + range_time_depth[0])

            x, y = segy.get_corrected_geo_coordinates(file_idx, fx, fy)

            self.file_obj.write(str(il) + space + str(xl) + space + str(x) + space + str(y) + space + str(z))

            if point_map.shape[1] > 3:
                for w in range(point_map.shape[1] - 3):
                    self.file_obj.write(space + str(point_map[k,3+w]))

            self.file_obj.write("\n")

        #print("\ndone.\n")

        return


def read(filename, sep="\s+", comment="*"):
    """Reads an ASCII file containing a horizon.

    Parameters
    ----------

    filename : str
        ASCII file name.

    sep : str
        Separator. "\s+" for space, "\t" for tab, etc.

    comment : str
        Character for comment or header (that should be ignored).

    Returns
    -------

    output : 2-D array
        [points, coords]

    """

    if _detect_format(filename) == 1:

        data = pd.read_csv(filename, sep=sep, header=None, comment=comment)

        X = data.values

    else:

        file = open(filename, 'r')

        buffer = ""

        for line in file:

            l = line.strip()

            if len(l) == 0:
                continue

            terms = l.split()

            if terms[0].isnumeric():
                buffer += line

        file.close()

        buf = io.StringIO()
        buf.write(buffer)

        buf.seek(0)

        data = pd.read_csv(buf, delim_whitespace=True, header=None)

        buf.close()

        all_data = data.values

        X = np.zeros((all_data.shape[0], 3))

        for i in range(all_data.shape[0]):
            nums = all_data[i, 2].split(".")
            X[i, 0] = nums[0]
            X[i, 1] = nums[1]
            X[i, 2] = nums[2]

    return X
