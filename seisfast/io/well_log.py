""" Copyright 2023 IBM Research. All Rights Reserved.
"""


import os

import lasio
import numpy as np


def convert_number(s):
    try:
        f = float(s)
        return f
    except:
        return s


def read_log(file, k, data_section=True):
    file.seek(0)

    if data_section:
        in_section = False
    else:
        in_section = True
        file.readline()

    log = []

    for line in file:

        if line.startswith("~A"):
            in_section = True
            continue

        if line.startswith("#"):
            continue

        if in_section and line.startswith("~"):
            return log

        if in_section:
            data = line.split()

            values = []

            values.append(convert_number(data[k].strip()))

            log.append(values)

    return log


def read_section(section, file, data_section=True):
    file.seek(0)
    in_section = False

    if section in ["V", "W", "C", "P"]:
        dictionary = []
    elif section == "O":
        dictionary = None
    else:
        dictionary = []

    if not data_section:
        in_section = True
        file.readline()

    for line in file:

        if line.startswith("~" + section):
            in_section = True
            continue

        if line.startswith("#"):
            continue

        if in_section and line.startswith("~"):
            return dictionary

        if in_section:

            if section in ["V", "W", "C", "P"]:

                # separates description
                idxcolon = str.rfind(line, ":")
                description = line[idxcolon + 1:].strip()

                # separates mnemonic
                idxdot = str.find(line, ".")
                mnemonic = line[:idxdot].strip()

                # separates units and seisfast
                idxspace = str.find(line, " ", idxdot)

                units = line[idxdot + 1:idxspace].strip()

                data = line[idxspace + 1:idxcolon].strip()

                dictionary.append({"mnemonic": mnemonic, "units": units, "seisfast": data,
                                   "description": description})

            elif section == "O":

                if dictionary is None:
                    dictionary = line
                else:
                    dictionary += line

            elif section == "A":

                data = line.split()

                values = []

                for d in data:
                    values.append(convert_number(d.strip()))

                dictionary.append(values)

    if section == "A":
        return np.array(dictionary)
    else:
        return dictionary


class LAS:
    null = None
    sections = [False] * 6

    # %% initialization

    def __init__(self, filename):
        """
        Sets the filename and initializes variables.
        """

        self.file = filename

        if os.path.isfile(self.file):

            self.file_obj = open(self.file, "r")

            value = self.get_null()

            if value is not None:
                self.null = value
            else:
                print("\nWarning: null value is missing. Use set_null() to set it.")

            self.file_obj.seek(0)

            for line in self.file_obj:

                if line.startswith("~V"):
                    self.sections[0] = True

                elif line.startswith("~W"):
                    self.sections[1] = True

                elif line.startswith("~C"):
                    self.sections[2] = True

                elif line.startswith("~P"):
                    self.sections[3] = True

                elif line.startswith("~O"):
                    self.sections[4] = True

                elif line.startswith("~A"):
                    self.sections[5] = True

        return

    def __del__(self):
        """
        Closes the file before destruction.
        """

        if self.file_obj is not None:
            self.file_obj.close()

    # sections

    def get_version_section(self):

        version = []

        if self.sections[0]:
            version = read_section("V", self.file_obj)

        return version

    def get_well_section(self):

        well = []

        if self.sections[1]:
            well = read_section("W", self.file_obj)

        return well

    def get_curve_section(self):

        curve = []

        if self.sections[2]:
            curve = read_section("C", self.file_obj)
        else:

            self.file_obj.seek(0)

            line = self.file_obj.readline()

            mnemonics = line.split()

            for m in mnemonics:
                m = m.strip()
                idxpar = str.find(m, "(")
                idxpar2 = str.find(m, ")")
                c = {"mnemonic": m[:idxpar], "unit": m[idxpar + 1:idxpar2], "description": "",
                     "seisfast": ""}
                curve.append(c)

        return curve

    def get_parameter_section(self):

        parameter = []

        if self.sections[3]:
            parameter = read_section("P", self.file_obj)

        return parameter

    def get_other_section(self):

        other = None

        if self.sections[4]:
            other = read_section("O", self.file_obj)

        return other

    def get_data_section(self):

        data = []

        data = read_section("A", self.file_obj, self.sections[5])

        return data

    # curves

    def get_curve_index(self, name):

        if self.sections[2]:

            curve = read_section("C", self.file_obj)

            values = []

            k = 0
            for c in curve:
                if ((name.lower() in c["description"].lower()) or
                        (name.lower() in c["mnemonic"].lower())):
                    c["index"] = k
                    values.append(c)
                k += 1

        else:

            self.file_obj.seek(0)

            line = self.file_obj.readline()

            mnemonics = line.split()

            values = []

            k = 0
            for m in mnemonics:
                m = m.strip()
                if name in m:
                    idxpar = str.find(m, "(")
                    idxpar2 = str.find(m, ")")
                    c = {"mnemonic": m[:idxpar], "index": k, "unit": m[idxpar + 1:idxpar2],
                         "description": "", "seisfast": ""}
                    values.append(c)
                k += 1

        return values

    def get_mnemonics(self):

        if self.sections[2]:

            curve = read_section("C", self.file_obj)

            values = []

            for c in curve:
                values.append(c["mnemonic"])

        else:

            self.file_obj.seek(0)

            line = self.file_obj.readline()

            mnemonics = line.split()

            values = []

            for m in mnemonics:
                m = m.strip()
                idxpar = str.find(m, "(")
                values.append(m[:idxpar])

        return values

    # single log

    def get_log(self, k, ignore_null=False):

        index_log = read_log(self.file_obj, 0, self.sections[5])

        if str(k).isdigit():
            log = read_log(self.file_obj, k, self.sections[5])
        else:
            curve = read_section("C", self.file_obj)

            values = []

            for c in curve:
                values.append(c["mnemonic"])

            log = read_log(self.file_obj, values.index(k), self.sections[5])

        if ignore_null:

            result = []
            index_result = []

            for i in range(len(log)):
                if not np.isclose(self.null, log[i]):
                    result.append(log[i])
                    index_result.append(index_log[i])

        else:
            result = log
            index_result = index_log

        return np.array(result), np.array(index_result)

    # well information

    def get_start(self):

        well = read_section("W", self.file_obj)

        value = None

        for w in well:
            if "start" in w["description"].lower():
                return convert_number(w["seisfast"])

        return value

    def get_stop(self):

        well = read_section("W", self.file_obj)

        value = None

        for w in well:
            if "stop" in w["description"].lower():
                return convert_number(w["seisfast"])

        return value

    def get_step(self):

        well = read_section("W", self.file_obj)

        value = None

        for w in well:
            if "step" in w["description"].lower():
                return convert_number(w["seisfast"])

        return value

    def get_null(self):

        if self.null is None:

            well = read_section("W", self.file_obj)

            value = None

            for w in well:
                if "null" in w["description"].lower():
                    return convert_number(w["seisfast"])

        else:

            value = self.null

        return value

    def set_null(self, value):

        self.null = value

    def get_latitude(self):

        well = read_section("W", self.file_obj)

        value = None

        for w in well:
            if "latitude" in w["description"].lower():
                return convert_number(w["seisfast"])

        return value

    def get_longitude(self):

        well = read_section("W", self.file_obj)

        value = None

        for w in well:
            if "longitude" in w["description"].lower():
                return convert_number(w["seisfast"])

        return value

    def get_location(self):

        well = read_section("W", self.file_obj)

        value = None

        for w in well:
            if "location" in w["description"].lower():
                return convert_number(w["seisfast"])

        return value

    def get_name(self):

        well = read_section("W", self.file_obj)

        value = None

        for w in well:
            if "well" in w["description"].lower():
                return convert_number(w["seisfast"])

        return value

    def get_info(self):

        print("\n------------")
        print("section info")
        print("------------")

        print("version - " + str(self.sections[0]))
        print("well - " + str(self.sections[1]))
        print("curve - " + str(self.sections[2]))
        print("parameter - " + str(self.sections[3]))
        print("other - " + str(self.sections[4]))
        print("seisfast - " + str(self.sections[5]))

        print("\n----------")
        print("curve info")
        print("----------")

        m = self.get_mnemonics()

        print(str(len(m)) + " curves")

        print(str(m))

        log = self.get_log(0)[0]

        print(str(len(log)) + " samples")

        print("\n---------")
        print("well info")
        print("---------")

        if self.sections[2]:
            curve = self.get_curve_section()
            print("units: " + str(curve[0]["units"]))
        else:
            self.file_obj.seek(0)
            line = self.file_obj.readline()
            mnemonics = line.split()
            m = mnemonics[0].strip()
            idxpar = str.find(m, "(")
            idxpar2 = str.find(m, ")")
            print("units: " + str(m[idxpar + 1:idxpar2]))

        if self.sections[1]:
            print("name: " + str(self.get_name()))

            print("start: " + str(self.get_start()))

            print("stop: " + str(self.get_stop()))

            print("step: " + str(self.get_step()))

            print("latitude: " + str(self.get_latitude()))

            print("longitude: " + str(self.get_longitude()))

            print("location: " + str(self.get_location()))


def read(filename):
    """Reads a LAS file.

    Parameters
    ----------

    filename : string
        Input file name.

    Returns
    -------

    output : object
        LAS object.
    """

    las = lasio.read(filename)

    return las
