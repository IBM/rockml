""" Copyright 2023 IBM Research. All Rights Reserved.
"""


from typing import TextIO

import numpy as np


def read(file_path: str):
    """
    Reads velocity functions from file in *file_path*.

    Parameters
    ----------

    file_path: str
        path to the velocity file.

    Returns
    -------

    dict
        Velocity functions.

    """

    file = open(file_path)

    velocity_functions = {}
    key = None

    for line in file.readlines():

        terms = line.split()

        if not terms[0].isnumeric():
            key = str(terms[1]) + "_" + str(terms[2])
            velocity_functions[key] = []
            continue

        pair = []

        for t in terms:

            if len(pair) < 2:
                pair.append(int(t))
            else:
                velocity_functions[key].append(pair)
                pair = []
                pair.append(int(t))

        velocity_functions[key].append(pair)

    file.close()

    return velocity_functions


def write_velocity_function(file: TextIO, il: int, xl: int, times: np.ndarray, velocities: np.ndarray):
    """
    Writes velocity function.

    Parameters
    ----------

    file: obj
        file object.

    il: int
        Inline number.

    xl: int
        Crossline number.

    times: list
        Time values.

    velocities: list
        Velocity values.

    """

    space = "     "

    file.write("VFUNC " + space + str(il) + space + str(xl) + "\n")

    count = 0
    last_v = None

    for t, v in zip(times, velocities):

        if last_v is None:
            last_v = int(v)
        else:
            if int(v) == last_v:
                continue

        last_v = int(v)

        file.write(str(int(t)) + space + str(int(v)) + space)
        count += 1
        if count == 4:
            file.write("\n")
            count = 0

    if count != 0:
        file.write("\n")

    return
