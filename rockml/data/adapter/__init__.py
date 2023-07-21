""" Copyright 2023 IBM Research. All Rights Reserved.
"""

from typing import List, Union, Callable

import tensorflow as tf

# Defining names for ML datasets

FEATURE_NAME = 'features'
LABEL_NAME = 'label'


class Datum(object):
    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        return str(self)


class DataDumper(object):
    """ Class for dumping Datum list to disk. """

    @staticmethod
    def bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def to_tfrecords(datum_list: List[Datum], path: str, serialize: Callable):
        with tf.io.TFRecordWriter(path) as writer:
            for idx, datum in enumerate(datum_list):
                writer.write(tf.train.Example(features=tf.train.Features(feature=serialize(datum))).SerializeToString())


class BaseDataAdapter(object):
    """ Base adapter class. """

    def __iter__(self) -> Datum:
        raise NotImplementedError()

    def __getitem__(self, key: Union[int, slice, list]) -> Union[Datum, List[Datum]]:
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
