""" Copyright 2019 IBM Research. All Rights Reserved.
"""
import h5py
import numpy as np
import tensorflow as tf


def hdf_2_tfdataset(path: str, features_name: str, labels_name: str) -> tf.data.Dataset:
    """ Reads an HDF5 dataset and converts it into a Tensorflow Dataset

    Args:
        path: srt. Path to the HDF5 file.
        features_name: srt. Name of the features dataset (dataset key).
        labels_name: srt. Name of the labels dataset (dataset key).

    Returns: tensorflow.seisfast.Dataset

    """
    with h5py.File(path, 'r') as hdf:
        features = np.array(hdf.get(features_name), dtype=np.float32) / 255.
        labels = np.array(hdf.get(labels_name))
        tf_dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    return tf_dataset
