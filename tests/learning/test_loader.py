""" Copyright 2019 IBM Research. All Rights Reserved. """

import os

import numpy as np
import pytest
from rockml.learning.keras.data_loaders import hdf_2_tfdataset


class TestTrainClassification:
    def setup_method(self):
        self.path = os.path.abspath('segy_test_data/dataset.h5')

    @pytest.mark.parametrize(
        "features_name, labels_name",
        [
            ('features',
             'label')
        ]
    )
    def test_train_model(self, features_name, labels_name):
        dataset = hdf_2_tfdataset(self.path, features_name, labels_name)

        feat, label = next(iter(dataset.take(1)))

        assert dataset.element_spec[0].dtype.as_numpy_dtype == np.float32
