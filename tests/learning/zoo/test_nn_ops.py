""" Copyright 2019 IBM Research. All Rights Reserved. """

import os

import pytest
from rockml.learning.keras.nn_ops import *


class TestNNOps:
    def setup_method(self):
        tf.config.experimental_run_functions_eagerly(True)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def test_residual(self):
        in_tensor = tf.random.uniform((1, 10, 10, 32))
        rb = residual_block(in_tensor, filters=32, strides=1, name='a')
        assert in_tensor.shape == rb.shape

        rb = residual_block(in_tensor, filters=33, strides=1, name='a')
        assert rb.shape == (1, 10, 10, 33)

        rb = residual_block(in_tensor, filters=32, strides=2, name='a')
        assert rb.shape == (1, 5, 5, 32)

        rb = residual_block(in_tensor, filters=35, strides=2, name='a')
        assert rb.shape == (1, 5, 5, 35)

        rb = residual_block(in_tensor, filters=35, strides=3, name='a')
        assert rb.shape == (1, 4, 4, 35)

    def test_bottleneck(self):
        in_tensor = tf.random.uniform((1, 10, 10, 32))
        rb = residual_bottleneck(in_tensor, filters=32, strides=1, name='a')
        assert in_tensor.shape == rb.shape

        rb = residual_bottleneck(in_tensor, filters=33, strides=1, name='a')
        assert rb.shape == (1, 10, 10, 33)

        rb = residual_bottleneck(in_tensor, filters=32, strides=2, name='a')
        assert rb.shape == (1, 5, 5, 32)

        rb = residual_bottleneck(in_tensor, filters=35, strides=2, name='a')
        assert rb.shape == (1, 5, 5, 35)

        rb = residual_bottleneck(in_tensor, filters=35, strides=3, name='a')
        assert rb.shape == (1, 4, 4, 35)

    def test_crop_border(self):
        current_tensor = tf.random.uniform((1, 10, 10, 32))
        current_tensor = crop_border(current_tensor, (1, 10, 10, 32))
        assert (1, 10, 10, 32) == current_tensor.shape

        current_tensor = tf.random.uniform((1, 11, 11, 32))
        current_tensor = crop_border(current_tensor, (1, 10, 10, 32))
        assert (1, 10, 10, 32) == current_tensor.shape

        current_tensor = tf.random.uniform((1, 9, 9, 32))
        with pytest.raises(AssertionError):
            assert crop_border(current_tensor, (1, 10, 10, 32))
