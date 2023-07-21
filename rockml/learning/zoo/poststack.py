""" Copyright 2023 IBM Research. All Rights Reserved.
"""

from typing import List, Tuple

import numpy as np
import rockml.learning.keras.nn_ops as ops
import tensorflow as tf
from rockml.data.adapter.seismic.segy.poststack import PostStackDatum
from rockml.learning.keras import KerasEstimator
from rockml.learning.keras.nn_ops import residual_block, residual_block_transposed
from tensorflow.keras import layers as kl
from tensorflow.keras import models as km


class PostStackEstimator(KerasEstimator):
    def __init__(self, model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_fn: tf.keras.losses.Loss,
                 train_metrics: List[tf.keras.metrics.Metric]):
        super().__init__(model, optimizer, loss_fn, train_metrics)

    def apply(self, datum_list: List[PostStackDatum], batch_size: int = 1) -> List[PostStackDatum]:
        features = np.stack([datum.features for datum in datum_list], axis=0)
        predictions = self.model.predict(features, batch_size=batch_size)
        for idx, element in enumerate(predictions):
            datum_list[idx].label = element
        return datum_list

    @staticmethod
    def load_model(path: str, is_best: bool = False, my_type=None):
        return KerasEstimator.load_model(path, is_best, PostStackEstimator)


def danet2(input_shape: Tuple[int, ...], num_classes: int, name: str = 'danet2') -> km.Model:
    conv_default = {'use_bias': True,
                    'activation': 'relu',
                    'kernel_initializer': 'glorot_uniform',
                    'bias_initializer': 'zeros',
                    'padding': 'same'}

    input_tensor = kl.Input(input_shape)

    tensor = kl.Conv2D(filters=64, kernel_size=5, strides=2, **conv_default)(input_tensor)
    tensor = kl.BatchNormalization()(tensor)
    tensor = kl.Conv2D(filters=128, kernel_size=3, strides=2, **conv_default)(tensor)
    tensor = kl.BatchNormalization()(tensor)
    tensor = kl.Conv2D(filters=128, kernel_size=3, strides=1, **conv_default)(tensor)
    tensor = kl.BatchNormalization()(tensor)
    tensor = kl.Conv2D(filters=128, kernel_size=3, strides=1, **conv_default)(tensor)
    tensor = kl.BatchNormalization()(tensor)
    tensor = kl.Conv2D(filters=256, kernel_size=3, strides=2, **conv_default)(tensor)
    tensor = kl.BatchNormalization()(tensor)
    tensor = kl.Conv2D(filters=256, kernel_size=3, strides=1, **conv_default)(tensor)
    tensor = kl.BatchNormalization()(tensor)
    tensor = kl.Conv2D(filters=256, kernel_size=3, strides=1, **conv_default)(tensor)
    tensor = kl.BatchNormalization()(tensor)

    tensor = kl.Dense(2048, activation='relu')(kl.Flatten()(tensor))
    tensor = kl.BatchNormalization()(tensor)
    tensor = kl.Dense(2048, activation='relu')(tensor)
    tensor = kl.Dropout(rate=0.5)(tensor)
    tensor = kl.Dense(num_classes, activation='sigmoid')(tensor)

    return km.Model(inputs=[input_tensor], outputs=[tensor], name=name)


def danet3(input_shape: Tuple[int, ...], num_classes: int, name: str = 'danet3') -> km.Model:
    input_tensor = kl.Input(input_shape)
    tensor = residual_block(input_tensor=input_tensor, filters=64, strides=1, name='a1')
    tensor = residual_block(input_tensor=tensor, filters=64, strides=2, name='a2')
    tensor = residual_block(input_tensor=tensor, filters=128, strides=1, name='b1')
    tensor = residual_block(input_tensor=tensor, filters=128, strides=2, name='b2')
    tensor = residual_block(input_tensor=tensor, filters=256, strides=1, name='c1')
    tensor = residual_block(input_tensor=tensor, filters=256, strides=2, name='c2')
    tensor = kl.Dense(num_classes, activation='sigmoid')(kl.Flatten()(tensor))

    return km.Model(inputs=[input_tensor], outputs=[tensor], name=name)


def unet(input_shape, output_channels):
    """ Build U-Net model

    Args:
        input_shape:
        output_channels:

    Returns:

    See: https://www.kaggle.com/advaitsave/tensorflow-2-nuclei-segmentation-unet

    """

    def encode_block(in_enc, filters: int, dropout: float):
        conv = kl.Conv2D(filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(in_enc)
        conv = kl.Dropout(dropout)(conv)
        conv = kl.Conv2D(filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv)
        pool = kl.MaxPooling2D((2, 2))(conv)
        return conv, pool

    def decode_block(in_dec, skip_origin, filters: int, dropout: float):
        up = kl.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(in_dec)
        up = kl.concatenate([up, skip_origin])
        conv = kl.Conv2D(filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(up)
        conv = kl.Dropout(dropout)(conv)
        conv = kl.Conv2D(filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv)
        return conv, up

    inputs = kl.Input(input_shape)

    c1, p1 = encode_block(inputs, filters=16, dropout=0.1)
    c2, p2 = encode_block(p1, filters=32, dropout=0.1)
    c3, p3 = encode_block(p2, filters=64, dropout=0.2)
    c4, p4 = encode_block(p3, filters=128, dropout=0.2)

    c5 = kl.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = kl.Dropout(0.3)(c5)
    c5 = kl.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    c6, u6 = decode_block(c5, c4, filters=128, dropout=0.2)
    c7, u7 = decode_block(c6, c3, filters=64, dropout=0.2)
    c8, u8 = decode_block(c7, c2, filters=32, dropout=0.1)
    c9, u9 = decode_block(c8, c1, filters=16, dropout=0.1)

    outputs = kl.Conv2D(output_channels, (1, 1), activation='softmax')(c9)

    model = km.Model(inputs=[inputs], outputs=[outputs])
    return model


def danet2fcn(input_shape, output_channels) -> km.Model:
    params = {'use_bias': True,
              'kernel_initializer': 'glorot_uniform',
              'padding': 'same',
              'bias_initializer': 'zeros'}

    input_tensor = kl.Input(input_shape)
    t1 = kl.Conv2D(filters=64, kernel_size=5, strides=2, **params)(input_tensor)
    t1 = tf.keras.activations.relu(kl.BatchNormalization()(t1))
    t2 = kl.Conv2D(filters=128, kernel_size=3, strides=2, **params)(t1)
    t2 = tf.keras.activations.relu(kl.BatchNormalization()(t2))
    t2 = kl.Conv2D(filters=128, kernel_size=3, strides=1, **params)(t2)
    t2 = tf.keras.activations.relu(kl.BatchNormalization()(t2))
    t2 = kl.Conv2D(filters=128, kernel_size=3, strides=1, **params)(t2)
    t2 = tf.keras.activations.relu(kl.BatchNormalization()(t2))
    t3 = kl.Conv2D(filters=256, kernel_size=3, strides=2, **params)(t2)
    t3 = tf.keras.activations.relu(kl.BatchNormalization()(t3))
    t3 = kl.Conv2D(filters=256, kernel_size=3, strides=1, **params)(t3)
    t3 = tf.keras.activations.relu(kl.BatchNormalization()(t3))
    t3 = kl.Conv2D(filters=256, kernel_size=3, strides=1, **params)(t3)
    t3 = tf.keras.activations.relu(kl.BatchNormalization()(t3))

    dec = kl.Conv2DTranspose(filters=256, kernel_size=3, strides=1, **params)(t3)
    dec = tf.keras.activations.relu(kl.BatchNormalization()(dec))
    dec = kl.Conv2DTranspose(filters=256, kernel_size=3, strides=1, **params)(dec)
    dec = tf.keras.activations.relu(kl.BatchNormalization()(dec))
    dec = kl.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same')(dec)
    dec = tf.keras.activations.relu(kl.BatchNormalization()(dec))
    dec = ops.crop_border(dec, t2.shape)
    dec = kl.Conv2DTranspose(filters=128, kernel_size=3, strides=1, **params)(dec)
    dec = tf.keras.activations.relu(kl.BatchNormalization()(dec))
    dec = kl.Conv2DTranspose(filters=128, kernel_size=3, strides=1, **params)(dec)
    dec = tf.keras.activations.relu(kl.BatchNormalization()(dec))
    dec = kl.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')(dec)
    dec = tf.keras.activations.relu(kl.BatchNormalization()(dec))
    dec = ops.crop_border(dec, t1.shape)
    dec = kl.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same')(dec)
    dec = tf.keras.activations.relu(kl.BatchNormalization()(dec))
    dec = ops.crop_border(dec, input_tensor.shape)
    outputs = kl.Conv2D(output_channels, (1, 1), activation='softmax')(dec)

    model = km.Model(inputs=[input_tensor], outputs=[outputs])
    return model


def danet3fcn(input_shape, output_channels) -> km.Model:
    params = {'use_bias': True,
              'activation': 'relu',
              'kernel_initializer': 'glorot_uniform',
              'bias_initializer': 'zeros'}

    input_tensor = kl.Input(input_shape)
    enc1 = kl.Conv2D(filters=64, kernel_size=5, strides=2, padding='same', **params)(input_tensor)
    enc2 = residual_block(input_tensor=enc1, filters=64, strides=1, name='a1')
    enc2 = residual_block(input_tensor=enc2, filters=64, strides=2, name='a2')
    enc3 = residual_block(input_tensor=enc2, filters=128, strides=1, name='b1')
    enc3 = residual_block(input_tensor=enc3, filters=128, strides=2, name='b2')
    enc4 = residual_block(input_tensor=enc3, filters=256, strides=1, name='c1')
    enc4 = residual_block(input_tensor=enc4, filters=256, strides=2, name='c2')

    dec = residual_block_transposed(input_tensor=enc4, filters=256, strides=2, name='ct2')
    dec = ops.crop_border(dec, enc3.shape)
    dec = residual_block_transposed(input_tensor=dec, filters=256, strides=1, name='ct1')
    dec = residual_block_transposed(input_tensor=dec, filters=128, strides=2, name='bt2')
    dec = ops.crop_border(dec, enc2.shape)
    dec = residual_block_transposed(input_tensor=dec, filters=128, strides=1, name='bt1')
    dec = residual_block_transposed(input_tensor=dec, filters=64, strides=2, name='at2')
    dec = ops.crop_border(dec, enc1.shape)
    dec = residual_block_transposed(input_tensor=dec, filters=64, strides=1, name='at1')
    dec = kl.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same')(dec)
    dec = kl.Conv2D(output_channels, (1, 1), activation='softmax')(dec)

    outputs = ops.crop_border(dec, input_tensor.shape)

    model = km.Model(inputs=[input_tensor], outputs=[outputs])
    return model
