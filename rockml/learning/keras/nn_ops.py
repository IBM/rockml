""" Copyright 2023 IBM Research. All Rights Reserved.
"""
import tensorflow as tf
from tensorflow.keras import layers as kl


def residual_bottleneck(input_tensor: tf.Tensor, filters: int, strides: int = 1,
                        name: str = 'residual_bottleneck') -> tf.Tensor:
    # Defining name basis
    conv_base_name = f'{name}_branch_'
    bn_name_base = f'bn_{name}_branch_'

    # First component of main path
    tensor = kl.Conv2D(filters=filters, kernel_size=1, strides=strides, padding='valid',
                       name=f'{conv_base_name}2a')(input_tensor)
    tensor = kl.BatchNormalization(axis=3, name=f'{bn_name_base}2a')(tensor)
    tensor = kl.Activation('relu')(tensor)

    # Second component of main path
    tensor = kl.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                       name=f'{conv_base_name}2b')(tensor)
    tensor = kl.BatchNormalization(axis=3, name=f'{bn_name_base}2b')(tensor)
    tensor = kl.ReLU()(tensor)

    # Third component of main path
    tensor = kl.Conv2D(filters=filters, kernel_size=1, strides=1, padding='valid',
                       name=f'{conv_base_name}2c')(tensor)
    tensor = kl.BatchNormalization(axis=3, name=f'{bn_name_base}2c')(tensor)

    # Shortcut fix for `s == 2`
    if input_tensor.shape != tensor.shape:
        input_tensor = kl.Conv2D(filters=filters, kernel_size=1, strides=strides, padding='valid',
                                 name=f'{conv_base_name}1')(input_tensor)
        input_tensor = kl.BatchNormalization(axis=3, name=f'{bn_name_base}1')(input_tensor)

    tensor = kl.Add()([tensor, input_tensor])
    tensor = kl.ReLU()(tensor)

    return tensor


def residual_block(input_tensor: tf.Tensor, filters: int, strides: int = 1,
                   name: str = 'residual_block') -> tf.Tensor:
    # Defining name basis
    conv_base_name = f'{name}_branch_'
    bn_name_base = f'bn_{name}_branch_'

    # First component of main path
    tensor = kl.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same',
                       name=f'{conv_base_name}2a')(input_tensor)
    tensor = kl.BatchNormalization(axis=3, name=f'{bn_name_base}2a')(tensor)
    tensor = kl.ReLU()(tensor)

    # Second component of main path
    tensor = kl.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                       name=f'{conv_base_name}2b')(tensor)
    tensor = kl.BatchNormalization(axis=3, name=f'{bn_name_base}2b')(tensor)
    tensor = kl.ReLU()(tensor)

    # Shortcut fix for `s == 2`
    if input_tensor.shape != tensor.shape:
        input_tensor = kl.Conv2D(filters=filters, kernel_size=1, strides=strides, padding='valid',
                                 name=f'{conv_base_name}1')(input_tensor)
        input_tensor = kl.BatchNormalization(axis=3, name=f'{bn_name_base}1')(input_tensor)

    tensor = kl.Add()([tensor, input_tensor])
    tensor = kl.ReLU()(tensor)

    return tensor


def residual_block_transposed(input_tensor: tf.Tensor, filters: int, strides: int = 1,
                              name: str = 'residual_block_transposed') -> tf.Tensor:
    # Defining name basis
    conv_base_name = f'{name}_branch_'
    bn_name_base = f'bn_{name}_branch_'

    # First component of main path
    tensor = kl.Conv2DTranspose(filters=filters, kernel_size=3, strides=strides, padding='same',
                                name=f'{conv_base_name}2a')(input_tensor)
    tensor = kl.BatchNormalization(axis=3, name=f'{bn_name_base}2a')(tensor)
    tensor = kl.ReLU()(tensor)

    # Second component of main path
    tensor = kl.Conv2DTranspose(filters=filters, kernel_size=3, strides=1, padding='same',
                                name=f'{conv_base_name}2b')(tensor)
    tensor = kl.BatchNormalization(axis=3, name=f'{bn_name_base}2b')(tensor)
    tensor = kl.ReLU()(tensor)

    # Shortcut fix for `s == 2`
    if input_tensor.shape != tensor.shape:
        input_tensor = kl.Conv2DTranspose(filters=filters, kernel_size=1, strides=strides, padding='same',
                                          name=f'{conv_base_name}1')(input_tensor)
        input_tensor = kl.BatchNormalization(axis=3, name=f'{bn_name_base}1')(input_tensor)

    tensor = kl.Add()([tensor, input_tensor])
    tensor = kl.ReLU()(tensor)

    return tensor


def crop_border(tensor, shape):
    """ This method fixes the size of output tensors from transposed convolutions with
        even sizes, e.g., conv_t of shape (7, 7) would be 14 instead of 13.
    Args:
        tensor: input tensor 4D (NHWC).
        shape: desired shape for the tensor as a list.
    Returns:
        output tensor
    """

    diff = []
    new_tensor = tensor
    tensor_shape = tensor.get_shape().as_list()
    for i in [1, 2]:
        d = tensor_shape[i] - shape[i]
        assert d >= 0
        diff.append(d)

    if diff[0] != 0:
        new_tensor = tensor[:, :-diff[0], :, :]

    if diff[1] != 0:
        new_tensor = new_tensor[:, :, :-diff[1], :]

    return new_tensor
