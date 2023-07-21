""" Copyright 2023 IBM Research. All Rights Reserved.

    Example
    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.6):
          print("\nReached 60% accuracy so cancelling training!")
          self.model.stop_training = True

    mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
"""

import collections
import csv

import numpy as np
import tensorflow as tf
from tensorflow.python.util.compat import collections_abc


class CSVLogger(tf.keras.callbacks.CSVLogger):
    def __init__(self, filename, append=False):
        super().__init__(filename, append=append)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, collections_abc.Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ['epoch'] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=fieldnames,
                dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs.get(key))) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.

    See: https://www.tensorflow.org/guide/keras/custom_callback
    """

    def __init__(self, patience: int = 0):
        super(EarlyStoppingAtMinLoss, self).__init__()

        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('loss')
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    """ See: https://www.tensorflow.org/guide/keras/custom_callback

    """

    def on_train_batch_end(self, batch, logs=None):
        print(f"For batch {batch}, loss is {logs['loss']:7.2f}.")

    def on_test_batch_end(self, batch, logs=None):
        print(f"For batch {batch}, loss is {logs['loss']:7.2f}.")

    def on_epoch_end(self, epoch, logs=None):
        print(f"The average loss for epoch {epoch} is {logs['loss']:7.2f}.")
