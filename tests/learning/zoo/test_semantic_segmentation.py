""" Copyright 2023 IBM Research. All Rights Reserved. """

import os

import pytest

import rockml.learning.keras.data_loaders as dl
from rockml.data.adapter.seismic.segy.poststack import Direction
from rockml.learning.keras import SparseMeanIoU
from rockml.learning.zoo.poststack import *


class TestStratigraphicSegmentation:
    def setup_method(self):
        pass
        tf.config.experimental_run_functions_eagerly(True)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_visible_devices(gpus[2], 'GPU')

    @pytest.mark.parametrize(
        "model_path, data_path, features_name, labels_name, num_classes",
        [
            ('segy_test_data',
             'segy_test_data/seg_test.h5',
             'features',
             'label',
             8
             )
        ]
    )
    def test_train_danet2fcn(self, model_path, data_path, features_name, labels_name, num_classes):
        # strategy = tf.distribute.MirroredStrategy()
        model_path = os.path.abspath(model_path)
        data_path = os.path.abspath(data_path)

        train_set = dl.hdf_2_tfdataset(data_path,
                                       features_name=features_name,
                                       labels_name=labels_name).batch(batch_size=32)
        val_set = dl.hdf_2_tfdataset(data_path,
                                     features_name=features_name,
                                     labels_name=labels_name).batch(batch_size=32)

        optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        train_metrics = [SparseMeanIoU(num_classes=num_classes)]
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        # with strategy.scope():
        model = danet2fcn(train_set.element_spec[0].shape[1:], num_classes)

        # est = self.training_loop(train_set, val_set, model, model_path, optimizer, loss_fn,
        #                          (train_metrics, val_metrics), epochs=10)
        e1 = PostStackEstimator(model=model,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                train_metrics=train_metrics)

        e1.fit(epochs=1, train_set=train_set, valid_set=val_set)
        # TODO: need to update saving function using new TF API
        # e1.save_model(model_path)
        # e2 = PostStackEstimator.load_model(model_path)
        #
        # datum_list = [PostStackDatum(
        #     features=np.ones((80, 120, 1), dtype=np.float32),
        #     label=np.ones((80, 120), dtype=np.uint8),
        #     direction=Direction.INLINE,
        #     line_number=100,
        #     pixel_depth=100,
        #     column=100
        # )] * 10
        #
        # e1_res = e1.apply(datum_list)
        # e2_res = e2.apply(datum_list)
        #
        # assert np.sum(e1_res[0].features - e2_res[0].features) == 0

    @pytest.mark.parametrize(
        "model_path, data_path, features_name, labels_name, num_classes",
        [
            ('segy_test_data/estimator.h5',
             'segy_test_data/seg_test.h5',
             'features',
             'label',
             8
             )
        ]
    )
    def test_train_danet3fcn(self, model_path, data_path, features_name, labels_name, num_classes):
        train_set = dl.hdf_2_tfdataset(data_path,
                                       features_name=features_name,
                                       labels_name=labels_name).batch(batch_size=32)
        val_set = dl.hdf_2_tfdataset(data_path,
                                     features_name=features_name,
                                     labels_name=labels_name).batch(batch_size=32)

        optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        train_metrics = [SparseMeanIoU(num_classes=num_classes)]
        val_metrics = [SparseMeanIoU(num_classes=num_classes)]
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        model = danet3fcn(train_set.element_spec[0].shape[1:], num_classes)

        e1 = PostStackEstimator(model=model,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                train_metrics=train_metrics)

        e1.fit(epochs=1, train_set=train_set, valid_set=val_set)
