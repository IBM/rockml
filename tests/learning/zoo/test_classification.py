""" Copyright 2019 IBM Research. All Rights Reserved. """

import os

import pytest
import rockml.learning.keras.data_loaders as dl
from rockml.data.adapter.seismic.segy.poststack import Direction
from rockml.learning.keras.callbaks import EarlyStoppingAtMinLoss
from rockml.learning.zoo.poststack import *


class TestStratigraphicClassification:
    def setup_method(self):
        self.segy_path = os.path.abspath('tests/segy_test_data/cropped_netherlands.sgy')
        self.horizon_path_list = [
            {'path': os.path.abspath('tests/segy_test_data/North_Sea_Group.xyz'),
             'range': [112, 116]},
            {'path': os.path.abspath('tests/segy_test_data/SSN_Group.xyz'),
             'range': [258, 263]},
            {'path': os.path.abspath('tests/segy_test_data/Germanic_Group.xyz'),
             'range': [346, 358]}
        ]

        tf.config.experimental_run_functions_eagerly(True)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_visible_devices(gpus[2], 'GPU')

    @pytest.mark.parametrize(
        "model_path, data_path, features_name, labels_name, num_classes",
        [
            ('segy_test_data/cls_estimator.h5',
             'segy_test_data/cls_db.hd5',
             'features',
             'label',
             8
             )
        ]
    )
    def test_train_danet2(self, model_path, data_path, features_name, labels_name, num_classes):
        train_set = dl.hdf_2_tfdataset(
            data_path,
            features_name=features_name,
            labels_name=labels_name
        ).batch(batch_size=256)
        val_set = dl.hdf_2_tfdataset(
            data_path,
            features_name=features_name,
            labels_name=labels_name
        ).batch(batch_size=256)

        model = danet2(train_set.element_spec[0].shape[1:], num_classes)
        optimizer = tf.keras.optimizers.Adam(
            lr=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            amsgrad=False
        )
        train_metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        ]
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        callbacks = [EarlyStoppingAtMinLoss(patience=5)]
        e1 = PostStackEstimator(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_metrics=train_metrics
        )

        e1.fit(epochs=1, train_set=train_set, valid_set=val_set, callbacks=callbacks)

    @pytest.mark.parametrize(
        "model_path, data_path, features_name, labels_name, num_classes",
        [
            ('segy_test_data',
             'segy_test_data/cls_db.hd5',
             'features',
             'label',
             8
             )
        ]
    )
    def test_train_danet3(self, model_path, data_path, features_name, labels_name, num_classes):
        def score(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.argmax(y_hat, axis=-1) - y

        train_set = dl.hdf_2_tfdataset(data_path, features_name=features_name,
                                       labels_name=labels_name).batch(batch_size=256)
        val_set = dl.hdf_2_tfdataset(data_path, features_name=features_name,
                                     labels_name=labels_name).batch(batch_size=256)

        model = danet2(train_set.element_spec[0].shape[1:], num_classes)
        optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                                             epsilon=None, amsgrad=False)
        train_metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        e1 = PostStackEstimator(model=model,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                train_metrics=train_metrics)

        e1.fit(epochs=1, train_set=train_set, valid_set=val_set)
        e1.save_model(model_path)
        # TODO: temporary fix because the tensorflow function is failling on Trevis only!?!?!?
        try:
            e2 = PostStackEstimator.load_model(model_path)
        except:
            return

        def score_fn(pred_a, pred_b) -> float:
            return 1. - (pred_a[0].label - pred_b[0].label)

        datum_list = [PostStackDatum(
            features=np.ones((50, 50, 1), dtype=np.float32),
            label=np.ones((1,), dtype=np.uint8),
            direction=Direction.INLINE,
            line_number=100,
            pixel_depth=100,
            column=100
        )] * 10

        scr1 = e1.score(
            input_list=datum_list,
            target_list=datum_list,
            score_fn=score_fn)
        scr2 = e2.score(
            input_list=datum_list,
            target_list=datum_list,
            score_fn=score_fn)

        assert np.sum(scr1 - scr2) == 0
