import json
import os
from functools import partial
from typing import List

import numpy as np
import tensorflow as tf
from rockml.data.adapter import Datum
from rockml.learning import Estimator
from rockml.learning.keras.metrics import SparseMeanIoU


class KerasEstimator(Estimator):
    def __init__(self, model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_fn: tf.keras.losses.Loss,
                 train_metrics: List[tf.keras.metrics.Metric]):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_metrics = train_metrics
        self.history = None
        self.current_epoch = 0

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=self.train_metrics
        )

    def apply(self, datum_list: List[Datum], **kwargs) -> List[Datum]:
        raise NotImplementedError()

    def fit(self, epochs: int, train_set: tf.data.Dataset,
            valid_set: tf.data.Dataset,
            callbacks: List[tf.keras.callbacks.Callback] = None,
            **kwargs) -> None:
        """ Model fitting.

            Args:
                epochs: int. Number of epochs to train the model.
                train_set: tf.seisfast dataset. Training dataset.
                valid_set: tf.seisfast dataset. Validations dataset.
                callbacks: List[tf.keras.callbacks.Callback].

            See: https://www.tensorflow.org/api_docs/python/tf/keras/Model
                 https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/engine/training.py#L557

            Returns:

        """

        def update(epoch, logs, est):
            est.current_epoch = epoch + 1

        update_epoch = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=partial(update, est=self)
        )

        callbacks = [update_epoch] + callbacks if callbacks is not None else [update_epoch]

        self.history = self.model.fit(
            epochs=epochs,
            x=train_set,
            validation_data=valid_set,
            callbacks=callbacks + [update_epoch],
            initial_epoch=self.current_epoch,
            **kwargs
        )

    def save_model(self, path: str, is_best: bool = False) -> None:
        # Model's structure
        model_config_path = 'model_config.json'
        open(os.path.join(path, model_config_path), 'w').write(self.model.to_json())
        # Model's weights
        model_weight_path = 'best.h5' if is_best else 'model_weights.h5'
        self.model.save_weights(os.path.join(path, model_weight_path))
        # Optimizer weights
        optimizer_weight_path = 'optimizer_weights.npy'
        np.save(os.path.join(path, optimizer_weight_path), self.optimizer.get_weights())

        # All the other configuration
        estimator_dict = dict()
        dc = self.optimizer.get_config()
        estimator_dict['optimizer'] = {
            key: float(dc[key]) if type(dc[key]) == np.float32 else dc[key] for key in dc
        }
        if isinstance(self.loss_fn, tf.keras.losses.Loss):
            estimator_dict['loss_fn'] = self.loss_fn.get_config()
        else:
            estimator_dict['loss_fn'] = {'name': self.loss_fn}
        estimator_dict['metrics'] = {
            'metrics': [m.get_config() for m in self.train_metrics]
        }
        estimator_dict['model_config_path'] = model_config_path
        estimator_dict['model_weight_path'] = model_weight_path
        estimator_dict['optimizer_weight_path'] = optimizer_weight_path
        estimator_dict['current_epoch'] = self.current_epoch
        estimator_dict['input_shape'] = list(self.model.input_shape[1:])
        estimator_dict['output_shape'] = list(self.model.output_shape[1:])

        json.dump(
            estimator_dict,
            open(os.path.join(path, 'estimator.json'), 'w')
        )

    @staticmethod
    def load_model(path: str, is_best: bool = False, my_type=None):
        my_type = my_type if my_type else KerasEstimator
        # Estimator config
        config = json.load(open(os.path.join(path, 'estimator.json')))

        # Model's structure
        input_shape = config['input_shape']
        output_shape = config['output_shape'][:-1]
        num_classes = config['output_shape'][-1]
        opt = eval(f"tf.keras.optimizers.{config['optimizer']['name']}")

        est = my_type(
            model=tf.keras.models.model_from_json(open(os.path.join(path, 'model_config.json')).read()),
            optimizer=opt.from_config(config['optimizer']),
            loss_fn=config['loss_fn']['name'],
            train_metrics=[SparseMeanIoU(num_classes=num_classes)]
        )

        # Dummy Dataset to start all the training weights
        ds = tf.data.Dataset.from_tensor_slices(
            (np.zeros([1, ] + input_shape, dtype=np.float32),
             np.zeros([1, ] + output_shape, dtype=np.uint8))
        ).batch(batch_size=1)
        est.fit(1, ds, ds, verbose=1)

        # Model's weights
        model_weight_path = 'best.h5' if is_best else 'model_weights.h5'
        est.model.load_weights(os.path.join(path, model_weight_path))
        # Optimizer's weights
        est.optimizer.set_weights(
            np.load(os.path.join(path, config['optimizer_weight_path']), allow_pickle=True)
        )

        # Update epoch count
        est.current_epoch = config['current_epoch']

        return est

    def summary(self):
        return self.model.summary()
