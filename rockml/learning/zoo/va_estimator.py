""" Copyright 2020 IBM/GPN joint development. All Rights Reserved. """

from typing import List, Union, Tuple

import numpy as np
import tensorflow as tf
from rockml.data.adapter.seismic.segy.prestack import CDPGatherDatum
from rockml.learning import Estimator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class VelocityAdjustment(Estimator):
    def __init__(self, input_shape: Tuple[int, int, int], model: Union[tf.keras.Model, None] = None,
                 best_valid_model_path: str = '', learning_rate: float = 0.001, early_stop_patience: int = 50,
                 additional_callbacks: List[tf.keras.callbacks.Callback] = None):
        """ Initialize VelocityAdjustment Estimator.

        Args:
            input_shape: tuple containing the input shape in the format (height, width, channels).
            model: a compiled tf.keras.Model. If None, the Estimator will use its default Xception based model.
            best_valid_model_path: (str) path where to save the best validation model.
            learning_rate: (float) learning rate for the optimizer; ignored if loading a compiled model.
            additional_callbacks: list of additional callbacks (the Estimator already uses EarlyStopping and
                ModelCheckpoint to save the best validation model).
        """

        super(VelocityAdjustment).__init__()

        self.input_shape = input_shape

        if model is None:
            self.model = self._get_model(learning_rate)
        else:
            self.model = model

        callbacks = [EarlyStopping(monitor='val_loss', patience=early_stop_patience),
                     ModelCheckpoint(best_valid_model_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')]

        additional_callbacks = [] if additional_callbacks is None else additional_callbacks

        self.callbacks = callbacks + additional_callbacks

    def _get_model(self, learning_rate: float) -> Model:
        """ Extend the Xception model with a dense layer and compile the model with *self.optimizer* and 'mse' loss.

        Returns:
           compiled  Keras Model.
        """

        base_model = Xception(include_top=False, weights=None,
                              input_shape=self.input_shape, pooling='avg')
        x = base_model.output
        x = Dense(1, activation="linear")(x)

        model = Model(inputs=base_model.input, outputs=x)

        model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

        return model

    def fit(self, epochs: int, train_set: tf.data.Dataset, valid_set: tf.data.Dataset, **kwargs):
        """ Model fitting.

        Args:
            epochs: (int) number of epochs to train the model.
            train_set: tf.seisfast.Dataset representing the training seisfast.
            valid_set: tf.seisfast.Dataset representing the validation seisfast.

        Returns:
            history of the trained model.
        """

        return self.model.fit(x=train_set,
                              epochs=epochs,
                              shuffle=True,
                              callbacks=self.callbacks,
                              validation_data=valid_set,
                              **kwargs)

    def apply(self, datum_list: List[CDPGatherDatum], batch_size: int = 1,
              velocity_range: List[int] = None) -> List[CDPGatherDatum]:
        """ Apply *self.model* to *datum_list*, modifying its contents with the model results. If *velocity_range* is
            given, the velocity delta predicted by the network is only applied if it falls into this range.

        Args:
            datum_list: list of datums to be processed by the model.
            batch_size: (int) batch size to process seisfast.
            velocity_range: list in the format [min_velocity, max_velocity]; if None, no limitation is applied.

        Returns:
            datum list with additional information.
        """

        features = np.stack([datum.features for datum in datum_list], axis=0)
        # features = features.astype(np.float32) / 255.
        predictions = self.model.predict(features, batch_size=batch_size)

        for i in range(len(predictions)):
            delta_v = predictions[i, 0]
            new_v = datum_list[i].velocities[0][1] - delta_v

            # avoid negative values
            if velocity_range is not None:
                new_v = max(new_v, velocity_range[0])
                new_v = min(new_v, velocity_range[1])

            datum_list[i].velocities[0][1] = new_v
            datum_list[i].label = delta_v

        return datum_list

    def save_model(self, path: str) -> None:
        self.model.save(path)

    @staticmethod
    def load_model(path: str):
        """ Load a saved Keras model into an Estimator.

        Args:
            path: (str) path to the hdf5 file containing a complete Keras model.

        Returns:
            Estimator with loaded model.
        """

        model = tf.keras.models.load_model(path, custom_objects=None, compile=True)

        return VelocityAdjustment(model=model, input_shape=model.input_shape[1:])

    def summary(self):
        return self.model.summary()
