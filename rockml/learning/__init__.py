""" Copyright 2019 IBM Research. All Rights Reserved.

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md

https://www.tensorflow.org/guide/function#debugging
tf.config.experimental_run_functions_eagerly(True)

Seismic experiments TF
https://github.ibm.com/BRLML/seismic_experiments/tree/76abd7edb1150c8a12b5fa33f258057bb9654f72
"""
from abc import abstractmethod, ABC
from typing import Callable
from typing import List, Any

from rockml.data.adapter import Datum


class Estimator(ABC):
    """ Why to use this instead of tf.estimator? -> because we can encapsulate provenance, horovod and provide
        a common API for seisfast and learning. The idea here is that the user will only provide a Keras model initially.
        Then, the user will provide a train_loader and a valid_loader for training, and provide a test_loader for the
        score function to test model's performance.

        Additionally, save and load model functions may provide more options to work with workflow.

        maybe horovod specs and other config parameters should be included here
    """

    @abstractmethod
    def fit(self, **kwargs):
        """ This method encapsulates all steps of a model training, including optimization
            parameters, loss function, training and validation metrics, model update, etc.

        Args:
            **kwargs: The user is free to add any necessary parameters.

        Returns: It may return some training history information.

        """
        pass

    @abstractmethod
    def apply(self, datum_list: List[Datum], **kwargs) -> List[Datum]:
        """ Applies the model in a set of features. It is also known as inference, prediction, run, etc.

        Args:
            datum_list: List[Datum]. A set of features in the format [n, shape], where *n* is the number of
                examples.

        Returns: List[Datum]. It returns another array with the predictions.

        """
        pass

    def score(self, input_list: List[Datum], target_list: List[Datum],
              score_fn: Callable[[List[Datum], List[Datum]], Any], **kwargs):
        """ This method also known as testing phase, computes the models performance against labels on the features
            dataset using the score function provided.

        Args:
            input_list: List[Datum]. A set of features in the format [n, shape], where *n* is the number of
                examples.
            target_list: List[Datum]. A set of labels in the format [n, shape], where *n* is the number of
                examples.
            score_fn: function.
            kwargs: dict. passed to the apply function.

        Returns: The results are specified by the score function.

        """
        return score_fn(self.apply(input_list, **kwargs), target_list)

    @abstractmethod
    def save_model(self, path: str) -> None:
        """ Saves the estimator together with its current state.

        Args:
            path: str. Path where the file will be saved.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_model(path: str):
        """ Loads the estimator on the saved state.

        Args:
            path: str. Path to the file.

        Returns: Estimator. Returns an instance of Estimator on saved state.

        """
        pass
