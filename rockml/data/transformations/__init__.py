""" Copyright 2019 IBM Research. All Rights Reserved.

    - Defines the Transformation and Composer classes.
"""

import pickle
from abc import abstractmethod, ABC
from typing import List, Callable

from rockml.data.adapter import Datum


class Transformation(ABC):
    @abstractmethod
    def __call__(self, dataset):
        pass


class Lambda(Transformation):
    def __init__(self, function: Callable[[Datum], Datum], **kwargs):
        self.function = function
        self.kwargs = kwargs

    def __call__(self, dataset: Datum) -> Datum:
        return self.function(dataset, **self.kwargs)

    def __repr__(self):
        return f'<Lambda>'


class Composer(object):
    def __init__(self, transformations: List[Transformation]):
        """ Initialize Composer.

        Args:
            transformations: list of callable transformations.
        """
        self.transformations = transformations

    def __str__(self):
        return f"Composition[{','.join(map(str, self.transformations))}]"

    def dump(self, path: str):
        """ Dump the Composer object into a pickle file.

        Args:
            path: (str) path to save the Composer.
        """

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """ Load the Composer object from a pickle file.

        Args:
            path: (str) path to load the Composer.
        """

        with open(path, 'rb') as f:
            composer = pickle.load(f)

        return composer

    def apply(self, dataset):
        """ Apply all transformations in *self.transformations* list in *dataset*, which can be a single object or a
            list of objects to be transformed.

        Args:
            dataset: object or list of objects to be transformed.

        Returns:
            transformed dataset, which can be a single dataset object, a list of transformed objects or None.
        """

        for transformation in self.transformations:
            if type(dataset) != list:
                if dataset is None:
                    return None
                dataset = transformation(dataset)
            else:
                new_dataset = []
                for element in dataset:
                    transformed_element = transformation(element)
                    if type(transformed_element) != list:
                        if transformed_element is not None:
                            new_dataset.append(transformed_element)
                    else:
                        new_dataset.extend(transformed_element)

                dataset = new_dataset

        return dataset
