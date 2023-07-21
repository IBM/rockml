""" Copyright 2023 IBM Research. All Rights Reserved.

    - Sampling functions and classes.
"""

from typing import Tuple

import numpy as np


class RandomSampler(object):
    def __init__(self, num_examples: int = None):
        """ Initialize RandomSampler.

        Args:
            num_examples: (int) number of examples
        """
        self.num_examples = num_examples

    def sample(self, dataset: list) -> list:
        """ Get a random sample of *dataset* with size = self.num_examples. If self.num_examples is bigger than
            *dataset* or it is None, *dataset* is only shuffled.

        Args:
            dataset: list of objects.

        Returns:
            new list of sampled objects.
        """
        if self.num_examples is None:
            self.num_examples = len(dataset)
        elif self.num_examples > len(dataset):
            self.num_examples = len(dataset)
            print('Warning!')

        return np.random.choice(dataset, size=self.num_examples, replace=False).tolist()


class ClassBalanceSampler(object):
    def __init__(self, num_examples_per_class: int = None):
        """ Initialize RandomSampler.

        Args:
            num_examples_per_class: (int) number of examples per class to be sampled
        """
        self.num_examples_per_class = num_examples_per_class

    def sample(self, dataset: list) -> list:
        """ Balance the dataset based on the labels. Each class is going to have the minimum between its quantity of
            elements and *self*.min_num_examples.

        Args:
            dataset: list of objects.

        Returns:
            new list of sampled objects.
        """
        labels = np.array([datum.label for datum in dataset])

        classes = {c: np.where(labels == c)[0] for c in np.unique(labels)}

        if self.num_examples_per_class is None:
            min_num_examples = min([c.size for c in classes.values()])
        else:
            min_num_examples = self.num_examples_per_class

        choices = [np.random.choice(class_idx, size=min(min_num_examples, len(class_idx)), replace=False)
                   for class_idx in classes.values()]

        return [dataset[idx] for choice in choices for idx in choice]


def split_dataset(dataset: list, valid_ratio: float) -> Tuple[list, list]:
    """ Randomly split *dataset* into train and validation sets using *valid_ratio*.

    Args:
        dataset: list of object in a dataset.
        valid_ratio: (float) ratio of examples that will go to the validation set.

    Returns:
        train set list and validation set list
    """

    idx = np.random.permutation(np.arange(len(dataset)))

    valid_ex = int(len(dataset) * valid_ratio)

    valid_set = [dataset[i] for i in idx[:valid_ex]]
    train_set = [dataset[i] for i in idx[valid_ex:]]

    return train_set, valid_set
