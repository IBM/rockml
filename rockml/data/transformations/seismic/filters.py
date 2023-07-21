""" Copyright 2019 IBM Research. All Rights Reserved.

    - Filters classes for seismic seisfast.
"""

from typing import Union

import numpy as np
from rockml.data.adapter.seismic.segy.poststack import PostStackDatum
from rockml.data.transformations import Transformation

""" Filters are a special kind of Transformation. These callable objects can return None. """


class MinimumTextureFilter(Transformation):
    def __init__(self, min_texture_in_features: float):
        """ Initialize MinimumTextureFilter.

        Args:
            min_texture_in_features: float in the range [0.0, 1.0] representing the minimum amount of texture
                that should be present in a a feature np.ndarray.
        """

        self.min_texture_in_features = min_texture_in_features

    def __call__(self, datum: PostStackDatum) -> Union[PostStackDatum, None]:
        """ Verifies whether *datum*.features has a user-defined minimum amount of seisfast with texture
            (self.min_texture_in_feature). In many cases the images have many areas with no acquisition seisfast. This
            function calculates the standard deviation for rows and columns checking if it has the minimum non-zero
            elements.

            Args:
                datum: PostStackDatum in which features is 3D np.ndarray (channel axis at the end) and label is a
                    2D np.ndarray.

            Returns:
                new PostStackDatum with the same attributes if the minimum texture is satisfied, otherwise return None.
        """

        height, width, _ = datum.features.shape

        if np.std(datum.features) == 0:
            return None

        min_count_per_row = self.min_texture_in_features * height
        if np.count_nonzero(np.std(datum.features, axis=1)) < min_count_per_row:
            return None

        min_count_per_column = self.min_texture_in_features * width
        if np.count_nonzero(np.std(datum.features, axis=0)) < min_count_per_column:
            return None

        return PostStackDatum(features=datum.features, label=datum.label, direction=datum.direction,
                              line_number=datum.line_number, pixel_depth=datum.pixel_depth,
                              column=datum.column)

    def __str__(self):
        return f"<MinimumTextureFilter min_texture_in_tile: {self.min_texture_in_features}>"


class ClassificationFilter(Transformation):
    def __init__(self, noise: float):
        """ Initialize ClassificationFilter.

        Args:
            noise: float in the range [0.0, 1.0] representing the maximum allowed ratio of pixels from other classes.
        """

        self.noise = noise

    def __call__(self, datum: PostStackDatum) -> Union[PostStackDatum, None]:
        """ This function gets *datum* with a label mask and defines the classification label for *datum* based on the
            dominant category; it allows for a maximum noise (pixels from other classes) of *self.noise*.

            First, it gets the dominant class in *datum*.label and calculate the ratio of the *datum*.features area it
            represents, verifying if the noise restriction is satisfied. If yes, it returns a new PostStackDatum with
            the defined label, otherwise it returns None.

            Args:
                datum: PostStackDatum in which features is 3D np.ndarray (channel axis at the end) and label is a
                    2D np.ndarray.

            Returns:
                new PostStackDatum with classification label if the maximum noise restriction is satisfied, otherwise
                    return None.
        """

        unique, count = np.unique(datum.label, return_counts=True)
        class_label = unique[np.argmax(count)]
        feature_noise = 1 - (np.max(count) / (datum.features.size / datum.features.shape[-1]))

        if feature_noise > self.noise:
            return None

        return PostStackDatum(features=datum.features, label=class_label, direction=datum.direction,
                              line_number=datum.line_number, pixel_depth=datum.pixel_depth,
                              column=datum.column)

    def __str__(self):
        return f"<ClassificationFilter noise: {self.noise}>"
