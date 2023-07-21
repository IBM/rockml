""" Copyright 2019 IBM Research. All Rights Reserved.

    - Test functions for seisfast.sampling.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from rockml.data.adapter.seismic.segy.poststack import Direction, _PostStackProps as props
from rockml.data.sampling import ClassBalanceSampler
from rockml.data.sampling import RandomSampler


def _get_stub_post_stack_datum(features, label, direction, line_number, pixel_depth, column):
    seismic_stub = Mock()
    seismic_stub.features = features
    seismic_stub.label = label
    seismic_stub.direction = direction
    seismic_stub.line_number = line_number
    seismic_stub.pixel_depth = pixel_depth
    seismic_stub.column = column
    return seismic_stub


def _get_stub_dataset_list(list_length):
    dataset_list = [None] * list_length
    for i in range(list_length):
        features = np.random.rand(5, 5)
        label = np.random.rand(5, 5)
        meta_data = {props.DIRECTION.value: Direction.INLINE,
                     props.LINE_NUMBER.value: np.random.randint(100, 200),
                     props.PIXEL_DEPTH.value: 0, props.COLUMN.value: 0}
        dataset_list[i] = _get_stub_post_stack_datum(features, label, **meta_data)

    return dataset_list


class TestRandomSampler:
    @pytest.mark.parametrize(
        "num_examples",
        [
            5
        ]
    )
    def test_random_sampler_init(self, num_examples):
        # Arrange:
        random_sampler_object = RandomSampler(num_examples)

        # Act:
        actual_num_examples = random_sampler_object.num_examples

        # Assert:
        assert actual_num_examples == num_examples

    @pytest.mark.parametrize(
        "num_examples, dataset_list, expected_dataset_list_length",
        [
            (5, _get_stub_dataset_list(5), 5),
            (3, _get_stub_dataset_list(5), 3),
            (7, _get_stub_dataset_list(5), 5),
            (None, _get_stub_dataset_list(7), 7),

            pytest.param(3, _get_stub_dataset_list(5), 5,
                         marks=pytest.mark.xfail(strict=True)),
            pytest.param(7, _get_stub_dataset_list(5), 7,
                         marks=pytest.mark.xfail(strict=True)),
            pytest.param(None, _get_stub_dataset_list(5), 7,
                         marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_random_sampler_sample(self, num_examples, dataset_list, expected_dataset_list_length):
        # Arrange:
        random_sampler_object = RandomSampler(num_examples)

        # Act:
        actual_dataset_list = random_sampler_object.sample(dataset_list)

        # Assert:
        assert len(actual_dataset_list) == expected_dataset_list_length


class TestClassBalanceSampler:
    @pytest.mark.parametrize(
        "num_examples_per_class",
        [
            5
        ]
    )
    def test_class_balance_sampler_init(self, num_examples_per_class):
        # Arrange:
        class_balance_sampler_object = ClassBalanceSampler(num_examples_per_class)

        # Act:
        num_examples_per_class = class_balance_sampler_object.num_examples_per_class

        # Assert:
        assert num_examples_per_class == num_examples_per_class

    @pytest.mark.parametrize(
        "num_examples_per_class, dataset_list, expected_dataset_list_length",
        [
            (2, [_get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 100, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 101, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 102, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 103, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 104, 0, 0),
                 ], 4),
            (3, [_get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 100, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 101, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 102, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 103, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 104, 0, 0),
                 ], 5),
            (1, [_get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 100, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 101, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 4, Direction.INLINE, 102, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 103, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 104, 0, 0),
                 ], 3),
            (3, [_get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 100, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 101, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 4, Direction.INLINE, 102, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 103, 0, 0),
                 _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 104, 0, 0),
                 ], 5),
            (None, [_get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 100, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 101, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 102, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 103, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 104, 0, 0),
                    ], 4),
            (None, [_get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 100, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 101, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 102, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 103, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 104, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 105, 0, 0),
                    ], 4),
            (None, [_get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 100, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 101, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 102, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 103, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 104, 0, 0),
                    _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 105, 0, 0),
                    ], 6),

            pytest.param(3, [_get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 100, 0, 0),
                             _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 101, 0, 0),
                             _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 102, 0, 0),
                             _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 103, 0, 0),
                             _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 104, 0, 0),
                             ], 4, marks=pytest.mark.xfail(strict=True)),
            pytest.param(None, [_get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 100, 0, 0),
                                _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 101, 0, 0),
                                _get_stub_post_stack_datum(np.random.rand(5, 5), 1, Direction.INLINE, 102, 0, 0),
                                _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 103, 0, 0),
                                _get_stub_post_stack_datum(np.random.rand(5, 5), 5, Direction.INLINE, 104, 0, 0),
                                ], 5, marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_class_balance_sampler_sample(self, num_examples_per_class, dataset_list, expected_dataset_list_length):
        # Arrange:
        class_balance_sampler_object = ClassBalanceSampler(num_examples_per_class)

        # Act:
        actual_dataset_list = class_balance_sampler_object.sample(dataset_list)

        # Assert:
        assert len(actual_dataset_list) == expected_dataset_list_length
