""" Copyright 2019 IBM Research. All Rights Reserved.

    - Test functions for seisfast.visualization.seismic.gather.
"""

from pathlib import Path

import numpy as np
import pytest
from rockml.data.visualization.seismic.gather import plot_scatter_hist, plot_difference


@pytest.mark.parametrize(
    "scores, output_path, plot_title",
    [(np.array([[100, 800, 97.0], [110, 800, 98.0]]), "segy_test_data/scatter.png", "test")]
)
def test_plot_scatter_hist(scores, output_path, plot_title):
    plot_scatter_hist(scores, output_path, plot_title)
    f_path = Path(output_path)
    assert f_path.is_file()


@pytest.mark.parametrize(
    "diff_values, output_dir",

    [(np.random.rand(2, 3), "segy_test_data/difference.png")
     ]
)
def test_plot_difference(diff_values, output_dir):
    plot_difference(diff_values, output_dir)
    f_path = Path(output_dir)
    assert f_path.is_file()
