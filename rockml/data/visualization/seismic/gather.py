""" Copyright 2020 IBM/GPN joint development. All Rights Reserved. """

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_boxplot(scores: np.ndarray, score_name: str, output_path: str):
    """ Plot boxplot of the scores.

    Args:
        scores: np.ndarray containing the scores in the format [[inline, crossline, score], ...].
        score_name: (str) name of the score to put as title of the graph.
        output_path: (str) path where to save the figure.
    """

    plt.figure(dpi=300)
    plt.title(f"{score_name} score boxplot")
    plt.boxplot(scores[:, 2][~np.isnan(scores[:, 2])], sym='')
    plt.ylabel("scores")
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight', format='png')


def plot_scatter_hist(scores: np.ndarray, output_path: str, plot_title: str = ''):
    """ Plots scatter histogram.

    Args:
        scores: np.ndarray containing the scores in the format [[inline, crossline, score], ...].
        plot_title: (str) title of the graph.
        output_path: (str) file path where to save the plot.
    """
    xs = scores[:, 0]
    ys = scores[:, 1]
    score_vec = scores[:, 2]

    plt.figure(figsize=(16, 8), dpi=150)

    # definitions for the axes
    left, width = 0.15, 0.7
    bottom, height = 0.1, 0.75
    rect_scatter = [left, bottom, width, height]
    rect_histy = [left + width, bottom, 0.05, height]

    plt.title(plot_title)

    plt.axis('off')

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the scatter plot:

    if np.nanmax(score_vec) > 100:
        newcm = LinearSegmentedColormap.from_list('customMap',
                                                  [(0, 'red'), (70 / np.nanmax(score_vec), 'yellow'),
                                                   (100 / np.nanmax(score_vec), 'green'),
                                                   (1, 'green')])
    else:
        newcm = LinearSegmentedColormap.from_list('customMap',
                                                  [(0, 'red'), (70 / np.nanmax(score_vec), 'yellow'),
                                                   (1, 'green')])
    max = np.nanmax(score_vec)

    if max < 100:
        max = 100

    sc = ax_scatter.scatter(xs, ys, c=score_vec, vmin=0, vmax=max, s=30, cmap=newcm)

    plt.colorbar(sc, ax=ax_scatter)
    ax_scatter.set_ylabel('crosslines')
    ax_scatter.set_xlabel('inlines')

    ax_histy.hist(score_vec, range=[0, np.nanmax(score_vec)], bins=500, orientation='horizontal')

    plt.savefig(output_path, bbox_inches='tight', format='png')


def plot_difference(diff_values: np.ndarray, output_path: str, plot_title: str = "velocity 1 x velocity 2", vmax=None):
    """ Plots *diff_values* and saves the figure in *output_dir*.

    Args:
        diff_values: np.ndarray of difference values in the format [[inline, crossline, difference_score], ...]
        output_path: (str) path where to save the plot.
        plot_title: (str) title of the plot.
        vmax: (int) maximum value in the code of colors; if None, the maximum is set as the maximum in the seisfast.
    """

    newcm = LinearSegmentedColormap.from_list('customMap', [(0, 'green'), (0.5, 'yellow'), (1, 'red')])

    plt.figure(figsize=(16, 8), dpi=150)
    plt.title(plot_title)
    plt.scatter(diff_values[:, 0], diff_values[:, 1], c=diff_values[:, 2], cmap=newcm, linewidth=0.5, vmin=0, vmax=vmax)
    plt.colorbar()
    plt.ylabel("crosslines")
    plt.xlabel("inlines")

    plt.savefig(output_path, bbox_inches='tight')

    return
