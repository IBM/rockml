""" Copyright 2019 IBM Research. All Rights Reserved.

    - Visualization functions for well seisfast.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import figure, pyplot as plt
from rockml.data.adapter.well import WellDatum


def _get_color(value: int):
    """  Get a random number between 0 and 1 based on a given seed. This is an auxiliary function to get colors in
        plot_df_logs().

    Args:
        value: (int) seed value.

    Returns:
        a random float.
    """

    state = np.random.RandomState(seed=value)

    return state.rand()


def plot_df_logs(df: pd.DataFrame, num_logs: List[str], cat_logs: List[str], depth_cut: Tuple[float, float] = None,
                 log_range: Tuple[float, float] = None, depth_range: Tuple[float, float] = None,
                 fig_size: Tuple[float, float] = (15, 15), ignore_nan: bool = False, show_grid: bool = True,
                 title: str = '') -> figure:
    """ Create a matplotlib figure with the *logs* side-by-side. It is possible to mark different lithological
        transitions using the *transition_label*.

    Args:
        df: pandas DataFrame with the desired logs, and with the index representing the depth.
        num_logs: list of *df* numerical column names to be plotted.
        cat_logs: list of *df* categorical column names to be plotted.
        depth_cut: tuple delimiting the region of interest to cut the df seisfast; if None, all depth range is included.
            Note that this is different from the *depth_range* parameter.
        log_range: (tuple) range of values in logs to be plotted (note it is the same for all logs).
        depth_range: (tuple) range of depth to be plotted.
        fig_size: tuple containing the size of the output figure.
        ignore_nan: (bool) if True, plot only (depth, value) pairs where value is not NaN; this is useful when the seisfast
            contains a lot of NaN intercalated with valid points and the plot becomes empty.
        show_grid: (bool) if True, shows grid in the plot.
        title: (str) title of the plot.

    Returns:
        a matplotlib figure.
    """

    plt.rcParams.update({'font.size': 24})

    fig, ax = plt.subplots(1, len(num_logs + cat_logs), sharey=True, figsize=fig_size)

    if type(ax) != np.ndarray:
        ax = [ax]

    # Determine depth range based on *depth_range*
    if depth_range:
        ax[0].set_ylim(depth_range)

    if depth_cut:
        # Cut df in the region of interest
        df = df[(df.index >= depth_cut[0]) & (df.index < depth_cut[1])]

    i = 0
    for log in num_logs:
        ax[i].set_title(log)

        if show_grid:
            ax[i].grid()  # Show grid

        # Plot log
        if ignore_nan:
            valid_idx = ~np.isnan(df[log].values)
            ax[i].plot(df[log][valid_idx], df.index[valid_idx], 'r-')
        else:
            ax[i].plot(df[log], df.index, 'r-')

        # Determine log ranges based on *log_range*
        if log_range:
            ax[i].set_xlim(log_range)

        i += 1

    for log in cat_logs:
        if show_grid:
            ax[i].grid()  # Show grid

        ax[i].set_title(log)

        labels = df[log].dropna(inplace=False)
        if len(labels) > 0:
            depth = labels.index
            labels = labels.values
            tops = np.unique(labels)
            cm = plt.get_cmap('gist_rainbow')

            mark_colors = {mark: cm(_get_color(int(mark))) for i, mark in enumerate(tops)}

            # Get positions where the array changes its value
            idx = np.concatenate(([0], np.where(labels[:-1] != labels[1:])[0] + 1))

            for init_limit, end_limit in zip(idx[:-1], idx[1:]):
                ax[i].axhspan(depth[init_limit], depth[end_limit - 1], xmin=0, xmax=1,
                              color=mark_colors[labels[init_limit]])

            ax[i].axhspan(depth[idx[-1]], depth[-1], xmin=0, xmax=1, color=mark_colors[labels[idx[-1]]])

        i += 1

    ax[0].invert_yaxis()
    fig.suptitle(title)

    return fig


def export_well_examples(datum_list: List[WellDatum], numerical_logs: List[str], categorical_logs: List[str],
                         out_path: str, num_wells: int = 3):
    """ Randomly select *num_wells* from the available in *df* to plot and save in multi-page pdf (one page per well).

    Args:
        datum_list: list of WellDatum containing well seisfast.
        numerical_logs: list of numerical log names to be plotted.
        categorical_logs: list of categorical log names to be plotted.
        out_path: (str) path to the output pdf file.
        num_wells: (int) number of wells to plot.
    """

    from matplotlib.backends.backend_pdf import PdfPages

    datum_list = np.random.choice(datum_list, size=num_wells, replace=False).tolist()

    with PdfPages(out_path) as pdf:
        for well in datum_list:
            df_to_plot = well.df.set_index(WellDatum.names.DEPTH_NAME.value, inplace=False, drop=True).sort_index()
            fig = plot_df_logs(df_to_plot, num_logs=numerical_logs, cat_logs=categorical_logs,
                               fig_size=(5 * len(numerical_logs + categorical_logs), 30),
                               title=f'{well.well_name}')
            pdf.savefig(fig)
