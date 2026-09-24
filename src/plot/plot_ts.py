#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Time Series Plotting<src.plot.plot_ts>` module.

Module Description
==================

Module for plotting time series data onto :class:`matplotlib.Axes`.

.. moduleauthor:: David Grethlein

Module Contents
===============

"""


import traceback

from typing import List, Tuple

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..ml.ts_distance.dtw import compute_sc_dtw
from ..ml.ts_distance.euclidean import compute_euclidean

from ..utils.misc import dbg, err


#==============================================================================
#       TIME SERIES PLOTTING FUNCTION(s)
#==============================================================================
def plot_ts_dfs(ts_dfs       : List[pd.DataFrame],
                verbose      : bool = False) -> Tuple[plt.Figure,plt.Axes]:
    """Plots all time series samples provided on a set of
    :class:`matplotlib Axes<matplotlib.pyplot.Axes>` and returns the result.

    Args:
        ts_dfs (List[pd.DataFrame]): A list of :class:`pandas DataFrame(s)<pandas.DataFrame>`
            containing all of the time series samples to be plotted.
        verbose (bool, optional): Boolean indicating whether or not to print out
            [DEBUG]-style messages to the console describing what is happening.
            Defaults to False.
    """
    fig, ax = None, None

    try:
        if verbose:
            print(f"\n// {dbg()}  Attempting to plot time series DataFrame(s)!")

        fig, ax = plt.subplots()

        for ts_df in ts_dfs:
            ax.plot(ts_df,
                    alpha=.1)

    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't plot the time series DataFrame(s)!\n")
        traceback.print_exc()

    return (fig, ax)


# pylint: disable=too-many-locals
def plot_non_outlier_ts_dfs(ts_dfs           : List[pd.DataFrame],
                            plot_method      : str = "PCA",
                            random_seed      : int = 0,
                            sc_dtw_band_size : int = 24,
                            ts_dist_func     : str = "Euc",
                            verbose          : bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """_summary_

    Args:
        ts_dfs (List[pd.DataFrame]): _description_
        plot_method (str, optional): _description_. Defaults to "PCA".
        random_seed (int, optional): _description_. Defaults to 0.
        ts_dist_func (str, optional): _description_. Defaults to "Euc".
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[plt.Figure, plt.Axes]: _description_
    """
    fig, ax = None, None

    try:
        # Concatenates all of the time series data into a single DataFrame.
        concat_ts_dfs = pd.concat(ts_dfs, axis=0).reset_index(drop=True)

        # Computes the mean and standard deviation of all the data.
        concat_mean = concat_ts_dfs["kwh_electricity_consumed"].mean()
        concat_std = concat_ts_dfs["kwh_electricity_consumed"].std()

        # Identify all outlier time series samples
        # (series-specific mean more than 3 standard deviations from the concatenated mean).
        non_outliers = []
        for ts_idx, ts_df in enumerate(ts_dfs):
            ts_df_mean = ts_df["kwh_electricity_consumed"].mean()

            if (ts_df_mean < concat_mean - 3 * concat_std
                or ts_df_mean > concat_mean + 3 * concat_std):

                if verbose:
                    print(f"// {dbg()}  Time Series DataFrame[{ts_idx}] is an outlier, removed!")

            else:
                non_outliers.append(ts_df)

        if verbose:
            print(f"\n// {dbg()}  {len(non_outliers)} non-outlier time series samples preserved!")
            print(f"\n// {dbg()}  Attempting to compute {len(non_outliers)}x{len(non_outliers)} "
                  + f"Euclidean distance matrix for plotting['{plot_method}']' "
                  + f"of {len(non_outliers)} time series samples!")

        # Computes a distance matrix relating how alike all time series samples are to one another.
        # Uses raw, untransformed time series samples for this.
        dfunc = None
        if ts_dist_func == "Euc":
            dfunc = compute_euclidean
        elif ts_dist_func == "SC_DTW":
            dfunc = compute_sc_dtw

        dmatrix = np.zeros((len(non_outliers), len(non_outliers)))
        for row_idx, _ in enumerate(non_outliers):
            for col_idx in range(row_idx + 1, len(non_outliers)):
                dmatrix[row_idx][col_idx] = dfunc(first_obj=non_outliers[row_idx],
                                                  second_obj=non_outliers[col_idx])
                dmatrix[col_idx][row_idx] = dmatrix[row_idx][col_idx]

        projector = None
        projections = None

        # Project the distance matrix of non-outliers time series samples
        # to 2-D PCA space from Euclidean distances.
        if plot_method == "PCA":
            projector = PCA(n_components=2)
            projector.fit(dmatrix)
            projections = projector.transform(dmatrix)

        # Project the distance matrix of non-outliers
        elif plot_method == "TSNE":
            projector = TSNE(n_components=2, random_state=random_seed)
            projector.fit(dmatrix)
            projections = projector.transform(dmatrix)

        fig, ax = plt.subplots()
        ax.scatter(projections[:,0],
                   projections[:,1])

    except (AttributeError, IndexError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't plot the non-outlier time series DataFrame(s)!\n")
        traceback.print_exc()

    return fig, ax


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
