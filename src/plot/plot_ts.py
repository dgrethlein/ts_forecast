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

from ..ml.ts_distance.dtw import compute_dtw
from ..ml.ts_distance.euclidean import compute_euclidean

from ..pre_process.split_ts_df import split_ts_df_into_train_and_test

from ..utils.misc import dbg, err
from ..utils.misc import is_non_empty_str


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
            Default is ``False``.
    """
    fig, ax = None, None

    try:
        fig, ax = plt.subplots()

        for ts_df in ts_dfs:
            ax.plot(ts_df,
                    alpha=.1)

    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't plot the time series DataFrame(s)!\n")
        traceback.print_exc()

    return (fig, ax)


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
