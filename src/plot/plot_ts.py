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
def plot_ts_dfs_in_2d_pca_space(ts_dfs       : List[pd.DataFrame],
                                ts_dist_func : str,
                                verbose      : bool = False) -> Tuple[plt.Figure,plt.Axes]:
    """Constructs a 2-D PCA plot of all time series samples provided, intended to 
    reveal if there are any latent groupings of time series samples in dataset.

    Args:
        ts_dfs (List[pd.DataFrame]): A list of :class:`pandas DataFrame(s)<pandas.DataFrame>`
            containing all of the time series samples to be plotted. 
        verbose (bool, optional): Boolean indicating whether or not to print out
            [DEBUG]-style messages to the console describing what is happening.
            Default is ``False``.
    """    
    fig, ax = None, None

    try:
        dist_func = None 
        if ts_dist_func == "Euc":
            dist_func = compute_euclidean
        
        elif ts_dist_func == "DTW":
            dist_func = compute_dtw

        if verbose:
            print(f"\n// {dbg()}  Attempting to compute distance['{ts_dist_func}'] matrix "
                  + f"using {len(ts_dfs)} time series samples!")

        dist_matrix = np.zeros((len(ts_dfs), len(ts_dfs)))

        # Compute a distance matrix comparing how alike all time series samples are.
        for row_idx in range(len(ts_dfs)):
            if verbose:
                print(f"\n// {dbg()}  Computing Row[{row_idx}] "
                      + f"of distance['{ts_dist_func}'] matrix!")
        
            for col_idx in range(row_idx + 1, len(ts_dfs)):        
                # Upper triangular of distance matrix.
                dist_matrix[row_idx][col_idx] = dist_func(first_obj=ts_dfs[row_idx],
                                                          second_obj=ts_dfs[col_idx])
                
                # Lower triangular of distance matrix.
                dist_matrix[col_idx][row_idx] = dist_matrix[row_idx][col_idx]

                if verbose:
                    print(f"// {dbg()}    DM[{row_idx}][{col_idx}] computed!")                



        if verbose:
            print(f"\n// {dbg()}  Attempting to project distance matrix to 2 PCA components!")

        # Transforms the data to 2 components for plotting on 2-D matplotlib Axes.
        pca_projections = PCA(n_components=2).fit_transform(dist_matrix)      

        if verbose:
            print(f"\n// {dbg()}  Attempting to plot 2 PCA components of distance matrix in 2-D!")

        # Plots the two PCA components of the distance matrix on 2-D matplotlib Axes.
        fig, ax = plt.subplots()

        ax.scatter(pca_projections[:, 0],
                   pca_projections[:, 1])

        ax.set_xlabel("1st PCA component")
        ax.set_ylabel("2nd PCA component")

        fig.suptitle(f"2-D Plot of Distance['{ts_dist_func}'] Matrix in 2 PCA Components")

    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't plot the time series DataFrame(s) in 2-D PCA space!\n")
        traceback.print_exc()

    return (fig, ax)


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
