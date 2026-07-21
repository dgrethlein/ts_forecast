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

from typing import Tuple

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from ..pre_process.split_ts_df import split_ts_df_into_train_and_test

from ..pre_process.transform_ts_df import yeo_johnson_transform
from ..pre_process.transform_ts_df import difference_transform

from ..utils.misc import dbg, err
from ..utils.misc import is_non_empty_str


#==============================================================================
#       TIME SERIES PLOTTING FUNCTION(s)
#==============================================================================
def setup_ts_df_plot(ax_x_label : str = None,
                     ax_y_label : str = None,
                     plot_title : str = None,
                     verbose    : bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """Summary
    """
    plot_fig = None
    plot_ax = None

    try:
        if verbose:
            print(f"\n// {dbg()}  Attempting to set up matplotlib plot "
                  + "for time series DataFrame!\n")

        # Initializes matplotlib Figure and Axes objects.
        plot_fig, plot_ax = plt.subplots(figsize=(10,5))

        # Ensures a valid x-axis label is used.
        if ax_x_label is None or not is_non_empty_str(ax_x_label):
            ax_x_label = "Time-steps (Hourly)"

        # Ensures a valid y-axis label is used.
        if ax_y_label is None or not is_non_empty_str(ax_y_label):
            ax_y_label = "Electricity consumption (kwh)"

        # Ensures a valid plot title is used.
        if plot_title is None or not is_non_empty_str(plot_title):
            plot_title = "Hourly electricity consumption (kwh) from 2011 to 2014."

        # Apply labeling and title.
        plot_ax.set_xlabel(ax_x_label)
        plot_ax.set_ylabel(ax_y_label)
        plot_fig.suptitle(plot_title)

        if verbose:
            print(f"\n// {dbg()}  Successfully set up matplotlib plot!\n")

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't set up a matplotlib plot for time series samples!\n")
        traceback.print_exc()

    return (plot_fig, plot_ax)


def plot_iqr_hlines_on_ax(ax       : plt.Axes,
                          data_df  : pd.DataFrame,
                          verbose  : bool = False):
    """Summary

    Args:
        ax (plt.Axes): Description
        data_df (pd.DataFrame): Description
        verbose (bool, optional): Description
    """
    try:
        data_median = data_df["kwh_electricity_consumed"].median()
        data_q1 = data_df["kwh_electricity_consumed"].quantile(0.25)
        data_q3 = data_df["kwh_electricity_consumed"].quantile(0.75)

        data_iqr = data_q3 - data_q1

        data_lower_bound = data_q1 - (1.5 * data_iqr)
        data_upper_bound = data_q3 + (1.5 * data_iqr)

        ax.axhline(y=data_median,
                   color="r",
                   linestyle="--",
                   linewidth=2,
                   label="Training Median")

        ax.axhline(y=data_q1,
                   color="purple",
                   linestyle="-.",
                   linewidth=2,
                   label="Training Q1")

        ax.axhline(y=data_q3,
                   color="pink",
                   linestyle=":",
                   linewidth=2,
                   label="Training Q3")

        ax.axhline(y=data_lower_bound,
                   color="indigo",
                   linestyle=":",
                   linewidth=2,
                   label="Lower Bound (Q1 - 1.5 * IQR)")

        ax.axhline(y=data_upper_bound,
                   color="magenta",
                   linestyle="-.",
                   linewidth=2,
                   label="Upper Bound (Q3 + 1.5 * IQR)")

        if verbose:
            print(f"// {dbg()}  Plotted horizontal IQR lines on matplotlib Axes!")

    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't plot IQR horizontal lines on matplotlib Axes!\n")
        traceback.print_exc()





def plot_train_and_test_dfs(data_df : pd.DataFrame,
                            holdout : float,
                            verbose : bool = False) -> Tuple[plt.Figure,plt.Axes]:
    """Summary

    Args:
        data_df (pd.DataFrame): Description
        holdout (float): Description
        verbose (bool, optional): Description
    """
    fig = None
    ax = None

    try:
        # Sets up a plotting Figure and Axes.
        fig, ax = setup_ts_df_plot(verbose=verbose)

        # Splits the provided time series DataFrame into train and test sets.
        train_df, test_df = split_ts_df_into_train_and_test(data_df=data_df,
                                                            holdout=holdout,
                                                            verbose=verbose)

        # Plots the training data.
        ax.plot(np.array(list(range(len(train_df)))),
                train_df["kwh_electricity_consumed"].values,
                alpha=0.5,
                label="Training data",
                c="k")

        # Plots the testing data (to be forecast).
        ax.plot(np.array(list(range(len(train_df), len(data_df)))),
                test_df["kwh_electricity_consumed"].values,
                alpha=0.5,
                label="Testing data",
                c="g")

        # Plots the IQR lines on the matplotlib Axes.
        plot_iqr_hlines_on_ax(ax=ax,
                              data_df=train_df,
                              verbose=verbose)

        ax.legend()
        fig.tight_layout()

    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't plot time series train and test "
              + "DataFrame(s) on matplotlib Axes!\n")
        traceback.print_exc()

    return (fig, ax)


# pylint: disable=too-many-locals
def plot_yj_transformed_train_and_test_dfs(data_df : pd.DataFrame,
                                           holdout : float,
                                           verbose : bool = False) -> Tuple[plt.Figure,plt.Axes]:
    """Summary

    Args:
        data_df (pd.DataFrame): Description
        holdout (float): Description
        verbose (bool, optional): Description
    """
    fig = None
    ax = None

    try:
        # Splits the time series sample into train and test sets.
        train_df, test_df = split_ts_df_into_train_and_test(data_df=data_df,
                                                            holdout=holdout,
                                                            verbose=verbose)

        # Computes the appropriate Yeo-Johnson transformation for the training data.
        transf_train_df, yj_transformer = yeo_johnson_transform(data_df=train_df,
                                                                verbose=verbose)

        yj_lambda = yj_transformer.lambdas_[0]

        # Applies the same transformation to the testing data.
        transf_test_df = yj_transformer.transform(test_df)

        # Sets up a plotting Figure and Axes.
        fig, ax = setup_ts_df_plot(ax_y_label=("Electricity consumption (kwh) "
                                               + "[Y-J Transformed "
                                               + f"(lambda = {yj_lambda:.3f})]"),
                                   plot_title=("Hourly electricity consumption (kwh) "
                                               + "from 2011 to 2014 "
                                               + "[Yeo-Johnson Transformed "
                                               + f"(lambda = {yj_lambda:.3f})]."),
                                   verbose=verbose)

        # Plots the training data.
        ax.plot(np.array(list(range(len(train_df)))),
                transf_train_df["kwh_electricity_consumed"].values,
                alpha=0.5,
                label="Training data",
                c="k")

        # Plots the testing data.
        ax.plot(np.array(list(range(len(train_df), len(data_df)))),
                transf_test_df["kwh_electricity_consumed"].values,
                alpha=0.5,
                label="Testing data",
                c="g")

        # Plots the IQR lines on the matplotlib Axes.
        plot_iqr_hlines_on_ax(ax=ax,
                              data_df=transf_train_df,
                              verbose=verbose)

        ax.legend()
        fig.tight_layout()


        fig2, ax2 = plt.subplots(nrows=1,
                                 ncols=2)

        train_df["kwh_electricity_consumed"].plot.hist(ax=ax2[0], bins=10)
        transf_train_df["kwh_electricity_consumed"].plot.hist(ax=ax2[1], bins=10)

        ax2[0].set_title("Histogram of training data")
        ax2[1].set_title(f"Histogram of [Y-J Transformed training data (lambda = {yj_lambda:.3f})]")

    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't plot the Yeo-Johnson transformed train and test "
              + "DataFrame(s) on matplotlib Axes!\n")
        traceback.print_exc()

    return (fig, ax, fig2, ax2)


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
