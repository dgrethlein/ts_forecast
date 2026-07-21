#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Time Series Forecasting<src.ts_forecast>` module.


Module Description
==================

Module for forecasting the value of time series data.

.. moduleauthor:: David Grethlein

Module Contents
===============

"""


import traceback


import numpy as np
import pandas as pd

from .plot.plot_ts import plot_train_and_test_dfs_on_ax
from .plot.plot_ts import plot_yj_transformed_train_and_test_dfs_on_ax

from .pre_process.split_ts_df import load_dataset_df_into_series_dfs
from .pre_process.split_ts_df import split_ts_df_into_train_and_test

from .pre_process.transform_ts_df import box_cox_transform
from .pre_process.transform_ts_df import yeo_johnson_transform

from .utils.args import parse_run_ts_forecast_args

from .utils.misc import dbg, err


def run_ts_forecast():

    try:
        # Parse command line arguments.
        pargs = parse_run_ts_forecast_args()

        # Loads time series data from (.tsf) file, splits into individual time series samples.
        dfs, names = load_dataset_df_into_series_dfs(verbose=True)

        # Iterates over each time series sample individually.
        for series_idx, series_df in enumerate(dfs[:3]):

            # Splits the time series sample into train and test sets.
            train_df, test_df = split_ts_df_into_train_and_test(data_df=series_df,
                                                                holdout=pargs["holdout_percentage"],
                                                                verbose=pargs["verbose"])

            # Plots the split time series data.
            plot_fig, plot_ax = plot_train_and_test_dfs_on_ax(data_df=series_df,
                                                              holdout=pargs["holdout_percentage"],
                                                              verbose=pargs["verbose"])

            # Plots the Yeo-Johnson transformed time series data.
            yj_plot_fig, yj_plot_ax = plot_yj_transformed_train_and_test_dfs_on_ax(data_df=series_df,
                                                                                   holdout=pargs["holdout_percentage"],
                                                                                   verbose=pargs["verbose"])

            # transformed_df, yj_transformer = yeo_johnson_transform(data_df=train_df,
            #                                                        verbose=pargs["verbose"])



    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't run time series forecasting!\n")
        traceback.print_exc()




#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    run_ts_forecast()

    # get_dataset_train_test_cv_splits(cv_random_seed=41,
    #                                  data_dfs=series_dfs,
    #                                  num_cv_folds=5,
    #                                  verbose=True)

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
