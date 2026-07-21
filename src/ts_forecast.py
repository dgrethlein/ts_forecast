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


from pathlib import Path

import traceback
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sktime.datasets import load_tsf_to_dataframe

from .utils.args import parse_run_ts_forecast_args

from .utils.misc import dbg, err
from .utils.misc import is_nonneg_finite_int

from .utils.plot_ts import plot_ts_df_on_ax


#==============================================================================
#       LOAD TIME SERIES DATASET FROM (.tsf) FILE FUNCTION(s)
#==============================================================================
def load_dataset_df_into_series_dfs(verbose : bool = False) -> Tuple[List[pd.DataFrame],List[str]]:
    """Summary

    Returns:
        Tuple[List[pd.DataFrame], List[str]]: Description

    Args:
        verbose (bool, optional): Description
    """
    series_dfs = []
    series_names = []

    try:
        # Load the dataset (.tsf) file into a MultiIndex DataFrame
        data_path = Path("data/electricity_hourly_dataset.tsf")
        data_df, metadata = load_tsf_to_dataframe(full_file_path_and_name=data_path,
                                                  replace_missing_vals_with="NaN",
                                                  value_column_name="kwh_electricity_consumed")
        if verbose:
            print(f"\n// {dbg()}  Dataset metadata := {metadata}\n")

        # Iterates over the 321 time series samples in the dataset.
        for sample_idx in range(1,322):
            sample_name = f"T{sample_idx}"

            # Splits the time series samples by name into individual pandas DataFrame(s).
            sample_df = data_df.loc[sample_name].reset_index(drop=True)

            if verbose:
                print(f"// {dbg()}  Sample['{sample_name}'] shape := {sample_df.shape}")

            series_dfs.append(sample_df)
            series_names.append(sample_name)

        if verbose:
            print(f"\n// {dbg()}  Loaded (N={len(series_dfs)}) named "
                  + "time series samples from file!")

    except (AttributeError, FileNotFoundError, IOError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't split dataset pandas DataFrame into individual "
              + "time series DataFrame(s).\n")
        traceback.print_exc()

    return (series_dfs, series_names)


#==============================================================================
#       SPLIT TIME SERIES DATASET FUNCTION(s)
#==============================================================================
def get_dataset_train_test_cv_splits(cv_random_seed : int,
                                     data_dfs       : List[pd.DataFrame],
                                     num_cv_folds      : int,
                                     verbose        : bool = False) -> Dict:
    """Summary

    Args:
        cv_random_seed (int): Description
        data_dfs (List[pd.DataFrame]): Description
        num_folds (int): Description
        verbose (bool, optional): Description

    Returns:
        Dict: Description
    """
    splits_dict = {}

    try:
        # If provided arguments were valid, generate K splits of a dataset.
        if is_nonneg_finite_int(num_folds) and is_nonneg_finite_int(cv_random_seed):
            kf = KFold(n_splits=num_folds,
                       shuffle=True,
                       random_state=int(cv_random_seed))

            if verbose:
                print(f"\n// {dbg()}  Splitting the dataset of (N={len(data_dfs)}) "
                      + f"time series samples into K={num_folds} splits for "
                      + "K-fold cross-validation!")
                print(f"// {dbg()}--------------------------------------------------------------")

            for split_idx, (train_idx, test_idx) in enumerate(kf.split(data_dfs)):
                splits_dict[f"fold_{split_idx}"] = {"train_idx" : list(train_idx),
                                                    "test_idx"  : list(test_idx)}

                if verbose:
                    print(f"\n// {dbg()}  K={num_folds} cross-validation fold[{split_idx}]")
                    print(f"// {dbg()}    # train_idx := {len(train_idx)}")
                    print(f"// {dbg()}    # test_idx  := {len(test_idx)}")


    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't get the train and test splits for dataset!\n")
        traceback.print_exc()

    return splits_dict



def split_ts_df_into_train_and_test(data_df : pd.DataFrame,
                                    holdout : float,
                                    verbose : bool = False) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """Summary

    Args:
        data_df (pd.DataFrame): Description
        holdout (float): Description
        verbose (bool, optional): Description
    """
    train_data_df = None
    test_data_df = None

    try:
        train_data_df = data_df.iloc[:int(len(data_df) * (1.0 - holdout))].reset_index(drop=True)
        test_data_df = data_df.iloc[int(len(data_df) * (1.0 - holdout)):].reset_index(drop=True)

        if verbose:
            print(f"\n// {dbg()}  Split time series sample of length "
                  + f"{len(data_df)} frames into train and test data!")
            print(f"// {dbg()}    # training frames := {len(train_data_df)}")
            print(f"// {dbg()}    # testing frames  := {len(test_data_df)}")

    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't split time series DataFrame into train and test data!\n")
        traceback.print_exc()

    return (train_data_df, test_data_df)


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    dfs, names = load_dataset_df_into_series_dfs(verbose=True)

    # for series_idx, series_df in enumerate(dfs[:3]):
    #     plot_ts_df_on_ax(data_df=series_df,
    #                      verbose=True)

    pargs = parse_run_ts_forecast_args()

    split_ts_df_into_train_and_test(data_df=dfs[0],
                                    holdout=pargs["holdout_percentage"],
                                    verbose=pargs["verbose"])

    # get_dataset_train_test_cv_splits(cv_random_seed=41,
    #                                  data_dfs=series_dfs,
    #                                  num_cv_folds=5,
    #                                  verbose=True)

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
