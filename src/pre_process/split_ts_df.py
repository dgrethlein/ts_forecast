#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Split Time Series DataFrame Pre-Processing<src.pre_process.split_ts_df>` module.


Module Description
==================

Module for splitting :class:`pandas DataFrame(s)<pandas.DataFrame>` containing time
series data recorded by many entities into individual time series samples, as well
as splitting methods for training and testing time series forecasting models.

.. moduleauthor:: David Grethlein

Module Contents
===============

"""

from pathlib import Path

import traceback

from typing import Dict, List, Tuple

import pandas as pd

from sklearn.model_selection import KFold
from sktime.datasets import load_tsf_to_dataframe

from ..utils.misc import dbg, err
from ..utils.misc import is_nonneg_finite_int


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
                                     num_cv_folds   : int,
                                     verbose        : bool = False) -> Dict:
    """Summary

    Args:
        cv_random_seed (int): Description
        data_dfs (List[pd.DataFrame]): Description
        num_cv_folds (int): Description
        verbose (bool, optional): Description

    Returns:
        Dict: Description

    Deleted Parameters:
        num_folds (int): Description
    """
    splits_dict = {}

    try:
        # If provided arguments were valid, generate K splits of a dataset.
        if is_nonneg_finite_int(num_cv_folds) and is_nonneg_finite_int(cv_random_seed):
            kf = KFold(n_splits=num_cv_folds,
                       shuffle=True,
                       random_state=int(cv_random_seed))

            if verbose:
                print(f"\n// {dbg()}  Splitting the dataset of (N={len(data_dfs)}) "
                      + f"time series samples into K={num_cv_folds} splits for "
                      + "K-fold cross-validation!")
                print(f"// {dbg()}--------------------------------------------------------------")

            for split_idx, (train_idx, test_idx) in enumerate(kf.split(data_dfs)):
                splits_dict[f"fold_{split_idx}"] = {"train_idx" : list(train_idx),
                                                    "test_idx"  : list(test_idx)}

                if verbose:
                    print(f"\n// {dbg()}  K={num_cv_folds} cross-validation fold[{split_idx}]")
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

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Description
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

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
