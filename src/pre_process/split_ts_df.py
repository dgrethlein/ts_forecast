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

Attributes:
    DATA_TSF_FILE_PATH (str): The file name path of the time series dataset.

        Stored on the local file system at: ``data/electricity_hourly_dataset.tsf``.

        Originally acquired from: https://zenodo.org/records/4656140 .

    NUM_INDIVIDUAL_TS_DFS (int): The number (``321``) of individual time series samples
        compactly stored in a single (.tsf) file, to be broken into individual series.
    VALUE_COLUMN_NAME (str): The column name (``kwh_electricity_consumed``) under which
        the time series data for this dataset is stored under.

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
#       TIME SERIES DATASET CONSTANT(s)
#==============================================================================
DATA_TSF_FILE_PATH = "data/electricity_hourly_dataset.tsf"
NUM_INDIVIDUAL_TS_DFS = 321
TS_VALUE_COLUMN_NAME = "kwh_electricity_consumed"


#==============================================================================
#       LOAD TIME SERIES DATASET FROM (.tsf) FILE FUNCTION(s)
#==============================================================================
def load_dataset_df_into_ts_dfs(verbose : bool = False) -> Tuple[List[pd.DataFrame],List[str]]:
    """Loads the entire time series dataset from a (.tsf) file, stored in
    :const:`DATA_TSF_FILE_PATH` on the local file system, and breaks the resulting
    :class:`pandas.DataFrame` into :const:`NUM_INDIVIDUAL_TS_DFS` (321) individual
    time series :class:`pandas DataFrame(s)<pandas.DataFrame>`.

    Args:
        verbose (bool, optional): Boolean indicating whether or not to print out
            [DEBUG]-style messages to the console describing what is happening.
            Default is ``False``.

    Returns:
        Tuple[List[pd.DataFrame], List[str]]: A tuple containing a list of 321
            individual time series samples, each stored in their own :class:`pandas.DataFrame>`,
            along with a list of 321 strings, corresponding to the names of each series.
    """
    series_dfs = []
    series_names = []

    try:
        # Load the dataset (.tsf) file into a MultiIndex DataFrame
        data_path = Path(DATA_TSF_FILE_PATH)
        data_df, metadata = load_tsf_to_dataframe(full_file_path_and_name=data_path,
                                                  replace_missing_vals_with="NaN",
                                                  value_column_name=TS_VALUE_COLUMN_NAME)
        if verbose:
            print(f"\n// {dbg()}  Dataset metadata := {metadata}\n")

        # Iterates over the 321 time series samples in the dataset.
        for sample_idx in range(NUM_INDIVIDUAL_TS_DFS):

            # Sample names are indexed starting at 1.
            sample_name = f"T{sample_idx + 1}"

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
def get_dataset_train_test_cv_splits_idx_dict(data_dfs       : List[pd.DataFrame],
                                              cv_random_seed : int = 0,
                                              num_cv_folds   : int = 5,
                                              verbose        : bool = False) -> Dict:
    """Gets a dictionary containing the indices of time series samples split into
    train and test sets for k-fold cross-validated experiments.

    Args:
        data_dfs (List[pd.DataFrame]): A list of individual time series samples stored
            in :class:`pandas DataFrame(s)<pandas.DataFrame>`.
        cv_random_seed (int, optional): The proposed non-negative finite integer
            that will be used to seed all random number generators used.
            Default value is ``0``.
        num_cv_folds (int, optional): The proposed non-negative finite integer greater
            than 1 that describes the number of disjoint folds to split the time series
            dataset into for the purpose of conducting cross-validated experiments.
            Default value is ``5``.
        verbose (bool, optional): Boolean indicating whether or not to print out
            [DEBUG]-style messages to the console describing what is happening.
            Default is ``False``.

    Returns:
        Dict: A dictionary containing the indices of time series samples split into
            train and test sets for k-fold cross-validated experiments.
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
    """Splits a single time series sample stored in a :class:`pandas.DataFrame>` into
    two :class:`pandas DataFrame(s)<pandas.DataFrame>`. The first DataFrame being used
    to train a predictive model, and the second used to test the model. The fraction of
    data withheld for testing is controlled using the :arg:`holdout` argument and is
    expected to be a valid percentage in the range (0.0, 1.0).

    Args:
        data_df (pd.DataFrame): A single time series sample stored in a :class:`pandas.DataFrame>`.
        holdout (float): The fraction of time series data to be withheld in the training set.
        verbose (bool, optional): Boolean indicating whether or not to print out
            [DEBUG]-style messages to the console describing what is happening.
            Default is ``False``.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two time series partitioned from
            the supplied time series sample, each stored in a :class:`pandas.DataFrame`.
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
