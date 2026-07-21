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


import argparse

import numpy as np
import pandas as pd

from pathlib import Path

import traceback
from typing import List, Tuple

import matplotlib.pyplot as plt

from sktime.datasets import load_tsf_to_dataframe

from .utils.misc import dbg, err



def split_dataset_df_into_series_dfs(verbose : bool = False) -> Tuple[List[pd.DataFrame],List[str]]:
    """Summary

    Returns:
        Tuple[List[pd.DataFrame], List[str]]: Description

    Args:
        verbose (bool, optional): Description
    """
    series_dfs = []
    series_names = []

    try:
        data_path = Path("data/electricity_hourly_dataset.tsf")

        # Load the file into a MultiIndex DataFrame
        data_df, metadata = load_tsf_to_dataframe(full_file_path_and_name=data_path,
                                                  replace_missing_vals_with="NaN",
                                                  value_column_name="kwh_electricity_consumed")
        if verbose:
            print(f"\n// {dbg()}  Dataset metadata := {metadata}\n")

        # Iterates over the 322 time series samples in the dataset.
        for sample_idx in range(1,322):
            sample_name = f"T{sample_idx}"

            # Splits the time series sampels by name into individual pandas DataFrame(s).
            sample_df = data_df.loc[sample_name].reset_index(drop=True)

            if verbose:
                print(f"Sample['{sample_name}'] shape := {sample_df.shape}")

            series_dfs.append(sample_df)
            series_names.append(sample_name)


    except (AttributeError, FileNotFoundError, IOError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't split dataset pandas DataFrame into individual "
              + "time series DataFrame(s).\n")
        traceback.print_exc()

    return (series_dfs, series_names)


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    series_dfs, series_names = split_dataset_df_into_series_dfs(verbose=True)

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
