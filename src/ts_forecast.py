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

import matplotlib.pyplot as plt

from sktime.datasets import load_tsf_to_dataframe

from .utils.misc import dbg, err


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")


    # Load the file into a MultiIndex DataFrame
    df, metadata = load_tsf_to_dataframe("data/wind_farms_minutely_dataset_with_missing_values.tsf")
    print(df.head())

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
