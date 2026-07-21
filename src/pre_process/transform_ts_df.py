#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Transform Time Series DataFrame Pre-Processing<src.pre_process.transform_ts_df>` module.


Module Description
==================

Module for transforming :class:`pandas DataFrame(s)<pandas.DataFrame>` containing time
series data recorded by individual time series samples, for stationarity, skew, and other
statistical tests used in pre-processing time series data to be fed into a time series 
forecasting model.

.. moduleauthor:: David Grethlein

Module Contents
===============

"""


import traceback

from typing import Tuple

import pandas as pd

from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer

from ..utils.misc import dbg, err
from ..utils.misc import is_finite_float


#==============================================================================
#       TRANSFORM TIME SERIES PANDAS DATAFRAME FUNCTION(s)
#==============================================================================
def box_cox_transform(data_df        : pd.DataFrame,
                      box_cox_lambda : float = None,
                      verbose        : bool = False) -> Tuple[pd.DataFrame,float]:
    """Summary

    Args:
        data_df (pd.DataFrame): Description
        box_cox_lambda (float, optional): Description
        verbose (bool, optional): Description

    Returns:
        Tuple[pd.DataFrame, float]: Description
    """

    transformed_df = None
    best_lambda = None

    try:
        transformed = None

        # Identifies the best lambda value for the Box-Cox transformation
        # to be applied to the supplied time series DataFrame for normalization.
        if box_cox_lambda is None:
            transformed, best_lambda = boxcox(data_df["kwh_electricity_consumed"].values)

        # Applies a specific Box-Cox transformation (using provided lambda value)
        # to supplied time series DataFrame for normalization.
        elif is_finite_float(box_cox_lambda):
            transformed, _ = boxcox(data_df["kwh_electricity_consumed"].values,
                                    box_cox_lambda)

        # Re-packages the transformed data into pandas DataFrame.
        transformed_df = pd.DataFrame(transformed, columns=data_df.columns.values)

    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't perform the Box-Cox transform on DataFrame!\n")
        traceback.print_exc()

    return (transformed_df, best_lambda)



def yeo_johnson_transform(data_df : pd.DataFrame,
                          verbose : bool = False) -> Tuple[pd.DataFrame, PowerTransformer]:
    """Summary

    Args:
        data_df (pd.DataFrame): Description
        verbose (bool, optional): Description

    Returns:
        Tuple[pd.DataFrame, PowerTransformer]: Description
    """
    transformed_df = None
    yj_transformer = None

    try:
        yj_transformer = PowerTransformer(method="yeo-johnson")
        yj_transformer.set_output(transform="pandas")
        yj_transformer.fit(data_df)

        transformed_df = yj_transformer.transform(data_df)

    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't perform the Yeo-Johnson transform on DataFrame!\n")
        traceback.print_exc()

    return (transformed_df, yj_transformer)


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
