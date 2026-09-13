#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Stationarity Time Series DataFrame Pre-Processing<src.pre_process.stationarity>` module.


Module Description
==================

Module for determining whether time series data stored in 
:class:`pandas DataFrame(s)<pandas.DataFrame>` are stationary or not.

.. moduleauthor:: David Grethlein

Module Contents
===============

"""


import traceback

from typing import Dict, Tuple

from statsmodels.tsa.stattools import adfuller

from ..utils.misc import dbg, err


#==============================================================================
#       STATIONARITY TESTING FUNCTION(s)
#==============================================================================
def perform_adf_test(series_df : pd.DataFrame,
                     threshold : float = 0.05,
                     verbose   : bool = False,
                     **kwargs  : Dict) -> Tuple:
    """Summary

    Args:
        series_df (pd.DataFrame): Description
        threshold (float, optional): Description
        verbose (bool, optional): Description
        **kwargs (Dict): Description

    Returns:
        Tuple: Description
    """
    adf_results = None

    try:
        if verbose:
            print(f"\n// {dbg()}  Attempting to perform an "
                  + "Augmented Dickey-Fuller test for stationarity!")


        adf_results = adfuller(series_df.values,
                               )


        if verbose:
            print(f"\n// {dbg()}  Successfully completed Augmented Dickey-Fuller "
                  + "test for stationarity!")

    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't perform the Augmented Dickey-Fuller Test!\n")
        traceback.print_exc()

    return adf_results


def perform_kpss_test(series_df : pd.DataFrame,
                      threshold : float = 0.05,
                      verbose   : bool = False,
                      **kwargs  : Dict) -> Tuple:
    """Summary

    Args:
        series_df (pd.DataFrame): Description
        threshold (float, optional): Description
        verbose (bool, optional): Description
        **kwargs (Dict): Description

    Returns:
        Tuple: Description
    """
    kpss_results = None

    try:
        if verbose:
            print(f"\n// {dbg()}  Attempting to perform a KPSS test for stationarity!")

        kpss_results =


        if verbose:
            print(f"\n// {dbg()}  Successfully completed a KPSS test for stationarity!")

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't perfom the KPSS test for stationarity!\n")
        traceback.print_exc()

    return kpss_results



#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
