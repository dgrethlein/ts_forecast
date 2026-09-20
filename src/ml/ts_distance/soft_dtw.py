#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Soft-min Dynamic Time Warping (Soft_DTW) Distance<src.ml.ts_distance.soft_dtw>` module.

Module Description
==================

Module for code implementing the Soft-min Dynamic Time Warping distance functions used in the
:mod:`Machine Learning Time Series Distance<src.ml.ts_distance>` package.

.. moduleauthor:: David Grethlein

Module Contents
===============

"""

import traceback
from typing import Union

import numpy as np
import pandas as pd

from ...sdtw import SoftDTW

from ...utils.misc import dbg, err


#==============================================================================
#       STANDALONE COMPUTE SOFT-MIN DTW DISTANCE MEASURE FUNCTION(s)
#==============================================================================
def compute_soft_dtw(first_obj  : Union[np.ndarray,pd.DataFrame],
                     second_obj : Union[np.ndarray,pd.DataFrame],
                     gamma      : float = 0.1) -> float:
    """Computes the soft-min DTW distance computed between two time series objects.

    Args:
        first_obj (Union[np.ndarray, pd.DataFrame]): The first object to be compared.
        second_obj (Union[np.ndarray, pd.DataFrame]): The second object to be compared.
        gamma (float, optional): Smoothing parameter for Soft-min DTW computation.
            Must be a valid percentage in range [0.0, 1.0], default value is 0.1.

    Returns:
        float: The soft-min DTW distance computed between two time series objects.
    """
    soft_dtw_value = None

    try:
        print(f"soft-dtw")

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't compute soft-min DTW!\n")
        traceback.print_exc()

    return soft_dtw_value


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
