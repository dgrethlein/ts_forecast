#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Dynamic Time Warping (DTW) Distance<src.ml.ts_distance.dtw>` module.

Module Description
==================

Module for code implementing the Dynamic Time Warping distance functions used in the
:mod:`Machine Learning Time Series Distance<src.ml.ts_distance>` package.

.. moduleauthor:: David Grethlein

Module Contents
===============

"""


import traceback

from typing import List, Tuple, Union

from dtaidistance import dtw
from dtaidistance import dtw_ndim

import numpy as np

from ...utils.misc import dbg, err


#==============================================================================
#       STANDALONE COMPUTE DTW DISTANCE MEASURE FUNCTION(s)
#==============================================================================
def compute_dtw(first_arr        : np.ndarray,
                second_arr       : np.ndarray,
                return_warp_path : bool = False) -> Union[float,Tuple[float,List[Tuple[int,int]]]]:
    """Computes the DTW distance between two arrays. Optionally returns the DTW warping
    path that minimizes the sum of residual differences between aligned frames from both arrays.

    Args:
        first_arr (np.ndarray): The first array to be compared.
        second_arr (np.ndarray): The second array to be compared.
        return_warp_path (bool, optional): Boolean indicator of whether the DTW warping path will
            be returned as well. Default is `False`.

    Returns:
        Union[float, Tuple[float, List[Tuple[int, int]]]]:
            Either just the DTW distance between
            two arrays. Or a tuple containing both the DTW distance and a list of tuples of
            two integers each that are the indices of the minimum warping path DTW finds when
            constructing an aggregate warping cost matrix. What is returned is controlled by
            the :attr:`return_warp_path` parameter.

    """
    dtw_tuple = None

    try:
        # Dealing with univariate time series.
        if first_arr.ndim == 1 and second_arr.ndim == 1:
            dtw_value, warp_matrix = dtw.warping_paths_fast(first_arr.astype(float),
                                                            second_arr.astype(float))

            # Returns the minimum warping path indices as well.
            if return_warp_path:
                dtw_warp_path = dtw.best_path(warp_matrix)

                dtw_tuple = (dtw_value,
                             dtw_warp_path)

            else:
                dtw_tuple = dtw_value

        # Dealing with multivariate time series.
        elif first_arr.ndim == 2 and second_arr.ndim == 2:
            dtw_value, warp_matrix = dtw_ndim.warping_paths_fast(first_arr.astype(float),
                                                                 second_arr.astype(float))

            # Returns the minimum warping path indices as well.
            if return_warp_path:
                dtw_warp_path = dtw.best_path(warp_matrix)

                dtw_tuple = (dtw_value,
                             dtw_warp_path)

            else:
                dtw_tuple = dtw_value

    except (AttributeError, IndexError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't compute DTW!\n")
        traceback.print_exc()

    return dtw_tuple


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
