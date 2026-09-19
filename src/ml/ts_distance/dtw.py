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
import pandas as pd

from ...utils.misc import dbg, err


#==============================================================================
#       STANDALONE COMPUTE DTW DISTANCE MEASURE FUNCTION(s)
#==============================================================================
def compute_dtw(first_obj        : Union[np.ndarray,pd.DataFrame],
                second_obj       : Union[np.ndarray,pd.DataFrame],
                return_warp_path : bool = False) -> Union[float,Tuple[float,List[Tuple[int,int]]]]:
    """Computes the DTW distance between two objects (:class:`numpy arrays<numpy.ndarray>` or
    :class:`pandas DataFrame(s)<pandas.DataFrame>`). Optionally returns the DTW warping
    path that minimizes the sum of residual differences between aligned frames from both objects.

    Args:
        first_obj (Union[np.ndarray, pd.DataFrame]): The first object to be compared.
        second_obj (Union[np.ndarray, pd.DataFrame]): The second object to be compared.
        return_warp_path (bool, optional): Boolean indicator of whether the DTW warping path will
            be returned as well. Default is `False`.

    Returns:
        Union[float, Tuple[float, List[Tuple[int, int]]]]: Either just the DTW distance between
            two object. Or a tuple containing both the DTW distance and a list of tuples of
            two integers each that are the indices of the minimum warping path DTW finds when
            constructing an aggregate warping cost matrix. What is returned is controlled by
            the :attr:`return_warp_path` parameter.

    """
    dtw_tuple = None

    try:
        dtw_val = None
        warp_matrix = None

        # Dealing with univariate time series.
        if first_obj.ndim == 1 and second_obj.ndim == 1:
            if isinstance(first_obj, pd.DataFrame) and isinstance(second_obj, pd.DataFrame):
                dtw_val, warp_matrix = dtw.warping_paths_fast(first_obj.values.astype(float),
                                                              second_obj.values.astype(float))
            else:
                dtw_val, warp_matrix = dtw.warping_paths_fast(first_obj.astype(float),
                                                              second_obj.astype(float))

            # Returns the minimum warping path indices as well.
            if return_warp_path:
                dtw_warp_path = dtw.best_path(warp_matrix)

                dtw_tuple = (dtw_val,
                             dtw_warp_path)

            else:
                dtw_tuple = dtw_val

        # Dealing with multivariate time series.
        elif first_obj.ndim == 2 and second_obj.ndim == 2:
            if isinstance(first_obj, pd.DataFrame) and isinstance(second_obj, pd.DataFrame):
                dtw_val, warp_matrix = dtw_ndim.warping_paths_fast(first_obj.values.astype(float),
                                                                   second_obj.values.astype(float))
            else:
                dtw_val, warp_matrix = dtw_ndim.warping_paths_fast(first_obj.astype(float),
                                                                   second_obj.astype(float))

            # Returns the minimum warping path indices as well.
            if return_warp_path:
                dtw_warp_path = dtw.best_path(warp_matrix)

                dtw_tuple = (dtw_val,
                             dtw_warp_path)

            else:
                dtw_tuple = dtw_val

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
