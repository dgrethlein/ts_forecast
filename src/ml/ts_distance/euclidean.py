#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Euclidean Distance<src.ml.ts_distance.euclidean>` module.

Module Description
==================

Module for code implementing the Euclidean distance functions used in the
:mod:`Machine Learning Time Series Distance<src.ml.ts_distance>` package.

.. moduleauthor:: David Grethlein

Module Contents
===============

"""


import traceback
from typing import Union

import numpy as np
import pandas as pd


from ...utils.misc import dbg, err


#==========================================================================
#       COMPUTE DISTANCE BETWEEN OBJECT(s) METHOD(s)
#==========================================================================
def compute_euclidean(first_obj  : Union[np.ndarray,pd.DataFrame],
                      second_obj : Union[np.ndarray,pd.DataFrame]) -> float:
    """Computes the Euclidean distance between two :class:`numpy arrays<numpy.ndarray>`
    or two :class:`pandas DataFrames<pandas.DataFrame>`.

    Args:
        first_obj (Union[np.ndarray, pd.DataFrame]): The first object to be compared.
        second_obj (Union[np.ndarray, pd.DataFrame]): The second object to be compared.

    Returns:
        float: The Euclidean distance between two objects.

    """
    distance = None

    try:
        # Both objects must be of the same class
        if first_obj.__class__ == second_obj.__class__:

            # Comparing data stored in numpy arrays
            if isinstance(first_obj, np.ndarray):
                distance = float(np.linalg.norm(first_obj - second_obj))

            # Comparing data stored in pandas DataFrames
            elif isinstance(first_obj, pd.DataFrame):
                distance = float(np.linalg.norm(first_obj.values - second_obj.values))

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't compute the Euclidean distance between objects!\n")
        traceback.print_exc()

    return distance


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
