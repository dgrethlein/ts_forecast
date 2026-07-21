#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Time Series Plotting Utility<src.utils.plot_ts>` module.


Module Description
==================

Module for plotting time series data onto :class:`matplotlib.Axes`.

.. moduleauthor:: David Grethlein

Module Contents
===============

"""


import traceback

from typing import Tuple

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from .misc import dbg, err


#==============================================================================
#       TIME SERIES PLOTTING FUNCTION(s)
#==============================================================================
def setup_ts_df_plot(verbose : bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """Summary
    """
    plot_fig = None
    plot_ax = None

    try:
        if verbose:
            print(f"\n// {dbg()}  Attempting to set up matplotlib plot "
                  + "for time series DataFrame!\n")

        plot_fig, plot_ax = plt.subplots(figsize=(10,5))

        plot_ax.set_xlabel("Time-steps (Hourly)")
        plot_ax.set_ylabel("Electricity consumption (kwh)")

        plot_fig.suptitle("Hourly electricity consumption (kwh) for from 2011 to 2014.")

        if verbose:
            print(f"\n// {dbg()}  Successfully set up matplotlib plot!\n")

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't set up a matplotlib plot for time series samples!\n")
        traceback.print_exc()

    return (plot_fig, plot_ax)


def plot_ts_df_on_ax(data_df : pd.DataFrame,
                     verbose : bool = False):
    """Summary

    Args:
        ax (plt.Axes): Description
        data_df (pd.DataFrame): Description
        verbose (bool, optional): Description
    """
    try:
        fig, ax = setup_ts_df_plot(verbose=verbose)

        ax.plot(np.array(list(range(len(data_df)))),
                data_df["kwh_electricity_consumed"].values,
                alpha=0.5)

        fig.tight_layout()
        plt.show()

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't plot time series DataFrame(s) on matplotlib Axes!\n")
        traceback.print_exc()


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
