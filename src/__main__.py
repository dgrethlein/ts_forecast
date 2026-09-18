#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Main<src.mains>` module.

Package Description
===================

:mod:`Main Source<src.__main__>` module.

.. moduleauthor:: David Grethlein

Package Contents
================

"""


import traceback
from typing import Dict

from .pre_process import load_dataset_df_into_series_dfs

from .utils.args import add_ts_cluster_args_parser
from .utils.args import add_ts_forecast_args_parser
from .utils.args import get_args_parser
from .utils.args import parse_args

from .utils.misc import dbg, err


#==============================================================================
#       TOP-LEVEL MAIN PACKAGE SCRIPT FOR INVOKATION FUNCTION(s)
#==============================================================================
def get_parsed_args_as_dict() -> Dict:
    """Gets a dictionary containing the parsed command line arguments for running
    experiments on time series data in a local (or remote) computing environment.

    Returns:
        Dict: A dictionary containing command line arguments parsed by
            an :class:`argparse.ArgumentParser`.
    """
    pargs = None

    try:
        main_parser, sub_parsers = get_args_parser()
        add_ts_cluster_args_parser(sub_parsers)
        add_ts_forecast_args_parser(sub_parsers)
        pargs = parse_args(main_parser)

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't get parsed arguments as a dictionary!\n")
        traceback.print_exc()

    return pargs


def main():
    """Main function for running experiments using this project's (./src/) top-level package.
    """
    try:
        # Parse command line arguments.
        main_args = get_parsed_args_as_dict()

        # Run time series clustering experiments.
        if main_args["command"] == "cluster":
            ts_cluster(main_args)

        # Run time series forecasting experiments.
        elif main_args["command"] == "forecast":
            ts_forecast(main_args)

        # Run time series prototyping experiments.
        elif main_args["command"] == "prototype":
            ts_prototype(main_args)

    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't run main package script invokation function!\n")
        traceback.print_exc()


#==============================================================================
#       COMMAND-SPECIFIC TOP-LEVEL MAIN PACKAGE FUNCTION(s)
#==============================================================================
def ts_cluster(args_dict : Dict):
    """Top-level function for running all time series clustering experiments.

    Args:
        args_dict (Dict): A dictionary containing command line arguments parsed by
            an :class:`argparse.ArgumentParser`.
    """
    dfs, names = load_dataset_df_into_series_dfs()


def ts_forecast(args_dict : Dict):
    """Top-level function for running all time series forecasting experiments.

    Args:
        args_dict (Dict): A dictionary containing command line arguments parsed by
            an :class:`argparse.ArgumentParser`.
    """
    dfs, names = load_dataset_df_into_series_dfs()


def ts_prototype(args_dict : Dict):
    """Top-level function for running all time series prototyping experiments.

    Args:
        args_dict (Dict): A dictionary containing command line arguments parsed by
            an :class:`argparse.ArgumentParser`.
    """
    dfs, names = load_dataset_df_into_series_dfs()


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    # Top-level function for running experiments.
    main()

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
