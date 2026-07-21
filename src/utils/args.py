#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Command Line Argument Parsing Utility<src.utils.args>` module.


Module Description
==================

Module for parsing command line arguments using a :class:`argparse.ArgumentParser`.

.. moduleauthor:: David Grethlein

Module Contents
===============

"""


import argparse
import traceback

from typing import Dict

from .misc import dbg, err


#==============================================================================
#       COMMAND LINE ARGUMENT(s) PARSER FUNCTION(s)
#==============================================================================
def get_ts_forecast_args_parser() -> argparse.ArgumentParser:
    """Summary

    Returns:
        argparse.ArgumentParser: Description
    """
    args_parser = None

    try:
        # Command line argument parser for running python modules directly from command line.
        args_parser = argparse.ArgumentParser(description=("Pipeline that will "
                                                           + "train and test time series "
                                                           + "forecasting models using "
                                                           + "electricity consumption data "
                                                           + "sampled hourly from 2011 to 2014."))

        # Adds command line arguments to anticipate to ArgumentParser.
        add_forecast_horizon_arg_to_parser(parser=args_parser)
        add_holdout_percentage_arg_to_parser(parser=args_parser)
        add_num_cv_folds_arg_to_parser(parser=args_parser)
        add_verbose_arg_to_parser(parser=args_parser)

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't get time series forecasting ArgumentParser!\n")
        traceback.print_exc()

    return args_parser


def add_forecast_horizon_arg_to_parser(parser : argparse.ArgumentParser):
    """Summary

    Args:
        parser (argparse.ArgumentParser): Description
    """
    try:
        parser.add_argument("--forecast_horizon",
                            default=240,
                            type=int,
                            required=False,
                            help=("The number of time-steps into the future for "
                                  + "time series forecasting models to predict "
                                  + "the future values of energy consumption. "
                                  + "Default is ``240`` (10 days sampled hourly)."))

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't add `forecast_horizon` arg to ArgumentParser!\n")
        traceback.print_exc()


def add_holdout_percentage_arg_to_parser(parser : argparse.ArgumentParser):
    """Adds the ``holdout_percentage`` argument to an :class:`argparse.ArgumentParser`.
    This is an optional argument that takes only non-negative finite floating point
    numbers in the range (0.0,1.0), and has a default value of 0.20. The holdout
    percentage is the fraction of time series data withheld from the training set of
    any time series forecasting models, used for testing/validating the forecasting
    ability of the trained time series forecasting model.

    Args:
        parser (argparse.ArgumentParser): A command line argument parser.
    """
    try:
        parser.add_argument("--holdout_percentage",
                            default=0.20,
                            type=float,
                            required=False,
                            help=("The holdout percentage of time series data "
                                  + "to be withheld from training sets for the "
                                  + "purposes of testing/validating/evaluating "
                                  + "the forecasting ability of trained time "
                                  + "series forecasting models. Default value is ``0.20``."))

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't add `holdout_percentage` arg to ArgumentParser!\n")
        traceback.print_exc()


def add_num_cv_folds_arg_to_parser(parser : argparse.ArgumentParser):
    """Adds the ``num_cv_folds`` argument to an :class:`argparse.ArgumentParser`.
    This is an optional argument that takes only non-negative finite integers,
    and has a default value of 5.

    Args:
        parser (argparse.ArgumentParser): A command line argument parser.
    """
    try:
        parser.add_argument("--num_cv_folds",
                            default=5,
                            type=int,
                            required=False,
                            help=("The number of folds `k` to split time series dataset into "
                                  + "for `k`-fold cross-validation. Default value is ``5``."))

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't add `num_cv_folds` arg to ArgumentParser!\n")
        traceback.print_exc()


def add_verbose_arg_to_parser(parser : argparse.ArgumentParser):
    """Adds the ``verbose`` argument to an :class:`argparse.ArgumentParser`.
    This is an optional argument that controls whether or not to print out
    [DEBUG]-style messages to the console describing what is happening.

    Args:
        parser (argparse.ArgumentParser): A command line argument parser.
    """
    try:
        parser.add_argument("-V",
                            "--verbose",
                            action="store_true",
                            default=False,
                            required=False,
                            help=("Indicator whether to print [DEBUG]-style console output. "
                                  + "Default value is ``False``."))

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't add `verbose` arg to ArgumentParser!\n")
        traceback.print_exc()


#==============================================================================
#       PARSING COMMAND LINE ARGUMENT(s) FUNCTION(s)
#==============================================================================
def parse_run_ts_forecast_args() -> Dict:
    """Parses all of the arguments from a :class:`argparse.ArgumentParser` that were
    configured to receive command-line arguments for running time series forecasting
    scripts; storing the result into a dictionary.

    Returns:
        Dict: A dictionary containing all of the command-line arguments that were
        parsed by a :class:`argparse.ArgumentParser` in order to
        run a time series forecasting.
    """
    pargs = None

    try:
        # Parses the command line arguments and stores results in a dictionary.
        pargs = vars(get_ts_forecast_args_parser().parse_args())

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't parse the run time series forecasting "
              + "arguments from the command line!\n")
        traceback.print_exc()

    return pargs


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
