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

from typing import Dict, Tuple

from .misc import dbg, err


#==============================================================================
#       COMMAND LINE ARGUMENT(s) PARSER FUNCTION(s)
#==============================================================================
def get_args_parser() -> Tuple[argparse.ArgumentParser,argparse._SubParsersAction]:
    """Creates and returns a top-level :class:`argparse.ArgumentParser` for
    parsing command line arguments for running experiments via script invokation.

    Returns:
        argparse.ArgumentParser: A command line argument parser.
    """
    args_parser = None
    sub_parse_cmds = None

    try:
        args_parser = argparse.ArgumentParser(prog="PROJECT",
                                              description=("Process command line arguments "
                                                           + "for time series forecasting."))
        sub_parse_cmds = args_parser.add_subparsers(dest="command",
                                                    required=True,
                                                    help="Available sub-commands")

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't get ArgumentParser!\n")
        traceback.print_exc()

    return (args_parser, sub_parse_cmds)


def add_ts_cluster_args_parser(sub_parsers : argparse._SubParsersAction):
    """Adds an :class:`argparse.ArgumentParser` that has been set up to receive
    several command line arguments when a python script is invoked for the purpose
    of clustering time series values.

    Args:
        sub_parsers (argparse._SubParsersAction): Argument parser sub-parser command actions,
            used for adding command-specific arguments to :class:`argparse.ArgumentParser`.
    """
    try:
        # Command line argument sub-parser for running time series clustering
        # modules directly from the command line.
        tsclu_parser = sub_parsers.add_parser("cluster",
                                              help="Cluster time series data.",
                                              description=("Pipeline that will "
                                                           + "train and test time series "
                                                           + "clustering models using "
                                                           + "electricity consumption data "
                                                           + "sampled hourly from 2011 to 2014."))

        # Adds command line arguments to anticipate to ArgumentParser.
        add_num_cv_folds_arg_to_parser(parser=tsclu_parser)
        add_verbose_arg_to_parser(parser=tsclu_parser)

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't add time series clustering ArgumentParser!\n")
        traceback.print_exc()


def add_ts_forecast_args_parser(sub_parsers : argparse._SubParsersAction):
    """Adds an :class:`argparse.ArgumentParser` that has been set up to receive
    several command line arguments when a python script is invoked for the purpose
    of forecasting time series values.

    Args:
        sub_parsers (argparse._SubParsersAction): Argument parser sub-parser command actions,
            used for adding command-specific arguments to :class:`argparse.ArgumentParser`.
    """
    try:
        # Command line argument sub-parser for running time series forecasting
        # modules directly from the command line.
        tsf_parser = sub_parsers.add_parser("forecast",
                                            help="Forecast time series data.",
                                            description=("Pipeline that will "
                                                         + "train and test time series "
                                                         + "forecasting models using "
                                                         + "electricity consumption data "
                                                         + "sampled hourly from 2011 to 2014."))

        # Adds command line arguments to anticipate to ArgumentParser.
        add_difference_period_arg_to_parser(parser=tsf_parser)
        add_forecast_horizon_arg_to_parser(parser=tsf_parser)
        add_holdout_percentage_arg_to_parser(parser=tsf_parser)
        add_num_cv_folds_arg_to_parser(parser=tsf_parser)
        add_verbose_arg_to_parser(parser=tsf_parser)

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't add time series forecasting ArgumentParser!\n")
        traceback.print_exc()


def add_ts_prototype_args_parser(sub_parsers : argparse._SubParsersAction):
    """Adds an :class:`argparse.ArgumentParser` that has been set up to receive
    several command line arguments when a python script is invoked for the purpose
    of prototyping time series values.

    Args:
        sub_parsers (argparse._SubParsersAction): Argument parser sub-parser command actions,
            used for adding command-specific arguments to :class:`argparse.ArgumentParser`.
    """
    try:
        tsp_parser = sub_parsers.add_parser("prototype",
                                            help="Prototype time series data.",
                                            description=("Pipeline that will "
                                                         + "train and test time series "
                                                         + "prototyping models using "
                                                         + "electricity consumption data "
                                                         + "sampled hourly from 2011 to 2014."))

        # Adds command line arguments to anticipate to ArgumentParser.
        add_verbose_arg_to_parser(parser=tsp_parser)

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't add time series prototyping ArgumentParser!\n")
        traceback.print_exc()


#==============================================================================
#       COMMAND LINE ADD ARGUMENT(s) TO PARSE FUNCTION(s)
#==============================================================================
def add_cluster_method_arg_to_parser(parser : argparse.ArgumentParser):
    """Adds the ``cluster_method`` argument to an :class:`argparse.ArgumentParser`.
    This is an optional argument that takes non-empty string values, and has
    a default value of ``K_Medoids``.

    .. note::
        The ``cluster_method`` argument must be chosen from one of the following options:

            * ``CLARA``     - Clustering for Large Applications (CLARA).
            * ``DBSCAN``    - Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
            * ``GMM``       - Gaussian Mixture Model (GMM).
            * ``IDEC``      - Improved Deep Embedding Clustering (IDEC).
            * ``K_Medoids`` - K Medoids partitioning around medoids (PAM).
            * ``OPTICS``    - Ordering Points to Identify the Cluster Structure (OPTICS).

    Args:
        parser (argparse.ArgumentParser): A command line argument parser.
    """
    try:
        parser.add_argument("--cluster_method",
                            default="K_Means",
                            choices=["CLARA",
                                     "DBSCAN",
                                     "GMM",
                                     "IDEC",
                                     "K_Medoids",
                                     "OPTICS"],
                            required=False,
                            type=str,
                            help="")

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't add cluster method argument to ArgumentParser!\n")
        traceback.print_exc()


def add_difference_period_arg_to_parser(parser : argparse.ArgumentParser):
    """Adds the ``difference_period`` argument to an :class:`argparse.ArgumentParser`.
    This is an optional argument that takes only non-negative finite integers, and has
    a default value of 1; corresponding to differencing adjacent time-steps. The difference
    period is designed to reveal seaonality in the time series data.

    Args:
        parser (argparse.ArgumentParser): A command line argument parser.
    """
    try:
        parser.add_argument("--difference_period",
                            default=1,
                            type=int,
                            required=False,
                            help=("The differencing period for testing for "
                                  + "seasonality in the time series data. "
                                  + "Default value is ``1`` time-step (1 hour)."))

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't add the `difference_period` arg to ArgumentParser!\n")
        traceback.print_exc()


def add_forecast_horizon_arg_to_parser(parser : argparse.ArgumentParser):
    """Adds the ``forecast_horizon`` argument to an :class:`argparse.ArgumentParser`.
    This is an optional argument that takes only non-negative finite integers, and
    has a default value of 240; corresponding to 10 days of hourly obeservations. The
    forecast horizon is how far into the future will a time series forecasting model
    be asked to predict.

    Args:
        parser (argparse.ArgumentParser): A command line argument parser.
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


def add_num_clusters_arg_to_parser(parser : argparse.ArgumentParser):
    """Adds the ``num_clusters`` argument to an :class:`argparse.ArgumentParser`.
    This is an optional argument that takes only positive finite integers greater than 1,
    and has a default value of 10.

    Args:
        parser (argparse.ArgumentParser): A command line argument parser.
    """
    try:
        parser.add_argument("--num_clusters",
                            default=10,
                            type=int,
                            required=False,
                            help=("The number of clusters to produce from time series dataset. "
                                  + "Default value is ``10``."))

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't add `num_clusters` arg to ArgumentParser!\n")
        traceback.print_exc()


def add_num_cluster_inits_arg_to_parser(parser : argparse.ArgumentParser):
    """Adds the ``num_cluster_inits`` argument to an :class:`argparse.ArgumentParser`.
    This is an optional argument that takes only positive finite integers,
    and has a default value of 5.

    Args:
        parser (argparse.ArgumentParser): A command line argument parser.
    """
    try:
        parser.add_argument("--num_cluster_inits",
                            default=5,
                            type=int,
                            required=False,
                            help=("The number of cluster initializations to use "
                                  + "in time series clustering experiments. "
                                  + "Default value is ``5``."))

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}. Couldn't add the `num_cluster_inits` arg to ArgumentParser!\n")
        traceback.print_exc()


def add_num_cluster_iters_arg_to_parser(parser : argparse.ArgumentParser):
    """Adds the ``num_cluster_iters`` argument to an :class:`argparse.ArgumentParser`.
    This is an optional argument that takes only positive finite integers,
    and has a default value of 25.

    Args:
        parser (argparse.ArgumentParser): A command line argument parser.
    """
    try:
        parser.add_argument("--num_cluster_iters",
                            default=25,
                            type=int,
                            required=False,
                            help=("The number of clustering iterations for grouping "
                                  + "samples from time series dataset into clusters. "
                                  + "Defaul value is ``25``."))

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}. Couldn't add `num_iterations` arg to ArgumentParser!\n")
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


def add_random_seed_arg_to_parser(parser : argparse.ArgumentParser):
    """Add the ``random_seed`` argument to an :class:`argparse.ArgumentParser`.
    The is an optional argument that takes only non-negative finite integers,
    and has a default value of 0.

    Args:
        parser (argparse.ArgumentParser): A command line argument parser.
    """
    try:
        parser.add_argument("--random_seed",
                            default=0,
                            type=int,
                            required=False,
                            help=("A random number generator seed value to be "
                                  + "used in experiments. Default value is ``0``."))

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't add `random_seed` arg to ArgumentParser!\n")
        traceback.print_exc()


def add_ts_dist_func_arg_to_parser(parser : argparse.ArgumentParser):
    """Adds the ``ts_dist_func`` argument to an :class:`argparse.ArgumentParser`.
    This is an optional argument that takes only non-empty strings,
    and has a default value of `DTW`.

    .. note::
        The ``ts_dist_func`` argument must be chosen from one of the following options:

            * ``Cos`` - Cosine distance (Cos).
            * ``DTW`` - Dynamic Time Warping (DTW).
            * ``Euc`` - Euclidean (Euc).
            * ``SAX`` - Symbolic Aggregate Approxmiation (SAX).

    Args:
        parser (argparse.ArgumentParser): A command line argument parser.
    """
    try:
        parser.add_argument("--ts_dist_func",
                            default="DTW",
                            choices=["Cos",
                                     "DTW",
                                     "Euc",
                                     "SAX"],
                            required=False,
                            help=("Name of the time series distance function to be used "
                                  + "in experiments to numerically compare time series to "
                                  + "one another. Default value is ``DTW``."))

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't add `ts_dist_func` arg to ArgumentParser!\n")
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
def parse_args(parser : argparse.ArgumentParser) -> Dict:
    """Parses all of the arguments from a :class:`argparse.ArgumentParser`,
    storing the result into a dictionary.

    Returns:
        Dict: A dictionary containing all of the command-line arguments that were
        parsed by a :class:`argparse.ArgumentParser`.

    Args:
        parser (argparse.ArgumentParser): Description
    """
    parsed_args = None

    try:
        # Parses the command line arguments and stores results in a dictionary.
        parsed_args = vars(parser.parse_args())

    except (AttributeError, TypeError, ValueError):
        print(f"\n// {err()}  Couldn't parse arguments from the command line!\n")
        traceback.print_exc()

    return parsed_args


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")

    print(f"\n// {dbg()}  All done here, nothing to see!\n")
