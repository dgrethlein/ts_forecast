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

from .utils.args import add_ts_cluster_args_parser
from .utils.args import add_ts_forecast_args_parser
from .utils.args import get_args_parser
from .utils.args import parse_args

from .utils.misc import dbg, err




def get_parsed_args_as_dict() -> Dict:
    """Summary

    Returns:
        Dict: Description
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


#==============================================================================
#       SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":

    print(f"\n// {dbg()}  Running File['{__file__}'] as __main__!\n")


    get_parsed_args_as_dict()


    print(f"\n// {dbg()}  All done here, nothing to see!\n")
