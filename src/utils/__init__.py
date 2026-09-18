#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Utilities<src.utils>` Sub-package.

Package Description
===================

Package for importing code from all :mod:`Utilities<src.utils>` modules.

.. moduleauthor:: David Grethlein

Package Contents
================

"""


#==============================================================================
#       COMMAND LINE ARGUMENT(s) PARSER FUNCTION(s)
#==============================================================================
from .args import get_args_parser
from .args import add_ts_cluster_args_parser
from .args import add_ts_forecast_args_parser
from .args import add_ts_prototype_args_parser


#==============================================================================
#       COMMAND LINE ADD ARGUMENT(s) TO PARSE FUNCTION(s)
#==============================================================================
from .args import add_cluster_method_arg_to_parser
from .args import add_difference_period_arg_to_parser
from .args import add_forecast_horizon_arg_to_parser
from .args import add_holdout_percentage_arg_to_parser
from .args import add_num_clusters_arg_to_parser
from .args import add_num_cluster_inits_arg_to_parser
from .args import add_num_cluster_iters_arg_to_parser
from .args import add_num_cv_folds_arg_to_parser
from .args import add_ts_dist_func_arg_to_parser
from .args import add_random_seed_arg_to_parser
from .args import add_verbose_arg_to_parser


#==============================================================================
#       PARSING COMMAND LINE ARGUMENT(s) FUNCTION(s)
#==============================================================================
from .args import parse_args


#==============================================================================
#       CONSOLE OUTPUT FORMATTING FUNCTION(s)
#==============================================================================
from .misc import bold_str
from .misc import fail_str
from .misc import header_str
from .misc import ok_blue_str
from .misc import ok_cyan_str
from .misc import ok_green_str
from .misc import underline_str
from .misc import warn_str


#==============================================================================
#       CONSOLE OUTPUT DEBUGGING/ERROR LOGGING FUNCTION(s)
#==============================================================================
from .misc import dbg
from .misc import err


#==============================================================================
#       EVALUATING TYPE-CAST ATTEMPT LEGITIMACY FUNCTION(s)
#==============================================================================
from .misc import repr_a_float


#==============================================================================
#       REUSABLE VALUE TESTING FUNCTION(s)
#==============================================================================
from .misc import is_finite_float
from .misc import is_finite_int
from .misc import is_nonneg_finite_float
from .misc import is_nonneg_finite_int
from .misc import is_non_empty_str
