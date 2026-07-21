#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Miscellaneous Utilities<src.utils.misc>` module.


Module Description
==================

Module for miscellaneous utilities including but not limited to console output
formatting, evaluating type-cast attempt legitimacy, and more.

.. moduleauthor:: David Grethlein

Module Contents
===============

Attributes:
    BOLD (str): ANSI escape sequence for bolded console output.
    ENDC (str): ANSI escape sequence for ending other applied effects.
    FAIL (str): ANSI escape sequence for formatting failure/error
                console output (typically red text foreground color).
    HEADER (str): ANSI escape sequence for formatting header console output
                  (typically magenta text foregroung color).
    OKBLUE (str): ANSI escape sequence for formatting OK console output in blue.
    OKCYAN (str): ANSI escape sequence for formatting OK console output in cyan.
    OKGREEN (str): ANSI escape sequence for formatting OK console output in green.
    UNDERLINE (str): ANSI escape sequence for formatting underlined console output.
    WARNING (str): ANSI escape sequence for formatting non-critical warning
                   console output (typically yellow text foreground color).

"""


import datetime

from typing import Generic

import numpy as np


#==============================================================================
#       ANSI ESCAPE SEQUENCE(s) FOR STDOUT FORMATTING CONSTANT(s)
#==============================================================================
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


#==============================================================================
#       CONSOLE OUTPUT FORMATTING FUNCTION(s)
#==============================================================================
def bold_str(value : str) -> str:
    """Formats the provided :class:`str` value to be printable to the console
    as bolded text.

    Args:
        value (str): The provided :class:`str` value to format as bolded text.

    Returns:
        str: The :class:`str` formatted as bolded text.
    """
    boldstr = None

    try:
        boldstr = f"{BOLD}{value}{ENDC}"

    except (AttributeError, TypeError, ValueError):
        pass

    return boldstr


def fail_str(value : str) -> str:
    """Formats the provided :class:`str` value to be printable to the console
    as failure/error (typically foreground color in red) text.

    Args:
        value (str): The provided :class:`str` value to format as failure/error text.

    Returns:
        str: The :class:`str` formatted as failure/warning text using ANSI escape sequences.
    """
    failstr = None

    try:
        failstr = f"{FAIL}{value}{ENDC}"

    except (AttributeError, TypeError, ValueError):
        pass

    return failstr


def header_str(value : str) -> str:
    """Formats the provided :class:`str` value to be printable to the console
    as header (typically foreground color in magenta) text.

    Args:
        value (str): The provided :class:`str` value to format as header text.

    Returns:
        str: The :class:`str` formatted as header text using ANSI escape sequences.
    """
    headstr = None

    try:
        headstr = f"{HEADER}{value}{ENDC}"

    except (AttributeError, TypeError, ValueError):
        pass

    return headstr


def ok_blue_str(value : str) -> str:
    """Formats the provided :class:`str` value to be printable to the console
    as OK blue (foreground color in blue) text.

    Args:
        value (str): The provided :class:`str` value to format as header text.

    Returns:
        str: The :class:`str` formatted as `OK blue` text using ANSI escape sequences.
    """
    okbluestr = None

    try:
        okbluestr = f"{OKBLUE}{value}{ENDC}"

    except (AttributeError, TypeError, ValueError):
        pass

    return okbluestr


def ok_cyan_str(value : str) -> str:
    """Formats the provided :class:`str` value to be printable to the console
    as OK cyan (foreground color in cyan) text.

    Args:
        value (str): The provided :class:`str` value to format as header text.

    Returns:
        str: The :class:`str` formatted as `OK cyan` text using ANSI escape sequences.
    """
    okcyanstr = None

    try:
        okcyanstr = f"{OKCYAN}{value}{ENDC}"

    except (AttributeError, TypeError, ValueError):
        pass

    return okcyanstr


def ok_green_str(value : str) -> str:
    """Formats the provided :class:`str` value to be printable to the console
    as OK green (foreground color in green) text.

    Args:
        value (str): The provided :class:`str` value to format as header text.

    Returns:
        str: The :class:`str` formatted as `OK green` text using ANSI escape sequences.
    """
    okgreenstr = None

    try:
        okgreenstr = f"{OKGREEN}{value}{ENDC}"

    except (AttributeError, TypeError, ValueError):
        pass

    return okgreenstr


def underline_str(value : str) -> str:
    """Formats the provided :class:`str` value to be printable to the console
    as underlined text.

    Args:
        value (str): The provided :class:`str` value to underline.

    Returns:
        str: The :class:`str` formatted for underlining using ANSI escape sequences.
    """
    underlinestr = None

    try:
        underlinestr = f"{UNDERLINE}{value}{ENDC}"

    except (AttributeError, TypeError, ValueError):
        pass

    return underlinestr


def warn_str(value : str) -> str:
    """Formats the provided :class:`str` value to be printable to the console
    as non-critical warning (typically foreground color in yellow) text.

    Args:
        value (str): The provided :class:`str` value to warn.

    Returns:
        str: The :class:`str` formatted for non-critical warnings using ANSI escape sequences.
    """
    warnstr = None

    try:
        warnstr = f"{WARNING}{value}{ENDC}"

    except (AttributeError, TypeError, ValueError):
        pass

    return warnstr


#==============================================================================
#       CONSOLE OUTPUT DEBUGGING/ERROR LOGGING FUNCTION(s)
#==============================================================================
def dbg(show_date : bool = False) -> str:
    """[DEBUG]-style console output statement, color-coded and ISO 8601
    timestamp formatted.

    Returns:
        str: Color-coded [DEBUG]-style ISO 8601 timestamp

    Args:
        show_date (bool, optional): The :class:`str` formatted for [DEBUG] console output.
    """
    dbg_str = (f"[{ok_blue_str(bold_str('DEBUG'))}:" +
               f" {warn_str(datetime.datetime.now().strftime('%H:%M:%S.%f'))}")

    if show_date:
        dbg_str += f" {header_str(underline_str(datetime.datetime.now().strftime('%a %b %d')))}"

    return dbg_str + "]"


def err(show_date : bool = False) -> str:
    """[ERROR]-style console output statement, color-coded and ISO 8601
    timestamp formatted.

    Returns:
        str: Color-coded [ERROR]-style ISO 8601 timestamp

    Args:
        show_date (bool, optional): The :class:`str` formatted for [ERROR] console output.
    """
    err_str = (f"[{fail_str('ERROR')}:" +
               f" {warn_str(datetime.datetime.now().strftime('%H:%M:%S.%f'))}")

    if show_date:
        err_str += f" {header_str(underline_str(datetime.datetime.now().strftime('%a %b %d')))}"

    return err_str + "]"


#==============================================================================
#       EVALUATING TYPE-CAST ATTEMPT LEGITIMACY FUNCTION(s)
#==============================================================================
def repr_a_float(value : Generic) -> bool:
    """Checks to see if the argument `value` is an object than can be cast to
    a :class:`float` without an error.

    Args:
        value (Generic): Value that will attempt a type-cast to a :class:`float`.

    Returns:
        bool: Boolean indicating whether provided value can be type-cast to
        a: class:`float` or not.
    """
    try:
        float(value)
        return True

    except (ValueError, TypeError):
        return False


#==============================================================================
#       REUSABLE VALUE TESTING FUNCTION(s)
#==============================================================================
def is_finite_float(value : Generic) -> bool:
    """Checks whether the provided value is a finite floating point number.

    Args:
        value (Generic): The value to be inspected.

    Returns:
        bool: Boolean indicating whether the provided value was a finite
            floating point number or not.
    """
    finite_float = False

    try:
        finite_float = (isinstance(value, float) and np.isfinite(value))

    except (AttributeError, TypeError, ValueError):
        pass

    return finite_float


def is_finite_int(value : Generic) -> bool:
    """Checks whether the provided value is a finite integer.

    Args:
        value (Generic): The value to be inspected.

    Returns:
        bool: Boolean indicating whether the provided value was a finite
            integer or not.
    """
    finite_int = False

    try:
        finite_int = (isinstance(value, int) and np.isfinite(value))

    except(AttributeError, TypeError, ValueError):
        pass

    return finite_int


def is_nonneg_finite_float(value : Generic) -> bool:
    """Checks whether the provided value is a non-negative finite floating point number.

    Args:
        value (Generic): The value to be inspected.

    Returns:
        bool: Boolean indicating whether the provided value was a
            non-negative and finite floating pointing number or not.
    """
    is_nonneg_finite = False

    try:
        is_nonneg_finite = (is_finite_float(value) and value >= 0.0)

    except (AttributeError, TypeError, ValueError):
        pass

    return is_nonneg_finite


def is_nonneg_finite_int(value : Generic) -> bool:
    """Checks whether the provided value is a non-negative finite integer.

    Args:
        value (Generic): The value to be inspected.

    Returns:
        bool: Boolean indicating whether the provided value was a
            non-negative integer or not.
    """
    is_nonneg_finite = False

    try:
        is_nonneg_finite = (is_finite_int(value) and value >= 0)

    except (AttributeError, TypeError, ValueError):
        pass

    return is_nonneg_finite


def is_non_empty_str(value : Generic) -> bool:
    """Checks whether the provided value is a non-empty string.

    Args:
        value (Generic): The value to be inspected.

    Returns:
        bool: Boolean indicating whether the provided value was a
            non-empty string or not.
    """
    return (isinstance(value, str) and len(value) > 0)
