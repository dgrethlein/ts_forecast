#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`Soft-min Dynamic Time Warping Distance<src.sdtw>` Sub-package.

Package Description
===================

Package for importing code from all 
:mod:`Soft-min Dynamic Time Warping Distance<src.sdtw>` modules.

.. moduleauthor:: David Grethlein

Package Contents
================

Code copied and adapted from Marco Cuturi and Mathieu Blondel's repo:
https://github.com/mblondel/soft-dtw/ 

"""


from .barycenter import sdtw_barycenter
from .soft_dtw import SoftDTW
