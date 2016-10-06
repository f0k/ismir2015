#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple file-based numpy array cache.

Author: Jan Schlüter
"""

import os

import numpy as np


def cached(filename, func, *args, **kwargs):
    """
    If ``filename`` exists, loads and returns it. Otherwise, calls
    the given function with the given arguments, saves the result
    to ``filename`` and returns it. If ``filename`` is false-ish,
    directly calls the given function and returns its result.
    """
    if filename and os.path.exists(filename):
        return np.load(filename)
    else:
        result = func(*args, **kwargs)
        if filename:
            np.save(filename, result)
        return result
