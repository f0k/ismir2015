#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provides routines for efficient Z-scoring.

Author: Jan Schlüter
"""

import numpy as np

def compute_mean_std(arrays, axis=0, keepdims=False, dtype=np.float64, ddof=1):
    """
    Computes the total mean and standard deviation of a set of arrays across
    a given axis or set of axes, using Welford's algorithm.
    """
    kwargs = dict(axis=axis, keepdims=True, dtype=dtype)
    n = m = s = 0
    for data in arrays:
        n += len(data)
        delta = (data - m)
        m += delta.sum(**kwargs) / n
        s += (delta * (data - m)).sum(**kwargs)
    s /= (n - ddof)
    np.sqrt(s, s)
    if not keepdims:
        axes = (axis,) if isinstance(axis, int) else axis
        index = tuple(0 if a in axes else slice(None) for a in range(m.ndim))
        return m[index], s[index]
    return m, s
