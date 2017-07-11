#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple file-based numpy array cache.

Author: Jan Schlüter
"""

import os

import numpy as np


class NpyOnDemand(object):
    """
    Proxy object for a read-only numpy array backed by a npy file stored
    in ``filename``. Can be used in place of in-memory numpy arrays as
    long as the data is always accessed via indexing, and only the shape,
    dtype and length are needed. In contrast to a memory-mapped numpy
    array, this proxy does not add to the number of open files.
    """
    def __init__(self, filename, shape=None, dtype=None):
        self.filename = filename
        self._shape = shape
        self._dtype = dtype

    def load_metadata(self):
        f = np.load(self.filename, mmap_mode='r')
        self._shape = f.shape
        self._dtype = f.dtype

    @property
    def shape(self):
        if self._shape is None:
            self.load_metadata()
        return self._shape

    @property
    def dtype(self):
        if self._dtype is None:
            self.load_metadata()
        return self._dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, obj):
        return np.load(self.filename, mmap_mode='r')[obj]


def cached(filename, func, *args, **kwargs):
    """
    If ``filename`` exists, loads and returns it. Otherwise, calls
    the given function with the given arguments, saves the result
    to ``filename`` and returns it. If ``filename`` is false-ish,
    directly calls the given function and returns its result.
    An optional ``loading_mode`` keyword argument controls whether
    the array is directly loaded into ``"memory"``, loaded as a
    ``"memmap"``, or returned as a proxy object reading the data
    ``"on-demand"`` (the latter two require a valid ``filename``).
    """
    loading_mode = kwargs.pop('loading_mode', 'memory')
    if filename and os.path.exists(filename):
        if loading_mode == 'memory':
            return np.load(filename)
        elif loading_mode == 'memmap':
            return np.load(filename, mmap_mode='r')
        elif loading_mode == 'on-demand':
            return NpyOnDemand(filename)
        else:
            raise ValueError("unsupported loading_mode '%s'" % loading_mode)
    else:
        result = func(*args, **kwargs)
        if filename:
            np.save(filename, result)
            if loading_mode == 'memmap':
                return np.load(filename, mmap_mode='r')
            elif loading_mode == 'on-demand':
                return NpyOnDemand(filename, shape=result.shape,
                                   dtype=result.dtype)
        if loading_mode != 'memory':
            raise ValueError("unsupported loading_mode '%s'" % loading_mode)
        return result
