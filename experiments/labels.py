#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Label preparation routines.

Author: Jan Schlüter
"""

import numpy as np

def create_aligned_targets(segments, timestamps, dtype=np.float32):
    targets = np.zeros(len(timestamps), dtype=dtype)
    if not segments:
        return targets
    starts, ends, labels = zip(*segments)
    starts = np.searchsorted(timestamps.ravel(), starts)
    ends = np.searchsorted(timestamps.ravel(), ends)
    for a, b, l in zip(starts, ends, labels):
        targets[a:b] = l
    return targets
