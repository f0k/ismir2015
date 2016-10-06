#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network architecture definition for Singing Voice Detection experiment.

Author: Jan Schlüter
"""

import numpy as np
import lasagne
from lasagne.layers import (InputLayer, Conv2DLayer, MaxPool2DLayer,
                            DenseLayer, dropout)

def architecture(input_var, input_shape):
    layer = InputLayer(input_shape, input_var)
    kwargs = dict(nonlinearity=lasagne.nonlinearities.leaky_rectify,
                  W=lasagne.init.Orthogonal())
    layer = Conv2DLayer(layer, 64, 3, **kwargs)
    layer = Conv2DLayer(layer, 32, 3, **kwargs)
    layer = MaxPool2DLayer(layer, 3)
    layer = Conv2DLayer(layer, 128, 3, **kwargs)
    layer = Conv2DLayer(layer, 64, 3, **kwargs)
    layer = MaxPool2DLayer(layer, 3)
    layer = DenseLayer(dropout(layer, 0.5), 256, **kwargs)
    layer = DenseLayer(dropout(layer, 0.5), 64, **kwargs)
    layer = DenseLayer(dropout(layer, 0.5), 1,
                       nonlinearity=lasagne.nonlinearities.sigmoid,
                       W=lasagne.init.Orthogonal())
    return layer
