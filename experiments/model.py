#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network architecture definition for Singing Voice Detection experiment.

Author: Jan Schlüter
"""

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import (InputLayer, Conv2DLayer, MaxPool2DLayer,
                            DenseLayer, ExpressionLayer, dropout, batch_norm)
batch_norm_vanilla = batch_norm
try:
    from lasagne.layers.dnn import batch_norm_dnn as batch_norm
except ImportError:
    pass

def architecture(input_var, input_shape, cfg):
    layer = InputLayer(input_shape, input_var)
    layer = ExpressionLayer(layer, T.log1p)
    layer = batch_norm_vanilla(layer, axes=(0, 2), beta=None, gamma=None)
    kwargs = dict(nonlinearity=lasagne.nonlinearities.leaky_rectify,
                  W=lasagne.init.Orthogonal())
    layer = Conv2DLayer(layer, 64, 3, **kwargs)
    layer = batch_norm(layer)
    layer = Conv2DLayer(layer, 32, 3, **kwargs)
    layer = batch_norm(layer)
    layer = MaxPool2DLayer(layer, 3)
    layer = Conv2DLayer(layer, 128, 3, **kwargs)
    layer = batch_norm(layer)
    layer = Conv2DLayer(layer, 64, 3, **kwargs)
    layer = batch_norm(layer)
    layer = Conv2DLayer(layer, 128, (3, 115), **kwargs)
    layer = batch_norm(layer)
    layer = MaxPool2DLayer(layer, (1, 4))
    layer = DenseLayer(dropout(layer, 0.5), 256, **kwargs)
    layer = batch_norm(layer)
    layer = DenseLayer(dropout(layer, 0.5), 64, **kwargs)
    layer = batch_norm(layer)
    layer = DenseLayer(dropout(layer, 0.5), 1,
                       nonlinearity=lasagne.nonlinearities.sigmoid,
                       W=lasagne.init.Orthogonal())
    return layer
