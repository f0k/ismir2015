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

    # filterbank, if any
    if cfg['filterbank'] == 'mel':
        import audio
        filterbank = audio.create_mel_filterbank(
                cfg['sample_rate'], cfg['frame_len'], cfg['mel_bands'],
                cfg['mel_min'], cfg['mel_max'])
        filterbank = filterbank[:input_shape[3]].astype(theano.config.floatX)
        layer = DenseLayer(layer, num_units=cfg['mel_bands'],
                num_leading_axes=-1, W=T.constant(filterbank), b=None,
                nonlinearity=None)
    elif cfg['filterbank'] != 'none':
        raise ValueError("Unknown filterbank=%s" % cfg['filterbank'])

    # magnitude transformation, if any
    if cfg['magscale'] == 'log':
        layer = ExpressionLayer(layer, lambda x: T.log(T.maximum(1e-7, x)))
    elif cfg['magscale'] == 'log1p':
        layer = ExpressionLayer(layer, T.log1p)
    elif cfg['magscale'] != 'none':
        raise ValueError("Unknown magscale=%s" % cfg['magscale'])

    # standardization per frequency band
    layer = batch_norm_vanilla(layer, axes=(0, 2), beta=None, gamma=None)

    # convolutional neural network
    kwargs = dict(nonlinearity=lasagne.nonlinearities.leaky_rectify,
                  W=lasagne.init.Orthogonal())
    maybe_batch_norm = batch_norm if cfg['arch.batch_norm'] else lambda x: x
    layer = Conv2DLayer(layer, 64, 3, **kwargs)
    layer = maybe_batch_norm(layer)
    layer = Conv2DLayer(layer, 32, 3, **kwargs)
    layer = maybe_batch_norm(layer)
    layer = MaxPool2DLayer(layer, 3)
    layer = Conv2DLayer(layer, 128, 3, **kwargs)
    layer = maybe_batch_norm(layer)
    layer = Conv2DLayer(layer, 64, 3, **kwargs)
    layer = maybe_batch_norm(layer)
    if cfg['arch'] == 'ismir2015':
        layer = MaxPool2DLayer(layer, 3)
    elif cfg['arch'] == 'ismir2016':
        layer = Conv2DLayer(layer, 128, (3, layer.output_shape[3] - 3), **kwargs)
        layer = maybe_batch_norm(layer)
        layer = MaxPool2DLayer(layer, (1, 4))
    else:
        raise ValueError('Unknown arch=%s' % cfg['arch'])
    layer = DenseLayer(dropout(layer, 0.5), 256, **kwargs)
    layer = maybe_batch_norm(layer)
    layer = DenseLayer(dropout(layer, 0.5), 64, **kwargs)
    layer = maybe_batch_norm(layer)
    layer = DenseLayer(dropout(layer, 0.5), 1,
                       nonlinearity=lasagne.nonlinearities.sigmoid,
                       W=lasagne.init.Orthogonal())
    return layer
