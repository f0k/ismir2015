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


class MelBankLayer(lasagne.layers.Layer):
    """
    Creates a mel filterbank layer of `num_bands` triangular filters, with
    the first filter initialized to start at `min_freq` and the last one
    to stop at `max_freq`. Expects to process magnitude spectra created
    from samples at a sample rate of `sample_rate` with a window length of
    `frame_len` samples. Learns a vector of `num_bands + 2` values, with
    the first value giving `min_freq` in mel, and remaining values giving
    the distance to the respective next peak in mel.
    """
    def __init__(self, incoming, sample_rate, frame_len, num_bands, min_freq,
                 max_freq, trainable=True, **kwargs):
        super(MelBankLayer, self).__init__(incoming, **kwargs)
        # mel-spaced peak frequencies
        min_mel = 1127 * np.log1p(min_freq / 700.0)
        max_mel = 1127 * np.log1p(max_freq / 700.0)
        spacing = (max_mel - min_mel) / (num_bands + 1)
        spaces = np.ones(num_bands + 2) * spacing
        spaces[0] = min_mel
        spaces = theano.shared(lasagne.utils.floatX(spaces))  # learned param
        peaks_mel = spaces.cumsum()

        # create parameter as a vector of real-valued peak bins
        peaks_hz = 700 * (T.expm1(peaks_mel / 1127))
        peaks_bin = peaks_hz * frame_len / sample_rate
        self.peaks = self.add_param(peaks_bin,
                shape=(num_bands + 2,), name='peaks', trainable=trainable,
                regularizable=False)

        # store what else is needed
        self.num_bands = num_bands

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.num_bands,)

    def get_output_for(self, input, **kwargs):
        num_bins = self.input_shape[-1] or input.shape[-1]
        x = T.arange(num_bins, dtype=input.dtype).dimshuffle(0, 'x')
        peaks = self.peaks
        l, c, r = peaks[0:-2], peaks[1:-1], peaks[2:]
        # triangles are the minimum of two linear functions f(x) = a*x + b
        # left side of triangles: f(l) = 0, f(c) = 1 -> a=1/(c-l), b=-a*l
        tri_left = (x - l) / (c - l)
        # right side of triangles: f(c) = 1, f(r) = 0 -> a=1/(c-r), b=-a*r
        tri_right = (x - r) / (c - r)
        # combine by taking the minimum of the left and right sides
        tri = T.minimum(tri_left, tri_right)
        # and clip to only keep positive values
        bank = T.maximum(0, tri)

        # the dot product of the input with this filter bank is the output
        return T.dot(input, bank)


class PowLayer(lasagne.layers.Layer):
    def __init__(self, incoming, exponent=lasagne.init.Constant(0), **kwargs):
        super(PowLayer, self).__init__(incoming, **kwargs)
        self.exponent = self.add_param(exponent, shape=(), name='exponent', regularizable=False)
    def get_output_for(self, input, **kwargs):
        return input ** self.exponent


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
    elif cfg['filterbank'] == 'mel_learn':
        layer = MelBankLayer(layer, cfg['sample_rate'], cfg['frame_len'],
                cfg['mel_bands'], cfg['mel_min'], cfg['mel_max'])
    elif cfg['filterbank'] != 'none':
        raise ValueError("Unknown filterbank=%s" % cfg['filterbank'])

    # magnitude transformation, if any
    if cfg['magscale'] == 'log':
        layer = ExpressionLayer(layer, lambda x: T.log(T.maximum(1e-7, x)))
    elif cfg['magscale'] == 'log1p':
        layer = ExpressionLayer(layer, T.log1p)
    elif cfg['magscale'].startswith('log1p_learn'):
        # learnable log(1 + 10^a * x), with given initial a (or default 0)
        a = float(cfg['magscale'][len('log1p_learn'):] or 0)
        a = T.exp(theano.shared(lasagne.utils.floatX(a)))
        layer = lasagne.layers.ScaleLayer(layer, scales=a,
                                          shared_axes=(0, 1, 2, 3))
        layer = ExpressionLayer(layer, T.log1p)
    elif cfg['magscale'].startswith('pow_learn'):
        # learnable x^sigmoid(a), with given initial a (or default 0)
        a = float(cfg['magscale'][len('pow_learn'):] or 0)
        a = T.nnet.sigmoid(theano.shared(lasagne.utils.floatX(a)))
        layer = PowLayer(layer, exponent=a)
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
