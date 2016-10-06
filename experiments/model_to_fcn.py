#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Converts a Lasagne CNN to a fully-convolutional network.

Author: Jan Schlüter
"""

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import (InputLayer, Conv2DLayer, MaxPool2DLayer,
                            DenseLayer, DilatedConv2DLayer, MergeLayer)
from copy import deepcopy


class TimeDilatedMaxPool2DLayer(lasagne.layers.MaxPool2DLayer):
    """
    Variant of Lasagne's MaxPool2DLayer that supports dilated pooling windows
    over the third axis (i.e., the temporal axis of spectrograms).
    """
    def __init__(self, *args, **kwargs):
        dilation = kwargs.pop('dilation', (1, 1))
        super(TimeDilatedMaxPool2DLayer, self).__init__(*args, **kwargs)
        self.dilation = lasagne.utils.as_tuple(dilation, 2, int)
        if self.dilation[1] != 1:
            raise NotImplementedError("Only implemented dilation over time")
        if self.stride[0] != 1:
            raise NotImplementedError("Only supports unstrided pooling over time")

    def get_output_shape_for(self, input_shape):
        shape = super(TimeDilatedMaxPool2DLayer, self).get_output_shape_for(input_shape)
        return (shape[0], shape[1],
                lasagne.layers.pool.pool_output_length(
                        input_shape[2],
                        pool_size=(self.pool_size[0] - 1) * self.dilation[0] + 1,
                        stride=self.stride[0],
                        pad=self.pad[0],
                        ignore_border=self.ignore_border),
                shape[3])

    def get_output_for(self, input, **kwargs):
        input_shape = input.shape
        if self.dilation[0] > 1:
            # pad such that the time axis length is divisible by the dilation factor
            pad_w = (self.dilation[0] - input_shape[2] % self.dilation[0]) % self.dilation[0]
            input = T.concatenate((input, T.zeros((input_shape[0], input_shape[1], pad_w, input_shape[3]), input.dtype)), axis=2)
            # rearrange data to fold the time axis into the minibatch dimension
            input = input.reshape((input_shape[0], input_shape[1], -1, self.dilation[0], input_shape[3]))
            input = input.transpose(0, 3, 1, 2, 4)
            input = input.reshape((-1,) + tuple(input.shape[2:]))
        output = super(TimeDilatedMaxPool2DLayer, self).get_output_for(input, **kwargs)
        if self.dilation[0] > 1:
            # restore the time axis from the minibatch dimension
            output = output.reshape((input_shape[0], self.dilation[0]) + tuple(output.shape[1:]))
            output = output.transpose(0, 2, 3, 1, 4)
            output = output.reshape((input_shape[0], output.shape[1], -1, output.shape[4]))
            # remove the padding
            output = output[:, :, :output.shape[2] - pad_w]
        return output


def model_to_fcn(output_layers, allow_unlink=False):
    """
    Converts a Lasagne CNN model for fixed-size spectrogram excerpts into a
    fully-convolutional network that can handle spectrograms of arbitrary
    length (but at least the fixed length the original CNN was designed for),
    producing the same results as if applying it to every possible excerpt of
    the spectrogram in sequence.
    
    This is done by replacing convolutional and pooling layers with dilated
    versions if they appear after temporal max-pooling in the original model,
    and the first dense layer with a convolutional layer.
    
    If `allow_unlink` is False, the converted model will share all parameters
    with the original model. Otherwise, some parameters may be unshared for
    improved performance.
    """
    converted = {}
    dilations = {}
    for layer in lasagne.layers.get_all_layers(output_layers):
        if isinstance(layer, InputLayer):
            # Input layer: Just set third dimension to be of arbitrary size
            converted[layer] = InputLayer(
                    layer.shape[:2] + (None,) + layer.shape[3:],
                    layer.input_var)
            dilations[layer] = 1

        elif isinstance(layer, Conv2DLayer):
            # Conv2DLayer: Make dilated if needed
            kwargs = dict(
                    incoming=converted[layer.input_layer],
                    num_filters=layer.num_filters,
                    filter_size=layer.filter_size,
                    nonlinearity=layer.nonlinearity,
                    b=layer.b)
            dilation = dilations[layer.input_layer]
            if dilation == 1:
                converted[layer] = Conv2DLayer(W=layer.W, **kwargs)
            else:
                W = layer.W.get_value() if allow_unlink else layer.W
                converted[layer] = DilatedConv2DLayer(
                        W=W.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1],
                        dilation=(dilation, 1), **kwargs)
            dilations[layer] = dilation

        elif isinstance(layer, MaxPool2DLayer):
            # MaxPool2DLayer: Make dilated if needed, increase dilation factor
            kwargs = dict(
                incoming=converted[layer.input_layer],
                pool_size=layer.pool_size,
                stride=(1, layer.stride[1]))
            dilation = dilations[layer.input_layer]
            if dilation == 1:
                converted[layer] = MaxPool2DLayer(**kwargs)
            else:
                converted[layer] = TimeDilatedMaxPool2DLayer(
                        dilation=(dilation, 1), **kwargs)
            dilations[layer] = dilation * layer.stride[0]

        elif isinstance(layer, DenseLayer):
            # DenseLayer: Turn into Conv2DLayer/DilatedConv2DLayer if needed,
            # reset dilation factor
            dilation = dilations[layer.input_layer]
            if getattr(layer, 'num_leading_axes', 1) == -1:
                # special case: num_leading_axes=-1, no dilation needed
                blocklen = 1
            elif len(layer.input_shape) == 4:
                blocklen = np.prod(layer.input_shape[1:]) // layer.input_shape[1] // layer.input_shape[-1]
            else:
                blocklen = 1
            if (blocklen > 1) or (dilation > 1):
                W = layer.W.get_value() if allow_unlink else layer.W
                W = W.T.reshape((layer.num_units, layer.input_shape[1], blocklen, layer.input_shape[-1])).transpose(1, 0, 2, 3)
                converted[layer] = DilatedConv2DLayer(
                        converted[layer.input_layer],
                        num_filters=layer.num_units,
                        filter_size=(blocklen, layer.input_shape[-1]),
                        W=W, b=layer.b,
                        dilation=(dilation, 1),
                        nonlinearity=None)
                converted[layer] = lasagne.layers.DimshuffleLayer(
                        converted[layer], (0, 2, 1, 3))
                converted[layer] = lasagne.layers.ReshapeLayer(
                        converted[layer], (-1, [2], [3]))
                converted[layer] = lasagne.layers.FlattenLayer(
                        converted[layer])
                converted[layer] = lasagne.layers.NonlinearityLayer(
                        converted[layer], layer.nonlinearity)
                dilations[layer] = 1
            else:
                converted[layer] = DenseLayer(
                        converted[layer.input_layer],
                        num_units=layer.num_units, W=layer.W, b=layer.b,
                        nonlinearity=layer.nonlinearity,
                        num_leading_axes=layer.num_leading_axes)
            dilations[layer] = 1

        elif not isinstance(layer, MergeLayer):
            # all other layers: deepcopy the layer
            # - set up a memo dictionary so the cloned layer will be linked to
            #   the converted part of the network, not to a new clone of it
            memo = {id(layer.input_layer): converted[layer.input_layer]}
            # - in addition, share all parameters with the existing layer
            memo.update((id(p), p) for p in layer.params.keys())
            # - perform the copy
            clone = deepcopy(layer, memo)
            # update the input shape of the cloned layer
            clone.input_shape = converted[layer.input_layer].output_shape
            # use the cloned layer, keep the dilation factor
            converted[layer] = clone
            dilations[layer] = dilations[layer.input_layer]

        else:
            raise ValueError("don't know how to convert %r" % layer)

    # Return list of converted output layers, or single converted output layer
    try:
        return [converted[layer] for layer in output_layers]
    except TypeError:
        return converted[output_layers]

