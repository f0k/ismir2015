#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Computes predictions with a neural network trained for singing voice detection.

For usage information, call with --help.

Author: Jan Schlüter
"""

from __future__ import print_function

import sys
import os
import io
from argparse import ArgumentParser

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX
import lasagne

from progress import progress
from simplecache import cached
import audio
import model
import augment
import config


def opts_parser():
    descr = ("Computes predictions with a neural network trained for singing "
             "voice detection.")
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to load the learned weights from (.npz format)')
    parser.add_argument('outfile', metavar='OUTFILE',
            type=str,
            help='File to save the prediction curves to (.npz format)')
    parser.add_argument('--dataset',
            type=str, default='jamendo',
            help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--pitchshift', metavar='PERCENT',
            type=float, default=0.0,
            help='Perform test-time pitch-shifting of given amount and '
                 'direction in percent (e.g., -10 shifts down by 10%%).')
    parser.add_argument('--mem-use',
            type=str, choices=('high', 'mid', 'low'), default='mid',
            help='How much main memory to use. More memory allows a faster '
                 'implementation, applying the network as a fully-'
                 'convolutional net to longer excerpts or the full files. '
                 '(default: %(default)s)')
    parser.add_argument('--cache-spectra', metavar='DIR',
            type=str, default=None,
            help='Store spectra in the given directory (disabled by default)')
    parser.add_argument('--plot',
            action='store_true', default=False,
            help='If given, plot each spectrogram with predictions on screen.')
    parser.add_argument('--vars', metavar='FILE',
            action='append', type=str,
            default=[os.path.join(os.path.dirname(__file__), 'defaults.vars')],
            help='Reads configuration variables from a FILE of KEY=VALUE '
                 'lines. Can be given multiple times, settings from later '
                 'files overriding earlier ones. Will read defaults.vars, '
                 'then files given here.')
    parser.add_argument('--var', metavar='KEY=VALUE',
            action='append', type=str,
            help='Set the configuration variable KEY to VALUE. Overrides '
                 'settings from --vars options. Can be given multiple times.')
    return parser

def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile
    outfile = options.outfile

    # read configuration files and immediate settings
    cfg = {}
    if os.path.exists(modelfile + '.vars'):
        options.vars.insert(1, modelfile + '.vars')
    for fn in options.vars:
        cfg.update(config.parse_config_file(fn))
    cfg.update(config.parse_variable_assignments(options.var))

    # read some settings into local variables
    sample_rate = cfg['sample_rate']
    frame_len = cfg['frame_len']
    fps = cfg['fps']
    mel_bands = cfg['mel_bands']
    mel_min = cfg['mel_min']
    mel_max = cfg['mel_max']
    blocklen = cfg['blocklen']
    batchsize = cfg['batchsize']
    
    bin_nyquist = frame_len // 2 + 1
    bin_mel_max = bin_nyquist * 2 * mel_max // sample_rate

    # prepare dataset
    print("Preparing data reading...")
    datadir = os.path.join(os.path.dirname(__file__),
                           os.path.pardir, 'datasets', options.dataset)

    # - load filelist
    with io.open(os.path.join(datadir, 'filelists', 'valid')) as f:
        filelist = [l.rstrip() for l in f if l.rstrip()]
    with io.open(os.path.join(datadir, 'filelists', 'test')) as f:
        filelist += [l.rstrip() for l in f if l.rstrip()]

    # - create generator for spectra
    spects = (cached(options.cache_spectra and
                     os.path.join(options.cache_spectra, fn + '.npy'),
                     audio.extract_spect,
                     os.path.join(datadir, 'audio', fn),
                     sample_rate, frame_len, fps)
              for fn in filelist)

    # - pitch-shift if needed
    if options.pitchshift:
        import scipy.ndimage
        spline_order = 2
        spects = (scipy.ndimage.affine_transform(
                    spect, (1, 1 / (1 + options.pitchshift / 100.)),
                    output_shape=(len(spect), mel_max),
                    order=spline_order)
                  for spect in spects)

    # - define generator for cropped spectra
    spects = (spect[:, :bin_mel_max] for spect in spects)

    # - define generator for silence-padding
    pad = np.zeros((blocklen // 2, bin_mel_max), dtype=floatX)
    spects = (np.concatenate((pad, spect, pad), axis=0) for spect in spects)

    # - we start the generator in a background thread (not required)
    spects = augment.generate_in_background([spects], num_cached=1)


    print("Preparing prediction function...")
    # instantiate neural network
    input_var = T.tensor3('input')
    inputs = input_var.dimshuffle(0, 'x', 1, 2)  # insert "channels" dimension
    network = model.architecture(inputs, (None, 1, blocklen, bin_mel_max), cfg)

    # load saved weights
    with np.load(modelfile) as f:
        lasagne.layers.set_all_param_values(
                network, [f['param%d' % i] for i in range(len(f.files))])

    # performant way: convert to fully-convolutional network
    if not options.mem_use == 'low':
        import model_to_fcn
        network = model_to_fcn.model_to_fcn(network, allow_unlink=True)

    # create output expression
    outputs = lasagne.layers.get_output(network, deterministic=True)

    # prepare and compile prediction function
    print("Compiling prediction function...")
    test_fn = theano.function([input_var], outputs)

    # run prediction loop
    print("Predicting:")
    predictions = []
    for spect in progress(spects, total=len(filelist), desc='File '):
        if options.mem_use == 'high':
            # fastest way: pass full spectrogram through network at once
            preds = test_fn(spect[np.newaxis])  # insert batch dimension
        elif options.mem_use == 'mid':
            # performant way: pass spectrogram in equal chunks of up to one
            # minute, taking care to overlap by `blocklen // 2` frames and to
            # not pass a chunk shorter than `blocklen` frames
            chunks = np.ceil(len(spect) / (fps * 60.))
            hopsize = int(np.ceil(len(spect) / chunks))
            chunksize = hopsize + blocklen - 1
            preds = np.vstack(test_fn(spect[np.newaxis, pos:pos + chunksize])
                              for pos in range(0, len(spect), hopsize))
        else:
            # naive way: pass excerpts of the size used during training
            # - view spectrogram memory as a 3-tensor of overlapping excerpts
            num_excerpts = len(spect) - blocklen + 1
            excerpts = np.lib.stride_tricks.as_strided(
                    spect, shape=(num_excerpts, blocklen, spect.shape[1]),
                    strides=(spect.strides[0], spect.strides[0], spect.strides[1]))
            # - pass mini-batches through the network and concatenate results
            preds = np.vstack(test_fn(excerpts[pos:pos + batchsize])
                              for pos in range(0, num_excerpts, batchsize))
        predictions.append(preds)
        if options.plot:
            if spect.ndim == 3:
                spect = spect[0]  # remove channel axis
            spect = spect[blocklen//2:-blocklen//2]  # remove zero padding
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.imshow(spect.T[::-1], vmin=-3, cmap='hot', aspect='auto',
                       interpolation='nearest')
            ax2.plot(preds)
            ax2.set_ylim(0, 1.1)
            plt.show()

    # save predictions
    print("Saving predictions")
    np.savez(outfile, **{fn: pred for fn, pred in zip(filelist, predictions)})

if __name__=="__main__":
    main()

