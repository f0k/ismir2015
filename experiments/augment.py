#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data augmentation routines.

Author: Jan Schlüter
"""

import sys
import os
import subprocess

import numpy as np
import scipy.ndimage


def grab_random_excerpts(spects, labels, batchsize, frames):
    """
    Exctracts random excerpts of `frames` frames from the spectrograms in
    `spects` paired with the label from `labels` associated with the central
    frame. Yields `batchsize` excerpts at a time. Draws without replacement
    until exhausted, then shuffles anew and continues.
    """
    # buffer to write each batch into
    batch_spects = np.empty((batchsize, frames, spects[0].shape[1]),
                            dtype=spects[0].dtype)
    batch_labels = np.empty((batchsize,) + labels[0].shape[1:],
                            dtype=labels[0].dtype)
    # array of all possible (spect_idx, frame_idx) combinations
    indices = np.vstack(np.vstack((np.ones(len(spect) - frames + 1,
                                           dtype=np.int) * spect_idx,
                                   np.arange(len(spect) - frames + 1,
                                             dtype=np.int))).T
                        for spect_idx, spect in enumerate(spects)
                        if len(spect) >= frames)
    # infinite loop
    while True:
        # shuffle all possible indices
        np.random.shuffle(indices)
        # draw without replacement until exhausted
        b = 0
        for spect_idx, frame_idx in indices:
            batch_spects[b] = spects[spect_idx][frame_idx:frame_idx + frames]
            batch_labels[b] = labels[spect_idx][frame_idx + frames//2]
            b += 1
            if b == batchsize:
                # copy the buffers to prevent changing returned data (not a
                # problem if it is consumed right away, but if it is collected)
                yield batch_spects.copy(), batch_labels.copy()
                b = 0


def apply_random_stretch_shift(batches, max_stretch, max_shift,
                               keep_frames, keep_bins, order=3,
                               prefiltered=False):
    """
    Apply random time stretching of up to +/- `max_stretch`, random pitch
    shifting of up to +/- `max_shift`, keeping the central `keep_frames` frames
    and the first `keep_bins` bins. For performance, the spline `order` can be
    reduced, and inputs can be `prefiltered` with scipy.ndimage.spline_filter.
    """
    for spects, labels in batches:
        outputs = np.empty((len(spects), keep_frames, keep_bins),
                           dtype=spects.dtype)
        randoms = (np.random.rand(len(spects), 2) - .5) * 2
        for spect, output, random in zip(spects, outputs, randoms):
            stretch = 1 + random[0] * max_stretch
            shift = 1 + random[1] * max_shift
            # We can do shifting/stretching and cropping in a single affine
            # transform (including setting the upper bands to zero if we shift
            # down the signal so far that we exceed the input's nyquist rate)
            scipy.ndimage.affine_transform(
                    spect, (1 / stretch, 1 / shift),
                    output_shape=(keep_frames, keep_bins), output=output,
                    offset=(.5 * (len(spect) * stretch - keep_frames), 0),
                    mode='constant', cval=0, order=order,
                    prefilter=not prefiltered)
        yield outputs, labels


def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq,
                          max_freq):
    """
    Creates a mel filterbank of `num_bands` triangular filters, with the first
    filter starting at `min_freq` and the last one stopping at `max_freq`.
    Returns the filterbank as a matrix suitable for a dot product against
    magnitude spectra created from samples at a sample rate of `sample_rate`
    with a window length of `frame_len` samples.
    """
    # prepare output matrix
    input_bins = (frame_len // 2) + 1
    filterbank = np.zeros((input_bins, num_bands))

    # mel-spaced peak frequencies
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    spacing = (max_mel - min_mel) / (num_bands + 1)
    peaks_mel = min_mel + np.arange(num_bands + 2) * spacing
    peaks_hz = 700 * (np.exp(peaks_mel / 1127) - 1)
    fft_freqs = np.linspace(0, sample_rate / 2., input_bins)
    peaks_bin = np.searchsorted(fft_freqs, peaks_hz)

    # fill output matrix with triangular filters
    for b, filt in enumerate(filterbank.T):
        # The triangle starts at the previous filter's peak (peaks_freq[b]),
        # has its maximum at peaks_freq[b+1] and ends at peaks_freq[b+2].
        left_hz, top_hz, right_hz = peaks_hz[b:b+3]  # b, b+1, b+2
        left_bin, top_bin, right_bin = peaks_bin[b:b+3]
        # Create triangular filter compatible to yaafe
        filt[left_bin:top_bin] = ((fft_freqs[left_bin:top_bin] - left_hz) /
                                  (top_bin - left_bin))
        filt[top_bin:right_bin] = ((right_hz - fft_freqs[top_bin:right_bin]) /
                                   (right_bin - top_bin))
        filt[left_bin:right_bin] /= filt[left_bin:right_bin].sum()

    return filterbank


def apply_filterbank(batches, filterbank):
    """
    Apply a filterbank to batches of spectrogram excerpts via a dot product.
    """
    for spects, labels in batches:
        # we reshape (batchsize, frames, bins) to (batchsize * frames, bins) so
        # we can transform all excerpts in a single dot product, then go back
        # to (batchsize, frames, filters)
        yield (np.dot(spects.reshape(-1, spects.shape[-1]), filterbank).reshape(
                (spects.shape[0], spects.shape[1], -1)), labels)


def apply_logarithm(batches, clip=1e-7):
    """
    Convert linear to logarithmic magnitudes, clipping magnitudes < `clip`.
    """
    for spects, labels in batches:
        yield np.log(np.maximum(spects, clip)), labels


def apply_random_filters(batches, filterbank, max_freq, max_db, min_std=5,
                         max_std=7):
    """
    Applies random filter responses to logarithmic-magnitude mel spectrograms.
    The filter response is a Gaussian of a standard deviation between `min_std`
    and `max_std` semitones, a mean between 150 Hz and `max_freq`, and a
    strength between -/+ `max_db` dezibel. Assumes the mel spectrograms have
    been transformed with `filterbank` and cover up to `max_freq` Hz.
    """
    for spects, labels in batches:
        batchsize, length, bands = spects.shape
        bins = len(filterbank)
        # sample means and std deviations on logarithmic pitch scale
        min_pitch = 12 * np.log2(150)
        max_pitch = 12 * np.log2(max_freq)
        mean = min_pitch + (np.random.rand(batchsize) *
                            (max_pitch - min_pitch))
        std = min_std + np.random.randn(batchsize) * (max_std - min_std)
        # convert means and std deviations to linear frequency scale
        std = 2**((mean + std) / 12) - 2**(mean / 12)
        mean = 2**(mean / 12)
        # convert means and std deviations to bins
        mean = mean * bins / max_freq
        std = std * bins / max_freq
        # sample strengths uniformly in dB
        strength = max_db * 2 * (np.random.rand(batchsize) - .5)
        # create Gaussians
        filt = (strength[:, np.newaxis] *
                np.exp(np.square((np.arange(bins) - mean[:, np.newaxis]) /
                                 std[:, np.newaxis]) * -.5))
        # transform from dB to factors
        filt = 10**(filt / 20.)
        # transform to mel scale
        filt = np.dot(filt.astype(spects.dtype), filterbank)
        # logarithmize
        filt = np.log(filt)
        # apply (it's a simple addition now, broadcasting over the second axis)
        yield spects + filt[:, np.newaxis, :], labels


def apply_znorm(batches, mean, istd):
    """
    Apply Z-scoring (subtract mean, multiply by inverse std deviation).
    """
    for spects, labels in batches:
        yield (spects - mean) * istd, labels


def generate_in_background(generators, num_cached=50, in_processes=False):
    """
    Runs generators in background threads or processes, caching up to
    `num_cached` items. Multiple generators are each run in a separate
    thread/process, and their items will be interleaved in unpredictable order.
    """
    if not in_processes:
        from Queue import Queue
        from threading import Thread as Background
        sentinel = object()  # guaranteed unique reference
    else:
        from multiprocessing import Queue
        from multiprocessing import Process as Background
        sentinel = None  # object() would be different between processes

    queue = Queue(maxsize=num_cached)

    # define producer (putting items into queue)
    def producer(generator, queue, sentinel, seed=None):
        if seed is not None:
            np.random.seed(seed)
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producers (in background threads or processes)
    active = 0
    for generator in generators:
        # when multiprocessing, ensure processes have different random seeds
        seed = np.random.randint(2**32-1) if in_processes and active else None
        bg = Background(target=producer,
                        args=(generator, queue, sentinel, seed))
        bg.daemon = True
        bg.start()
        active += 1
        if active > num_cached:
            raise ValueError("generate_in_background() got more generators "
                             "than cached items (%d). Make sure you supplied "
                             "a list or iterable of generators as the first "
                             "argument, not a generator." % num_cached)

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while True:
        if item is sentinel:
            active -= 1
            if not active:
                break
        else:
            yield item
        item = queue.get()

