#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluates singing voice predictions against ground truth.

For usage information, call with --help.

Author: Jan Schlüter
"""

from __future__ import print_function

import sys
import os
import io
from argparse import ArgumentParser

import numpy as np
import scipy.ndimage.filters

from progress import progress
from labels import create_aligned_targets

def opts_parser():
    descr = "Evaluates singing voice predictions against ground truth."
    parser = ArgumentParser(description=descr)
    parser.add_argument('infile', nargs='+', metavar='INFILE',
            type=str,
            help='File to load the prediction curves from (.npz format). '
                 'If given multiple times, prediction curves will be '
                 'averaged.')
    parser.add_argument('--dataset',
            type=str, default='jamendo',
            help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--threshold',
            type=float, default=None,
            help='If given, use this threshold instead of optimizing it on '
                 'the validation set.')
    parser.add_argument('--auroc',
            action='store_true', default=False,
            help='If given, compute AUROC on the test set.')
    parser.add_argument('--smooth-width', metavar='WIDTH',
            type=int, default=56,
            help='Apply temporal smoothing over WIDTH frames (default: '
                 '(default)s)')
    parser.add_argument('--smooth-method', metavar='METHOD',
            type=str, choices=('median', 'mean'), default='median',
            help='Temporal smoothing method (default: %(default)s)')
    return parser

def load_labels(filelist, predictions, fps, datadir):
    labels = []
    for fn in filelist:
        ffn = os.path.join(datadir, 'labels', fn.rsplit('.', 1)[0] + '.lab')
        with io.open(ffn) as f:
            segments = [l.rstrip().split() for l in f if l.rstrip()]
        segments = [(float(start), float(end), label == 'sing')
                    for start, end, label in segments]
        timestamps = np.arange(len(predictions[fn])) / float(fps)
        labels.append(create_aligned_targets(segments, timestamps, np.bool))
    return labels

def evaluate(predictions, truth, threshold=None, smoothen=56,
        smooth_fn='median', collapse_files=True, compute_auroc=False):
    assert len(predictions) == len(truth)

    # preprocess network outputs
    if smoothen:
        if smooth_fn == 'median':
            smooth_fn = lambda p: scipy.ndimage.filters.median_filter(
                    p, smoothen, mode='nearest')
        elif smooth_fn == 'mean':
            smooth_fn = lambda p: scipy.ndimage.filters.uniform_filter(
                    p, smoothen, mode='nearest')
        predictions = [smooth_fn(pred)
                if len(pred) > 1 else pred
                for pred in predictions]

    # evaluate
    if threshold is None or compute_auroc:
        thresholds = np.hstack(([1e-5, 1e-4, 1e-3],
                np.arange(1, 100) / 100.0,
                [1-1e-3, 1-1e-4, 1-1e-5]))
        if (threshold is not None) and (threshold not in thresholds):
            thresholds = np.insert(thresholds,
                                   np.searchsorted(thresholds, threshold),
                                   threshold)
    else:
        thresholds = np.array([threshold])
    tp = np.zeros((len(predictions), len(thresholds)), dtype=np.int)
    fp = np.zeros((len(predictions), len(thresholds)), dtype=np.int)
    tn = np.zeros((len(predictions), len(thresholds)), dtype=np.int)
    fn = np.zeros((len(predictions), len(thresholds)), dtype=np.int)
    for idx in range(len(predictions)):
        preds = predictions[idx] > thresholds[:, np.newaxis]
        target = truth[idx]
        nopreds = ~preds
        correct = (preds == target)
        incorrect = ~correct
        tp[idx] = (correct * preds).sum(axis=1)
        fp[idx] = (incorrect * preds).sum(axis=1)
        tn[idx] = (correct * nopreds).sum(axis=1)
        fn[idx] = (incorrect * nopreds).sum(axis=1)
    if collapse_files:
        # treat all files as a single long file, rather than
        # averaging over file-wise results afterwards
        tp = tp.sum(axis=0, keepdims=True)
        fp = fp.sum(axis=0, keepdims=True)
        tn = tn.sum(axis=0, keepdims=True)
        fn = fn.sum(axis=0, keepdims=True)
    def savediv(a, b):
        b[a == 0] = 1
        return a / b.astype(np.float64)
    accuracy    = (tp + tn) / (tp + fp + tn + fn).astype(np.float64)
    precision   = savediv(tp, tp + fp)
    recall      = savediv(tp, tp + fn)
    specificity = savediv(tn, tn + fp)
    fscore      = savediv(2 * precision * recall, precision + recall)
    if compute_auroc:
        if not collapse_files:
            raise NotImplementedError("Sorry, we didn't need this so far.")
        rec = np.concatenate(([0], recall.ravel()[::-1], [1]))
        one_minus_spec = np.concatenate(([0], 1 - specificity.ravel()[::-1], [1]))
        auroc = np.trapz(rec, one_minus_spec)
    else:
        auroc = np.nan
    if threshold is None:
        best = np.argmax(accuracy.mean(axis=0))
    else:
        best = np.searchsorted(thresholds, threshold)
    return thresholds[best], {
            'accuracy': accuracy[:, best],
            'precision': precision[:, best],
            'recall': recall[:, best],
            'specificity': specificity[:, best],
            'fscore': fscore[:, best],
            'auroc': auroc,
            }

def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    infiles = options.infile
    fps = 70
    
    # load network predictions
    preds = np.load(infiles[0])
    if len(infiles) > 1:
        preds = {fn: preds[fn] / len(infiles) for fn in preds.files}
        for infile in infiles[1:]:
            morepreds = np.load(infile)
            for fn in preds:
                preds[fn] += morepreds[fn] / len(infiles)
        del morepreds

    # load file lists
    datadir = os.path.join(os.path.dirname(__file__),
                           os.path.pardir, 'datasets', options.dataset)
    with io.open(os.path.join(datadir, 'filelists', 'valid')) as f:
        filelist_valid = [l.rstrip() for l in f if l.rstrip()]
    with io.open(os.path.join(datadir, 'filelists', 'test')) as f:
        filelist_test = [l.rstrip() for l in f if l.rstrip()]

    # optimize threshold on validation set if needed
    if options.threshold is None:
        options.threshold, _ = evaluate(
                [preds[fn].ravel() for fn in filelist_valid],
                load_labels(filelist_valid, preds, fps, datadir),
                smoothen=options.smooth_width, smooth_fn=options.smooth_method)
    
    # evaluate on test set
    threshold, results = evaluate(
                [preds[fn].ravel() for fn in filelist_test],
                load_labels(filelist_test, preds, fps, datadir),
                smoothen=options.smooth_width, smooth_fn=options.smooth_method,
                threshold=options.threshold, compute_auroc=options.auroc)

    # print results
    if sys.stdout.isatty():
        BOLD = '\033[1m'
        UNBOLD = '\033[0m'
    else:
        BOLD = UNBOLD = ''
    print("thr: %.2f, prec: %.3f, rec: %.3f, spec: %.3f, f1: %.3f, err: %s%.3f%s" % (
            threshold, results['precision'].mean(), results['recall'].mean(),
            results['specificity'].mean(), results['fscore'].mean(),
            BOLD, 1 - results['accuracy'].mean(), UNBOLD))
    if options.auroc:
        print("auroc: %.3f" % results['auroc'])

if __name__=="__main__":
    main()

