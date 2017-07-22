#!/bin/bash

# Runs all experiments listed at the bottom. Each experiment consists of a
# given number of repetitions using a particular base name for the weights and
# predictions files. Each single repetition checks if it was already run or is
# currently being run, creates a lockfile, trains the network, computes the
# predictions, and removes the lockfile. To distribute runs between multiple
# GPUs, just run this script multiple times with different THEANO_FLAGS.

train_if_free() {
	modelfile="$1"
	echo "$modelfile"
	if [ ! -f "$modelfile" ] && [ ! -f "$modelfile.lock" ]; then
		echo "$HOSTNAME: $THEANO_FLAGS" > "$modelfile.lock"
		OMP_NUM_THREADS=1 ./train.py "$modelfile" --augment --cache=cache --load-spectra=memory --validate --save-errors "${@:2}"
		./predict.py "$modelfile" "${modelfile%.npz}.pred.pkl" --cache=cache
		rm "$modelfile.lock"
	fi
}

train() {
	repeats="$1"
	name="$2"
	for (( r=1; r<=$repeats; r++ )); do
		train_if_free "$name"$r.npz "${@:3}"
	done
}


mkdir -p spectlearn

# starting point: nothing is learned
train 5 spectlearn/allfixed_

# learned magnitude transformations
train 5 spectlearn/maglearn_log1p0_ --var magscale=log1p_learn --var first_params_log=50
train 5 spectlearn/maglearn_log1p0_boost10_ --var magscale=log1p_learn --var first_params_log=50 --var first_params_eta_scale=10
train 5 spectlearn/maglearn_log1p0_boost50_ --var magscale=log1p_learn --var first_params_log=50 --var first_params_eta_scale=50
train 5 spectlearn/maglearn_log1p7_boost10_ --var magscale=log1p_learn7 --var first_params_log=50 --var first_params_eta_scale=10
train 5 spectlearn/maglearn_log1p7_boost50_ --var magscale=log1p_learn7 --var first_params_log=50 --var first_params_eta_scale=50
train 5 spectlearn/maglearn_pow_ --var magscale=pow_learn --var first_params_log=50
train 5 spectlearn/maglearn_pow_boost10_ --var magscale=pow_learn --var first_params_log=50 --var first_params_eta_scale=10
train 5 spectlearn/maglearn_pow_boost50_ --var magscale=pow_learn --var first_params_log=50 --var first_params_eta_scale=50

# starting point 2: nothing is learned, but using the melbank layer
# (which gives a slightly different filterbank, but should perform the same)
train 5 spectlearn/allfixed2_ --var filterbank=mel_learn --var first_params_eta_scale=0

# learned mel filterbanks
train 5 spectlearn/mellearn_ --var filterbank=mel_learn --var first_params_log=50
train 5 spectlearn/mellearn_boost10_ --var filterbank=mel_learn --var first_params_log=50 --var first_params_eta_scale=10
train 5 spectlearn/mellearn_boost50_ --var filterbank=mel_learn --var first_params_log=50 --var first_params_eta_scale=50


# dropout variants
mkdir -p dropout
train 5 dropout/channels_ --var arch.convdrop=channels
train 5 dropout/bands_ --var arch.convdrop=bands
train 5 dropout/independent_ --var arch.convdrop=independent

