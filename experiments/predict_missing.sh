#!/bin/bash

# Computes network predictions for all trained networks that do not have their
# predictions computed yet. If all predictions are available, nothing happens.

for fn in */*.npz.vars; do
	modelfile="${fn%.vars}"
	predfile="${modelfile%npz}pred.pkl"
	[ ! -f "$predfile" ] && echo ./predict.py "$modelfile" "$predfile" --cache=cache
done
