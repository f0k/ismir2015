#!/bin/bash

# Evaluates all available predictions in sets of five repetitions.
# If an argument is given ($1), will only evaluate $1*{1..5}.pred.pkl.

pattern="${1:-*/}"
for fn in $pattern*1.pred.pkl; do
	basefile="${fn%1.pred.pkl}"
	echo "$basefile"
	for r in {1..5}; do
		predfile="$basefile$r.pred.pkl"
		if [ -f "$predfile" ]; then
			./eval.py "$predfile" || exit
		fi
	done
done
