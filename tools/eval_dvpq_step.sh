#!/usr/bin/env bash

RESULTS_DIR=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/eval_dvpq_step.py $RESULTS_DIR