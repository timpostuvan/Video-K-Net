#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
RESULTS_DIR=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/inference.py $CONFIG $CHECKPOINT --show-dir $RESULTS_DIR ${@:5}