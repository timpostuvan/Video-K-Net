#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
RESULTS_DIR=$4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_step.py $CONFIG $CHECKPOINT --launcher pytorch --show-dir $RESULTS_DIR ${@:5}