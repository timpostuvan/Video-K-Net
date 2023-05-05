#!/usr/bin/env bash

WORK_DIR=$1
CONFIG=$2
GPUS=$3
PORT=${PORT:-$((29500 + $RANDOM % 29))}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch  --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4}
