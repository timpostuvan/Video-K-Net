#!/usr/bin/env bash

RESULTS_DIR=$1

# Colorize images
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/colorize_img.py $RESULTS_DIR

# Create GIFs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/convert_colored_img_to_gif.py $RESULTS_DIR