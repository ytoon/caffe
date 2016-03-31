#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/media/ytoon/Elements/lsmnet
DATA=/media/ytoon/Elements/lsmnet
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/lsm_train_lmdb \
  $DATA/lsmnet_mean.binaryproto

echo "Done."
