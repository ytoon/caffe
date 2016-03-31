#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

EXAMPLE=/media/ytoon/Elements/lsmnet
DATA=/home/ytoon/PycharmProjects/LSM/data
TOOLS=build/tools

TRAIN_DATA_ROOT=/media/ytoon/Elements/dataset/sequences/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=370
  RESIZE_WIDTH=1226
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --shuffle \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --check_size \
    $TRAIN_DATA_ROOT \
    $DATA/kitti_image_list.txt \
    $EXAMPLE/lsm_train_lmdb

echo "Done."
