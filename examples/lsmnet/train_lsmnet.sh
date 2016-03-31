#!/usr/bin/env sh

./build/tools/caffe train \
	--solver=models/lsmnet/kitti_solver.prototxt \
	# --weights=models/lsmnet/caffe_lsmnet_train_iter_1004.caffemodel