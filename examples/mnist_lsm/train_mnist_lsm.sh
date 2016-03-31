#!/usr/bin/env sh

./build/tools/caffe train \
	--solver=/home/ytoon/PycharmProjects/LSM/models/mnist_lsm/mnist_lsm_finetune_solver.prototxt \
	--weights=/home/ytoon/PycharmProjects/LSM/output/default/mnist_lsm_final.caffemodel