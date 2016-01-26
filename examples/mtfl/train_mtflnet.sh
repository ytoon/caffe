#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/mtfl/solver.prototxt 
    # --weights=models/mtfl/caffe_mtflnet_train_iter_3700.caffemodel