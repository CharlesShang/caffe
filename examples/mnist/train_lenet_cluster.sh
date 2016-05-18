#!/usr/bin/env sh

# GLOG_logtostderr=1 ./build/tools/caffe train --solver=examples/mnist/lenet_solver_cluster.prototxt \
# -gpu 0 2>&1 | tee examples/mnist/lenet_solver_cluster.log

GLOG_logtostderr=1 ./build/tools/caffe train --solver=examples/mnist/lenet_solver_cluster.prototxt \
--weights=examples/mnist/lenet_iter_5000.caffemodel \
-gpu 0 2>&1 | tee examples/mnist/lenet_solver_cluster.log