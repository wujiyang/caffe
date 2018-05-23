#!/bin/bash

TOOLS=./build/tools
GLOG_logtostderr=0 GLOG_log_dir=face_example/sphereface/ResNet50_SphereFace/log \
$TOOLS/caffe train \
 --solver=face_example/sphereface/ResNet50_SphereFace/sphereface_solver.prototxt \
 --weights=face_example/sphereface/ResNet50_SphereFace/snapshot/single/Res50_sphere_single_iter_80000.caffemodel
