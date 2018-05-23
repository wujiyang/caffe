#!/usr/bin/env sh

TOOLS=./build/tools
GLOG_logtostderr=0 GLOG_log_dir=face_example/center-loss/log/ \
$TOOLS/caffe train \
  --solver=face_example/center-loss/face_solver.prototxt \
  --weights=face_example/center-loss/Snap_Vggface2/vggface2_iter_18216.caffemodel \
  --gpu=0
