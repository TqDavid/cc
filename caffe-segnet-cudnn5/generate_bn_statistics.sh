#!bin/sh
echo "------------------generating bn statistics is begin-----------------------------"
python generate_bn_statistics.py examples/segnet/segnet_train.prototxt examples/segnet/segnet_train/segnet_basic/seg_iter_15700.caffemodel models/inference  # compute BN statistics for SegNet
echo "------------------generating bn statistics is end-----------------------------"
