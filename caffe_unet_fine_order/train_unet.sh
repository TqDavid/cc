#!/bin/sh
echo "----------------Begining to train U-Net model----------------------"
echo " "

pwd
#export PYTHONPATH=/data/dengtingqiang/caffe_unet_fine_order:$PYTHONPATH
export PYTHONPATH=/data/dengtingqiang/caffe_unet_fine_order/python:$PYTHONPATH

#export PYTHONPATH=. 
SOLVER=examples/train_unet/unet_solver.prototxt
#WEIGHTS=premodel/VGG_ILSVRC_16_layers_fc_reduced.caffemodel
build/tools/caffe train --gpu 4 --solver=$SOLVER  2>&1 | tee $LOG #--timing
#--weights=$WEIGHTS

echo "-------------The trained U-Net model is end-------------------------"
echo " "