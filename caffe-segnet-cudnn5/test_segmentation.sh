#!bin/sh
echo "-------------------test segmentation is begin---------------------"
python test_segmentation.py --model models/inference/segmentation_inference.prototxt --weights models/inference/test_weights_15750.caffemodel --iter 26 #12250 #15750  # Test SegNet
echo "-------------------test segmentation is end---------------------"
