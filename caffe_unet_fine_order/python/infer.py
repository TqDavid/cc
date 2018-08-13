import numpy as np
from PIL import Image
import cv2

import caffe
#import vis
def rgb2gray(rgb):    
    return np.dot(rgb[...,:3], [1.0, 0.0, 0.0])

# the demo image is "2007_000129" from PASCAL VOC

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#im = Image.open('huaweishouji_20170720_6_1_0_1.jpg')#huaweishouji_20170720_6_1_0_1.jpg
im = Image.open('huaweishouji_20170720_6_1_0_1.jpg')
im.resize((572, 572),Image.ANTIALIAS)
print(im)
#im = rgb2gray(im)
im.convert('L')
in_ = np.array(im, dtype=np.float32)
print(in_)
#in_ = in_[:,:,::-1]
#in_ = in_[:,:,::-1]
#in_ -= np.array([128])
#np.array((104.00698793,116.66876762,122.67891434))#

in_ = in_/255.0
in_ -= 9.5
#in_ = in_.transpose((2,0,1))

#in_.reshape(1,572,572)
print(in_)
# load net
net = caffe.Net('deploy.prototxt', 'unet_iter_2000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
print(in_.shape)
net.blobs['data_New'].reshape(1, *in_.shape)
net.blobs['data_New'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

# visualize segmentation in PASCAL VOC colors
#voc_palette = vis.make_palette(21)
#out_im = Image.fromarray(vis.color_seg(out, voc_palette))
out_im.save('demo/out.png')
#masked_im = Image.fromarray(vis.vis_seg(im, out, voc_palette))
masked_im.save('demo/visualization.jpg')
