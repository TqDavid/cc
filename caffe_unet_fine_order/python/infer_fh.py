caffe_root = '../'
import sys
sys.path.insert(0, caffe_root + 'python')

import numpy  as np
from PIL import Image as PILImage
import cv2
import os
import glob as gb
import caffe

def rgb2gray(rgb):    
    return np.dot(rgb[...,:3], [1.0, 0.0, 0.0])

MODEL_FILE = 'deploy.prototxt'
PRETRAINED = 'unet_iter_228766.caffemodel'
IMAGE_FILE = 'huaweishouji_20170720_6_1_0_1.jpg'

net = caffe.Net(MODEL_FILE,PRETRAINED,caffe.TEST)
caffe.set_mode_gpu()
caffe.set_device(0)

transformer = caffe.io.Transformer({'data': net.blobs['img'].data.shape})
#transformer.set_transpose('data', (2,0,1))
mean_val = np.array([128])
transformer.set_mean('data', mean_val)
transformer.set_raw_scale('data', 255) 
#transformer.set_channel_swap('data', (2,1,0))
net.blobs['img'].reshape(1,1,572,572)
print IMAGE_FILE
im  = caffe.io.load_image(IMAGE_FILE)
im = cv2.resize(im,(572,572),interpolation=cv2.INTER_CUBIC)
gray = rgb2gray(im)
gray = gray.reshape(572,572,1)
print(gray.shape)
#print(transformer.preprocess('data',gray))
net.blobs['img'].data[...] = transformer.preprocess('data',gray).transpose((2, 0, 1)) / 255
	
net.forward()
out = net.blobs['crop5'].data[0].argmax(axis=0)
print("out is :", out.shape)
print(out)

'''
img_path = gb.glob("pic_4096_4096/*.jpg") 
for IMAGE_FILE in img_path:
	print IMAGE_FILE
	im  = caffe.io.load_image(IMAGE_FILE)
	img = cv2.resize(im,(572,572),interpolation=cv2.INTER_CUBIC)
	gray = rgb2gray(im)
	
	net.blobs['data/New'].data[...] = transformer.preprocess('data',gray)
	
	out = net.forward()
	#print out
	num = out['loss'].shape[0]
	
	img = cv2.imread(IMAGE_FILE)
	
	for i in range(num):
		out_data = out['loss'][i]
		pts = []
		for j in range(4):
			pt = [out_data[2*j+1]*4096, out_data[2*j+2]*4096]
			pts.append(pt)
		cv2.line(img, (int(pts[0][0]),int(pts[0][1])), (int(pts[1][0]),int(pts[1][1])), (0, 255, 255),3)
		cv2.line(img, (int(pts[1][0]),int(pts[1][1])), (int(pts[2][0]),int(pts[2][1])), (0, 255, 255),3)
		cv2.line(img, (int(pts[2][0]),int(pts[2][1])), (int(pts[3][0]),int(pts[3][1])), (0, 255, 255),3)
		cv2.line(img, (int(pts[3][0]),int(pts[3][1])), (int(pts[0][0]),int(pts[0][1])), (0, 255, 255),3)
	
	cv2.imwrite('result/' + IMAGE_FILE.split('/')[1],img)
'''