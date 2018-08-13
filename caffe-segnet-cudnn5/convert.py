import cv2
import os
import numpy as np
convertdir = '/data/dengtingqiang/caffe-segnet-cudnn/caffe-segnet-cudnn5/'
convertdir = '/data/dengtingqiang/caffe-segnet-cudnn/caffe-segnet-cudnn5/'
imgdir = "/data/dengtingqiang/caffe-segnet-cudnn/caffe-segnet-cudnn5/data/data_prepare/Img/"
maskdir = '/data/dengtingqiang/caffe-segnet-cudnn/caffe-segnet-cudnn5/data/data_prepare/Mask/'
imgtxt = "/data/dengtingqiang/caffe-segnet-cudnn/caffe-segnet-cudnn5/data/data_prepare/test/accu_pos_list.txt"
lines = open(imgtxt, 'r').readlines()
print(lines)
#extension = ".png"
def load_mask(idx):
    outimg = np.empty((2,160,160))
    #lines = lines[0][:-2]
    imname = maskdir + lines[idx] #+ extension :-1
    print(lines[idx]) # [:-1]
    print(imname)
    #imname = imname[:-2]
    #print 'load mask %s' %imname
    im = cv2.imread(imname)
    print(im.shape)
    cv2.imwrite("original.png", im,)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("original_gray.png", im,)
    #im = cv2.resize(im,(160,160))
    ret, img = cv2.threshold(im, 0.1, 255, cv2.THRESH_BINARY)
    cv2.imwrite('3_'+lines[idx] + ".png", img) #[:-1]
    cv2.imwrite('ret_0.png', ret)
    #ret, back = cv2.threshold(im, 0.5, 1.0, cv2.THRESH_BINARY_INV)
    #outimg[0, ...] = img;
    #outimg[1, ...] = back;
    #outimg.astype(np.uint8)
    return img[np.newaxis, :]



def load_mask_test(idx):
    outimg = np.empty((2,160,160))
    #lines = lines[0][:-2]
    imname = maskdir + 'huaweishouji_20170720_1_1_0_2.png.png' #+ extension :-1
    #print(lines[idx]) # [:-1]
    print(imname)
    #imname = imname[:-2]
    #print 'load mask %s' %imname
    im = cv2.imread(imname)
    print(im.shape)
    cv2.imwrite('huaweishouji_20170720_1_1_0_2_test.png', im,)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("original_gray.png", im,)
    #im = cv2.resize(im,(160,160))
    ret, img = cv2.threshold(im, 0.5, 1, cv2.THRESH_BINARY)
    cv2.imwrite('mask_huaweishouji_20170720_1_1_0_2_test.png', img) #[:-1]
    cv2.imwrite('ret_huaweishouji_20170720_1_1_0_2_test.png' , ret)
    #ret, back = cv2.threshold(im, 0.5, 1.0, cv2.THRESH_BINARY_INV)
    #outimg[0, ...] = img;
    #outimg[1, ...] = back;
    #outimg.astype(np.uint8)
    return img[np.newaxis, :] 

def resize_image(idx):
    imname = imgdir + lines[idx][:-1]
    print(imname)
    im = cv2.imread(imname, 0)
    im = cv2.resize(im, (480, 360))
    im_RGB = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    print(im.shape)
    print(im_RGB.shape)
    print(im)
    print("============================")
    print(im_RGB[:,:,0])
    print(im_RGB[:,:,1])
    print(im_RGB[:,:,2])
    print(im.dtype)
    print(im_RGB.dtype)
    cv2.imwrite('1_' + lines[idx][:-1], im_RGB)

from skimage import io
def check_channel(idx):
    imname = maskdir + "huaweishouji_20170720_11_1_0_3.png"
    #imname = '/data/dengtingqiang/caffe_unet_fine_order/data/unet/huaweishouji_20170720/Mask/huaweishouji_20170720_11_1_0_0.jpg'
    np.set_printoptions(threshold='nan')
    
    print(imname)
    im = io.imread(imname,1)
    #print(im[30:50][:])
    print("-----------------------------------")
    print(im.shape)
    img = (im * 10).astype(np.uint8)
    #print(img[30:50][:] )
    _, im_th= cv2.threshold(img, 0.01, 1, cv2.THRESH_BINARY)
    #print(im_th*100)
    cv2.imwrite('3.png', im_th)
	#print(im_th * 100)
    #cv2.imwrite('da_0.png',im_th* 100)
    
    
    
    #print(im[:,:,0])
    #print(im[:,:,1])
    #print(im[:,:,2])
    print(im.dtype)
    print(img.dtype)
    print(im_th.dtype)
	
def check_channel_convert(idx):
    imname = convertdir + "rgb.png"
    #imname = '/data/dengtingqiang/caffe_unet_fine_order/data/unet/huaweishouji_20170720/Mask/huaweishouji_20170720_11_1_0_0.jpg'
    np.set_printoptions(threshold='nan')
    
    print(imname)
    im = io.imread(imname,1)
    print(im[30:50][:])
    print("-----------------------------------")
    print(im.shape)
    img = (im * 50000).astype(np.uint8)
    print(img[30:50][:] )
    #_, im_th= cv2.threshold(img, 0.01, 1, cv2.THRESH_BINARY)
    #print(im_th*100)
    #cv2.imwrite('3.png', im_th)
	#print(im_th * 100)
    #cv2.imwrite('da_0.png',im_th* 100)
    
    cv2.imwrite("rgb_int8.png", img )
    
    
    #print(im[:,:,0])
    #print(im[:,:,1])
    #print(im[:,:,2])
    print(im.dtype)
    print(img.dtype)
    #print(im_th.dtype)
	 
if __name__ == "__main__":
    #load_mask(2)
    #load_mask_test(2)
	#resize_image(3)
	#check_channel(0) # in inet project
	check_channel_convert(0)
    