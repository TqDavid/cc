#!/usr/bin/python

import numpy as np
from skimage import io
import cv2

def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i  = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)
 
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_

def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
 
        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_

def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)
 
    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_

def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)
 
    sum_k_t_k = get_pixel_area(eval_segm)
    
    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_

'''
Auxiliary functions used during evaluation.
'''
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
###############now we do some eval.
# test image only 1 image
def eval_segm(preddir, gtdir):
	pred = io.imread(preddir, 1)
	gt = io.imread(gtdir, 1)
	pred = (pred ).astype(np.uint8)
	np.set_printoptions(threshold='nan')
	#print(pred[10:50,:])
	_, pred_th= cv2.threshold(pred, 0.0000000000000001, 1, cv2.THRESH_BINARY)
	#print(gt[10:50,:])
	gt = (gt).astype(np.uint8)
	_, gt_th= cv2.threshold(gt, 0.0000000000000001, 1, cv2.THRESH_BINARY)
	
	pixel_accu = pixel_accuracy(pred_th, gt_th)
	mean_accu = mean_accuracy(pred_th, gt_th)
	mean_iou = mean_IU(pred_th, gt_th)
	fw_iou = frequency_weighted_IU(pred_th, gt_th)
	print("pixel_accu is: ", pixel_accu)
	print("mean_accu is: ", mean_accu)
	print("mean_iou is: ",mean_iou)
	print("fw_iou is: ", fw_iou)
	return pixel_accu, mean_accu, mean_iou, fw_iou
# test batch image
def eval_batch(rootdir):
	res_sum = []
	pixel_accu = 0.0
	mean_accu = 0.0
	mean_iou = 0.0
	fw_iou = 0.0
	
	for i in range(16):
		preddir = rootdir + str(i)+"predicted.png"
		gtdir = rootdir + str(i) + "rgb_gt.png"
		print("===============%d==================", i)
		resperimage = eval_segm(preddir, gtdir)
		res_sum.append(resperimage)
	# compute avg eval metrics	
	print("==================avg eval seg=========================")
	len_res_sum = len(res_sum)
	for i in range(len_res_sum):
		pixel_accu += res_sum[i][0]
		mean_accu += res_sum[i][1]
		mean_iou += res_sum[i][2]
		fw_iou += res_sum[i][3]
	print("avg pixel_accu : ", pixel_accu / len_res_sum, "avg mean_accu : ", mean_accu / len_res_sum,\
	"avg mean_iou : ", mean_iou / len_res_sum, "avg fw_iou : ", fw_iou/len_res_sum)
	
# get the contours of huizibiao	
def get_contour(imagedir, preddir):
	#np.set_printoptions(threshold='nan')

	pred = io.imread(preddir, 1)
	print(pred.dtype)
	#print(pred[:,10:50])
	pred = (pred ).astype(np.uint8)
	#print(" ")
	image = io.imread(imagedir, 1) # because it is float64 
	print(image.dtype)
	print(image.shape)
	#print(image[:,10:50])
	image = (image* 255).astype(np.uint8)	
	#cv2.imwrite("image.png",image)
	_, pred_th= cv2.threshold(pred, 0.0000000000000001, 1, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(pred_th,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
	
	pred_contours = image
	
	for i in range(len(contours)):
		cv2.drawContours(pred_contours, contours[i], -1, (0, 255, 0), 1)
	
	return pred_contours
# batch test contours of huizibiao
def get_contour_batch(rootdir):
	for i in range(16):
		preddir = rootdir + str(i)+"predicted.png"
		imagedir = rootdir + str(i) + "image.png"
		print("=================================", i)
		cv2.imwrite(str(i)+"image_countours.png", get_contour(imagedir, preddir))
		
if __name__ == "__main__":
	'''
	# test only one image.
	preddir = "/data/dengtingqiang/caffe-segnet-cudnn/caffe-segnet-cudnn5/test_result/3_iter7700/2/predicted.png"
	gtdir = "/data/dengtingqiang/caffe-segnet-cudnn/caffe-segnet-cudnn5/test_result/3_iter7700/2/rgb_gt.png"
	eval_segm(preddir, gtdir)
	'''
	
	# test batch image
	#rootdir = "/data/dengtingqiang/caffe-segnet-cudnn/caffe-segnet-cudnn5/test_result/116/iter17w/"
	rootdir = "/data/dengtingqiang/caffe-segnet-cudnn/caffe-segnet-cudnn5/"
	#eval_batch(rootdir)
	
	#draw contours on the one  image
	#preddir = "/data/dengtingqiang/caffe-segnet-cudnn/caffe-segnet-cudnn5/0predicted.png"
	#imagedir = "/data/dengtingqiang/caffe-segnet-cudnn/caffe-segnet-cudnn5/0image.png"
	#get_contour(imagedir, preddir)
	
	#test batch
	get_contour_batch(rootdir)