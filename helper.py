import os
import copy
import glob
import queue
from urllib.request import urlopen
import argparse
import numpy as np
from tqdm import tqdm
from pdb import set_trace

import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms

import utils
import vision_transformer as vits
from pdb import set_trace


def read_frame_list(video_dir):
	frame_list = [img for img in glob.glob(os.path.join(video_dir,"*.jpg"))]
	frame_list = sorted(frame_list)
	return frame_list


def get_seg_list(frame_list):
	seg_list = [frame.replace("JPEGImages", "Annotations").replace("jpg", "png") for frame in frame_list]
	return seg_list


def read_frame(frame_dir, scale_size=[480]):
	"""
	read a single frame & preprocess
	"""
	img = cv2.imread(frame_dir)
	ori_h, ori_w, _ = img.shape
	if len(scale_size) == 1:
		if(ori_h > ori_w):
			tw = scale_size[0]
			th = (tw * ori_h) / ori_w
			th = int((th // 64) * 64)
		else:
			th = scale_size[0]
			tw = (th * ori_w) / ori_h
			tw = int((tw // 64) * 64)
	else:
		th, tw = scale_size
	img = cv2.resize(img, (tw, th))
	img = img.astype(np.float32)
	img = img / 255.0
	img = img[:, :, ::-1]
	img = np.transpose(img.copy(), (2, 0, 1))
	img = torch.from_numpy(img).float()
	img = color_normalize(img)
	return img, ori_h, ori_w


def restrict_neighborhood(h, w, size_mask_neighborhood):
	# We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
	mask = torch.zeros(h, w, h, w)
	for i in range(h):
		for j in range(w):
			for p in range(2 * size_mask_neighborhood + 1):
				for q in range(2 * size_mask_neighborhood + 1):
					if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
						continue
					if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
						continue
					mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

	mask = mask.reshape(h * w, h * w)
	return mask.cuda(non_blocking=True)


def read_seg(seg_dir, factor, scale_size=[480]):
	seg = Image.open(seg_dir)
	_w, _h = seg.size # note PIL.Image.Image's size is (w, h)
	if len(scale_size) == 1:
		if(_w > _h):
			_th = scale_size[0]
			_tw = (_th * _w) / _h
			_tw = int((_tw // 64) * 64)
		else:
			_tw = scale_size[0]
			_th = (_tw * _h) / _w
			_th = int((_th // 64) * 64)
	else:
		_th = scale_size[1]
		_tw = scale_size[0]
	small_seg = np.array(seg.resize((_tw // factor, _th // factor), 0))
	small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
	return to_one_hot(small_seg), np.asarray(seg)


# N = h*w, i.e. the number of patches in an image
# new_feature: N x d
# feature_stack: num_context x d x N
# diffusion_num: ---"k = number of diffusions wanted", "infinite"
def diffusion(feat_tar, feat_sources, diffusion_num, size_mask_neighborhood, topk, h, w, 
			  ncontext, mask_neighborhood, multiscale_rate, att_alpha = 0.1, infi_alpha = 1, 
			  sparse = True, multiscale = False, take_current=0):
	f_dim = feat_tar.shape[1]
	tr_aff = None
	if diffusion_num == "infinite":
		overall_feat = torch.cat((feat_tar.unsqueeze(0),feat_sources))
		overall_feat = F.normalize(overall_feat,dim=2,p=2)
		overall_feat = overall_feat.reshape((ncontext+1)*h*w,f_dim)
		# Normalizing
		aff = torch.softmax(torch.matmul(overall_feat,overall_feat.T)/att_alpha,1)
		tr_aff = copy.deepcopy(aff[:h*w,h*w:(ncontext+1)*h*w])
		# Implementing S = (I-\alpha*A)^{-1}
		inf_aff = torch.inverse(torch.eye(aff.shape[0]).cuda()-multiscale_rate*aff)
		if take_current:
			aff = inf_aff[:h*w,:]
			aff = aff.transpose(1,0).reshape(1+ncontext,h*w,h*w).transpose(2,1)
		else:
			aff = inf_aff[:h*w,h*w:(ncontext+1)*h*w]
			aff = aff.transpose(1,0).reshape(ncontext,h*w,h*w).transpose(2,1)
		tr_aff = tr_aff.transpose(1,0).reshape(ncontext,h*w,h*w).transpose(2,1)
	else:
		assert type(diffusion_num)==int
		overall_feat = torch.cat((feat_tar.unsqueeze(0),feat_sources))
		overall_feat = F.normalize(overall_feat,dim=2,p=2)
		overall_feat = overall_feat.reshape((ncontext+1)*h*w,f_dim)
		aff = torch.matmul(overall_feat,overall_feat.T)/att_alpha
		# Normalizing
		if diffusion_num>1:
			aff = torch.softmax(aff,1)
			if multiscale:
				aff_bank = []
				aff_bank.append(aff)
			for i in range(diffusion_num-1):
				aff = torch.matmul(aff,aff)
				if multiscale:
					aff_bank.append(aff)
			if multiscale:
				aff_bank = torch.stack(aff_bank)
				aff = (1-multiscale_rate)*aff_bank[0]+multiscale_rate*aff_bank[1]
				#torch.mean(aff_bank,0)
		else:
			aff = torch.exp(aff) # nmb_context x h*w (tar: query) x h*w (source: keys)
		if take_current:
			aff = inf_aff[:h*w,:]
			aff = aff.transpose(1,0).reshape(1+ncontext,h*w,h*w).transpose(2,1)
		else:
			aff = inf_aff[:h*w,h*w:(ncontext+1)*h*w]
			aff = aff.transpose(1,0).reshape(ncontext,h*w,h*w).transpose(2,1)

	if size_mask_neighborhood > 0:
		if mask_neighborhood is None:
			mask_neighborhood = restrict_neighborhood(h, w, size_mask_neighborhood)
			if take_current:
				mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext+1, 1, 1)
			else:
				mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
		print(mask_neighborhood.shape, aff.shape, tr_aff.shape)
		aff *= mask_neighborhood
		tr_aff *= mask_neighborhood[1:,:,:]

	aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
	tr_aff = tr_aff.transpose(2, 1).reshape(-1, h * w)
	if sparse:
		tk_val, _ = torch.topk(aff, dim=0, k=topk)
		tk_val_min, _ = torch.min(tk_val, dim=0)
		aff[aff < tk_val_min] = 0

		tk_val, _ = torch.topk(tr_aff, dim=0, k=topk)
		tk_val_min, _ = torch.min(tk_val, dim=0)
		tr_aff[tr_aff < tk_val_min] = 0

	aff = aff / torch.sum(aff, keepdim=True, axis=0)
	tr_aff = tr_aff / torch.sum(tr_aff, keepdim=True, axis=0)
	return aff, mask_neighborhood, tr_aff


def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
	for t, m, s in zip(x, mean, std):
		t.sub_(m)
		t.div_(s)
	return x


def to_one_hot(y_tensor, n_dims=None):
	"""
	Take integer y (tensor or variable) with n dims &
	convert it to 1-hot representation with n+1 dims.
	"""
	if(n_dims is None):
		n_dims = int(y_tensor.max()+ 1)
	_,h,w = y_tensor.size()
	y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
	n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
	y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
	y_one_hot = y_one_hot.view(h,w,n_dims)
	return y_one_hot.permute(2, 0, 1).unsqueeze(0)


def imwrite_indexed(filename, array, color_palette):
	""" Save indexed png for DAVIS."""
	if np.atleast_3d(array).shape[2] != 1:
	  raise Exception("Saving indexed PNGs requires 2D array.")

	im = Image.fromarray(array)
	im.putpalette(color_palette.ravel())
	im.save(filename, format='PNG')


def salt_pepper(image, noise):
	s_vs_p = 0.5
	amount = noise
	out = copy.deepcopy(image)
	# Salt mode
	num_salt = np.ceil(amount * image.nelement() * s_vs_p)
	coords = [np.random.randint(0, i, int(num_salt))
				for i in image.shape]
	out[tuple(coords)] = 1

	# Pepper mode
	num_pepper = np.ceil(amount* image.nelement() * (1. - s_vs_p))
	coords = [np.random.randint(0, i, int(num_pepper))
				for i in image.shape]
	out[tuple(coords)] = 0
	return out


def norm_mask(mask):
	c, h, w = mask.size()
	for cnt in range(c):
		mask_cnt = mask[cnt,:,:]
		if(mask_cnt.max() > 0):
			mask_cnt = (mask_cnt - mask_cnt.min())
			mask_cnt = mask_cnt/mask_cnt.max()
			mask[cnt,:,:] = mask_cnt
	return mask


# N = h*w, i.e. the number of patches in an image
# new_feature: N x d
def diffusion_image(feat_tar, size_mask_neighborhood, topk, h, w, 
			  mask_neighborhood, att_alpha = 0.1, infi_alpha = 1, 
			  sparse = True):
	f_dim = feat_tar.shape[1]
	# overall_feat = feat_tar.unsqueeze(0)
	overall_feat = F.normalize(feat_tar,dim=1,p=2)
	# overall_feat = overall_feat.reshape(h*w,f_dim)
		
	# Normalizing
	aff = torch.softmax(torch.matmul(overall_feat,overall_feat.T)/att_alpha,1)
	# Implementing S = (I-\alpha*A)^{-1}
	aff = torch.inverse(torch.eye(aff.shape[0]).cuda()-infi_alpha*aff)
	# aff = aff.transpose(1,0).reshape(ncontext,h*w,h*w).transpose(2,1)
	
	if size_mask_neighborhood > 0:
		if mask_neighborhood is None:
			mask_neighborhood = restrict_neighborhood(h, w, size_mask_neighborhood)
			# mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
		aff *= mask_neighborhood

	aff = aff.transpose(1, 0).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
	if sparse:
		tk_val, _ = torch.topk(aff, dim=0, k=topk)
		tk_val_min, _ = torch.min(tk_val, dim=0)
		aff[aff < tk_val_min] = 0
	aff = aff / torch.sum(aff, keepdim=True, axis=0)
	return aff, mask_neighborhood




