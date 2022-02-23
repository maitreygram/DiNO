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
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

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

# def _convert_image_to_rgb(image):
# 	return image.convert("RGB")

def preprocess_clip(image, tw, th):
	preprocess = transforms.Compose([
		transforms.Resize((th, tw), interpolation=transforms.InterpolationMode.BICUBIC),
		# CenterCrop(n_px),
		_convert_image_to_rgb,
		transforms.ToTensor(),
		transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
	])

	return preprocess(image)

def preprocess_general(image, tw, th):
	image = cv2.resize(image, (tw, th), interpolation = cv2.INTER_CUBIC)
	image = image.astype(np.float32)
	image = image / 255.0
	image = image[:, :, ::-1]
	image = np.transpose(image.copy(), (2, 0, 1))
	image = torch.from_numpy(image).float()
	image = color_normalize(image)
	return image

def preprocess_vit(image, tw, th):
	image = transforms.Compose([
		transforms.Resize((th, tw)), 
		transforms.ToTensor(),
		transforms.Normalize(0.5, 0.5),
	])(image)

	return image

def preprocess_swin(image, tw, th):
	image = transforms.Compose([
		transforms.Resize((th, tw)), 
		# transforms.RandomCrop((th - th%7, tw)), 
		transforms.ToTensor(),
		transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
	])(image)

	return image

def preprocess_mae(image, tw, th):
	image = transforms.Compose([
		# _convert_image_to_rgb, 
		transforms.Resize((th, tw)), 
		transforms.ToTensor(),
		transforms.Normalize(
			torch.tensor(IMAGENET_DEFAULT_MEAN), 
			torch.tensor(IMAGENET_DEFAULT_STD))
	])(image)

	return image

def preprocess_convnext(image, tw, th):
	image = transforms.Compose([
		# _convert_image_to_rgb, 
		transforms.Resize((th, tw), 
			interpolation=transforms.InterpolationMode.BICUBIC), 
		transforms.ToTensor(),
		transforms.Normalize(
			torch.tensor(IMAGENET_DEFAULT_MEAN), 
			torch.tensor(IMAGENET_DEFAULT_STD))
	])(image)

	return image

def preprocess_swav(image, tw, th):
	image = transforms.Compose([
		# _convert_image_to_rgb, 
		transforms.Resize((th, tw)), 
		transforms.ToTensor(),
		transforms.Normalize(
			torch.tensor([0.485, 0.456, 0.406]), 
			torch.tensor([0.228, 0.224, 0.225]))
	])(image)

	return image

def read_frame(frame_dir, model_name, scale_size=[480]):
	"""
	read a single frame & preprocess
	"""

	cv2_models = ['dino.vit', 'dino.conv', 'deit', 'mlp_mixer', 'resnet50', 'resnet152', 'resnet200', 'resnext', 'beit']
	if model_name in cv2_models:
		img = cv2.imread(frame_dir)
		ori_h, ori_w, _ = img.shape

	else:
		img = Image.open(frame_dir).convert('RGB')
		ori_w, ori_h = img.size

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
	
	if 'clip' in model_name:
		img = preprocess_clip(img, tw, th)
	elif 'dino' in model_name or model_name == 'deit' or 'resnet' in model_name or model_name == 'resnext':
		img = preprocess_general(img, tw, th)
	elif 'vit' in model_name == 'vit_B_16_imagenet1k' or model_name == 'vit_small_patch16_224':
		img = preprocess_vit(img, tw, th)
	elif model_name == 'swin':
		img = preprocess_swin(img, tw, th)
	elif model_name == 'mae':
		img = preprocess_mae(img, tw, th)
	elif model_name == 'convnext':
		img = preprocess_convnext(img, tw, th)
	elif 'swav' in model_name:
		img = preprocess_swav(img, tw, th)
	elif model_name == 'mlp_mixer' or model_name == 'beit':
		img = preprocess_mlp_mixer_or_beit(img, tw, th)

	return img, ori_h, ori_w


def resize_seg(seg, _tw, _th, factor, model_name, layer):
	sp_models = ['swin', 'convnext', 'swav', 'swav_w2', 'clip.conv', 'dino.conv']
	if model_name not in sp_models:
		return np.array(seg.resize((_tw // factor, _th // factor), 0))

	if model_name == 'swin':
		if layer < 3:
			return np.array(seg.resize((_tw // factor // 2, _th // factor // 2), 0))
		if layer == 3:
			return np.array(seg.resize((_tw // factor, _th // factor), 0))
		if layer == 4:
			return np.array(seg.resize((_tw // factor * 2, _th // factor * 2), 0))
	elif model_name == 'convnext' or model_name == 'clip.conv' or model_name == 'dino.conv' or model_name == 'swav_w2':
		if layer == 1:
			return np.array(seg.resize((_tw // factor // 2, _th // factor // 2), 0))
		if layer == 2:
			return np.array(seg.resize((_tw // factor, _th // factor), 0))
		if layer == 3:
			return np.array(seg.resize((_tw // factor * 2, _th // factor * 2), 0))
		if layer == 4:
			return np.array(seg.resize((_tw // factor * 4, _th // factor * 4), 0))
	elif model_name == 'swav':
		if layer == 1:
			return np.array(seg.resize((_tw // factor // 2 + 1, _th // factor // 2 + 1), 0))
		if layer == 2:
			return np.array(seg.resize((_tw // factor + 1, _th // factor + 1), 0))
		if layer == 3:
			return np.array(seg.resize((_tw // factor * 2 + 1, _th // factor * 2 + 1), 0))
		if layer == 4:
			return np.array(seg.resize((_tw // factor * 4 + 1, _th // factor * 4 + 1), 0))

	return


def read_seg(seg_dir, factor, model_name, layer, scale_size=[480]):
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
		_th = scale_size[0]
		_tw = scale_size[1]
	# small_seg = np.array(seg.resize((_tw // factor, _th // factor), 0))
	small_seg = resize_seg(seg, _tw, _th, factor, model_name, layer)
	small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
	return to_one_hot(small_seg), np.asarray(seg)


def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
	for t, m, s in zip(x, mean, std):
		t.sub_(m)
		t.div_(s)
	return x


def imwrite_indexed(filename, array, color_palette):
	""" Save indexed png for DAVIS."""
	if np.atleast_3d(array).shape[2] != 1:
	  raise Exception("Saving indexed PNGs requires 2D array.")

	im = Image.fromarray(array)
	im.putpalette(color_palette.ravel())
	im.save(filename, format='PNG')


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


def norm_mask(mask):
	c, h, w = mask.size()
	for cnt in range(c):
		mask_cnt = mask[cnt,:,:]
		if(mask_cnt.max() > 0):
			mask_cnt = (mask_cnt - mask_cnt.min())
			mask_cnt = mask_cnt/mask_cnt.max()
			mask[cnt,:,:] = mask_cnt
	return mask
