# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Some parts are taken from https://github.com/Liusifei/UVC
"""
import os
import copy
import glob
import queue
from urllib.request import urlopen
import argparse
import numpy as np
from tqdm import tqdm
from pdb import set_trace

import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
import torchvision.models as models

import utils
import vision_transformer as vits
from transformers import BeitModel
# from modeling import MlpMixer
import ml_collections

from read_helper import *
from affinity_helper import *
from util_helper import *
from extract_features import *


@torch.no_grad()
def eval_video_tracking_davis(args, model, model_name, frame_list, video_dir, first_seg, seg_ori, color_palette, img_size):
	"""
	Evaluate tracking on a video given first frame & segmentation
	"""
	video_folder = os.path.join(args.output_dir + "video_" + str(args.layer) + "_mask_" + str(args.size_mask_neighborhood) + "_topk_" + str(args.topk), video_dir.split('/')[-1])
	os.makedirs(video_folder, exist_ok=True)

	# The queue stores the n preceeding frames
	que = queue.Queue(args.n_last_frames)

	# first frame
	frame1, ori_h, ori_w = read_frame(frame_list[0], model_name, scale_size=img_size)
	set_trace()
	# extract first frame feature

	patch_h = first_seg.shape[2]
	patch_w = first_seg.shape[3]
	frame1_feat = extract_feature(model, args, frame1, layer=args.layer, model_name=model_name, patch_h=patch_h, patch_w=patch_w).T #  dim x h*w

	# saving first segmentation
	out_path = os.path.join(video_folder, "00000.png")
	imwrite_indexed(out_path, seg_ori, color_palette)
	mask_neighborhood = None
	for cnt in tqdm(range(1, len(frame_list))):
		frame_tar = read_frame(frame_list[cnt], model_name, scale_size=img_size)[0]

		# we use the first segmentation and the n previous ones
		used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
		used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

		frame_tar_avg, feat_tar, mask_neighborhood = label_propagation(args, model, frame_tar, used_frame_feats, used_segs, mask_neighborhood)

		# pop out oldest frame if neccessary
		if que.qsize() == args.n_last_frames:
			que.get()
		# push current results into queue
		seg = copy.deepcopy(frame_tar_avg)
		que.put([feat_tar, seg])

		# upsampling & argmax
		frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=args.patch_size, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
		frame_tar_avg = norm_mask(frame_tar_avg)
		_, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

		# saving to disk
		frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
		frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0))
		frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg", ".png")
		imwrite_indexed(os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette)


def label_propagation(args, model, frame_tar, list_frame_feats, list_segs, mask_neighborhood=None):
	"""
	propagate segs of frames in list_frames to frame_tar
	"""
	## we only need to extract feature of the target frame
	feat_tar, h, w = extract_feature(model, args, frame_tar, return_h_w=True, layer=args.layer, model_name=model_name)

	return_feat_tar = feat_tar.T # dim x h*w

	ncontext = len(list_frame_feats)
	feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w

	feat_tar = F.normalize(feat_tar, dim=1, p=2)
	feat_sources = F.normalize(feat_sources, dim=1, p=2)

	feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
	aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1) # nmb_context x h*w (tar: query) x h*w (source: keys)

	if args.size_mask_neighborhood > 0:
		if mask_neighborhood is None:
			mask_neighborhood = restrict_neighborhood(h, w, args)
			mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
		aff *= mask_neighborhood

	aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
	tk_val, _ = torch.topk(aff, dim=0, k=args.topk)
	tk_val_min, _ = torch.min(tk_val, dim=0)
	aff[aff < tk_val_min] = 0

	aff = aff / torch.sum(aff, keepdim=True, axis=0)
	# aff = torch.inverse(torch.eye(aff.shape[0]).cuda()-infi_alpha*aff)
	# aff = aff / torch.sum(aff, keepdim=True, axis=0)

	list_segs = [s.cuda() for s in list_segs]
	segs = torch.cat(list_segs)
	nmb_context, C, h, w = segs.shape
	segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
	set_trace()
	seg_tar = torch.mm(segs, aff.to(segs.dtype))
	seg_tar = seg_tar.reshape(1, C, h, w)
	return seg_tar, return_feat_tar, mask_neighborhood


# building network
def load_model(model_name, args, ori_h, ori_w):
	if model_name=='dino.vit':
		model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

	if model_name=='dino.conv':
		model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
	
	elif model_name=='deit':
		model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
		'''
		model = list(model.children())
		final_model = []
		for i in range(len(model)):
			if i!=2:
				final_model.append(model[i])
			else:
				final_model+=list(model[i])
		model = torch.nn.Sequential(*(final_model[:-1]))
		'''  
	
	elif model_name=='clip.vit':
		import clip
		model, preprocess = clip.load("ViT-B/16")
		model = model.visual
		model.transformer = [ele for ele in model.transformer.children()][0]
		model.transformer = torch.nn.Sequential(*[ele for ele in model.transformer.children()])

	elif model_name=='clip.conv':
		import clip
		model, preprocess = clip.load("RN50x4")
		model = model.visual
	
	elif model_name=='vit_B_16_imagenet1k':
		# from pytorch_pretrained_vit import ViT
		# model = ViT('B_16_imagenet1k', pretrained=True)
		# model.patch_size = args.patch_size
		# model.transformer = [ele for ele in model.transformer.children()][0]
		# model.transformer = torch.nn.Sequential(*[ele for ele in model.transformer.children()])
		import timm
		model = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=(ori_h, ori_w))

	elif model_name=='vit_small_patch16_224':
		import timm
		model = timm.create_model('vit_small_patch16_224', pretrained=True, img_size=(ori_h, ori_w))

	elif model_name=='vit_B_32_imagenet1k':
		from pytorch_pretrained_vit import ViT
		model = ViT('B_32_imagenet1k', pretrained=True)
		model.transformer = [ele for ele in model.transformer.children()][0]
		model.transformer = torch.nn.Sequential(*[ele for ele in model.transformer.children()])

	elif model_name=="swin":
		import timm
		model_weights = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True).state_dict()
		model_keys = list(model_weights.keys())
		model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, img_size=(ori_h, ori_w))
		for k in model_keys:
			if (model.state_dict()[k].shape != model_weights[k].shape):
				model_weights.pop(k, None)
				# print(k)
		model.load_state_dict(model_weights, strict=False)

	elif model_name=="mae":
		import models_mae
		model = getattr(models_mae, "mae_vit_base_patch16")(img_size=(ori_h, ori_w))
		checkpoint = torch.load("/playpen-storage/maitrey/ViT_VOS/mae_pretrain_vit_base.pth", map_location='cpu')
		checkpoint['model']['pos_embed'] = interpolate_pos_encoding_mae(
			model, checkpoint['model']['pos_embed'].squeeze(0), ori_h//args.patch_size, ori_w//args.patch_size)
		model.load_state_dict(checkpoint['model'], strict=False)

	elif model_name=="convnext":
		import timm
		model = timm.create_model(
			"convnext_tiny", 
			pretrained=True, 
			# drop_path_rate=0.2,
			# layer_scale_init_value=1e-6,
			# head_init_scale=1.0,
			)

	elif model_name=="swav":
		model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
		model.padding = nn.ConstantPad2d(1, 0.0)

	elif model_name=="swav_w2":
		model = torch.hub.load('facebookresearch/swav:main', 'resnet50w2')
		# model.padding = nn.ConstantPad2d(1, 0.0)

	elif model_name=='mlp_mixer':
		def get_mixer_b16_config():
			"""Returns Mixer-B/16 configuration."""
			config = ml_collections.ConfigDict()
			config.name = 'Mixer-B_16'
			config.patches = ml_collections.ConfigDict({'size': (16, 16)})
			config.hidden_dim = 768
			config.num_blocks = 12
			config.tokens_mlp_dim = 384
			config.channels_mlp_dim = 3072
			return config
		def get_mixer_l16_config():
			"""Returns Mixer-L/16 configuration."""
			config = ml_collections.ConfigDict()
			config.name = 'Mixer-L_16'
			config.patches = ml_collections.ConfigDict({'size': (16, 16)})
			config.hidden_dim = 1024
			config.num_blocks = 24
			config.tokens_mlp_dim = 512
			config.channels_mlp_dim = 4096
			return config
		config = get_mixer_b16_config()
		model = MlpMixer(config, 224 , num_classes=10, patch_size=16, zero_head=True)
		model.load_from(np.load('./imagenet1k-Mixer-B_16.npz'))

	elif model_name == 'resnet50':
		model = models.resnet50(pretrained=True)

	elif model_name == 'resnet152':
		model = models.resnet152(pretrained=True)

	elif model_name == 'resnet200':
		import timm
		model = timm.create_model('resnet200d', pretrained=True)

	elif model_name == 'resnext':
		model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)

	elif model_name == 'beit':
		model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224')

	else:
		raise Exception('Model name not registered')

	model.patch_size = args.patch_size
	model.cuda()
	for param in model.parameters():
		param.requires_grad = False
	model.eval()
	print(f"Model {model_name} - {args.patch_size}. Image size = {ori_h}, {ori_w}. Parameters = {np.round(sum(p.numel() for p in model.parameters())/1000000, 2)}M.")
	return model


if __name__ == '__main__':
	parser = argparse.ArgumentParser('Evaluation with video object segmentation on DAVIS 2017')
	parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
	parser.add_argument('--arch', default='vit_base', type=str,
		choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
	parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
	parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
	parser.add_argument('--output_dir', default=".", help='Path where to save segmentations')
	parser.add_argument('--data_path', default='/path/to/davis/', type=str)
	parser.add_argument("--n_last_frames", type=int, default=7, help="number of preceeding frames")
	parser.add_argument("--size_mask_neighborhood", default=12, type=int,
		help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
	parser.add_argument("--topk", type=int, default=5, help="accumulate label from top k neighbors")
	parser.add_argument("--bs", type=int, default=6, help="Batch size, try to reduce if OOM")
	parser.add_argument("--layer", type=int, default=1, help="Which model layer features from the end to use for affinity matrix")
	parser.add_argument("--gpu", type=str, default="0", help="GPU number to use")
	parser.add_argument("--model_name", type=str, default="dino", help="model_name to use")
	args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	print("git:\n  {}\n".format(utils.get_sha()))
	print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

	model_name = args.model_name
	img_size = [480, 896]
	model = load_model(model_name, args, img_size[0], img_size[1])
	set_trace()

	color_palette = []
	for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
		color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
	color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)

	video_list = open(os.path.join(args.data_path, "ImageSets/2017/val.txt")).readlines()
	for i, video_name in enumerate(video_list):
		video_name = video_name.strip()
		print(f'[{i}/{len(video_list)}] Begin to segmentate video {video_name}.')
		video_dir = os.path.join(args.data_path, "JPEGImages/480p/", video_name)
		frame_list = read_frame_list(video_dir)
		seg_path = frame_list[0].replace("JPEGImages", "Annotations").replace("jpg", "png")
		first_seg, seg_ori = read_seg(seg_path, args.patch_size, model_name, args.layer, scale_size=img_size)
		eval_video_tracking_davis(args, model, model_name, frame_list, video_dir, first_seg, seg_ori, color_palette, img_size)


# Per-sequence results saved in /playpen-storage/maitrey/ViT_VOS/ViT_VOS_outputs/temp/video_1_mask_5_topk_4/per-sequence_results-val.csv
# --------------------------- Global results for val ---------------------------
#  J&F-Mean   J-Mean  J-Recall  J-Decay   F-Mean  F-Recall  F-Decay
#  0.265386 0.300683  0.248218 0.155161 0.230089  0.060328 0.114954