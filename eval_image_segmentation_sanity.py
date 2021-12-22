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
import sys
import copy
import glob
import queue
from urllib.request import urlopen
import argparse
import numpy as np
from tqdm import tqdm
from pdb import set_trace
import multiprocessing as mp

import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms

import utils
import vision_transformer as vits
import helper


@torch.no_grad()
def eval_img_seg_sanity_davis(args, model, frame_list, video_dir, seg_list, color_palette, noise, diffuse):
	"""
	Evaluate tracking on a video given first frame & segmentation
	"""
	# video_folder = os.path.join(args.output_dir, video_dir.split('/')[-1])
	video_folder = os.path.join(args.output_dir+'_'+str(noise)+'_'+str(diffuse), video_dir.split('/')[-1])
	os.makedirs(video_folder, exist_ok=True)

	# The queue stores the n preceeding frames
	# que = queue.Queue(args.n_last_frames)

	# first frame
	_, ori_h, ori_w = helper.read_frame(frame_list[0])
	# extract first frame feature
	# frame1_feat = extract_feature(model, frame1).T #  dim x h*w

	# saving first segmentation
	# out_path = os.path.join(video_folder, "00000.png")
	# imwrite_indexed(out_path, seg_ori, color_palette)
	mask_neighborhood = None
	for cnt in tqdm(range(0, len(frame_list))):
		frame_tar = helper.read_frame(frame_list[cnt])[0]
		seg, seg_ori = helper.read_seg(seg_list[cnt], args.patch_size)

		# we use the first segmentation and the n previous ones
		# used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
		# used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

		# pop out oldest frame if neccessary
		# if que.qsize() == args.n_last_frames:
		# 	que.get()
		# push current results into queue
		# seg = copy.deepcopy(frame_tar_avg)
		# que.put([feat_tar, seg])

		seg = helper.salt_pepper(seg, noise)
		frame_tar_avg, _, mask_neighborhood = label_propagation(args, model, frame_tar, seg, diffuse, mask_neighborhood)
		# frame_tar_avg = copy.deepcopy(seg)
		# set_trace()

		# upsampling & argmax
		frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=args.patch_size, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
		frame_tar_avg = helper.norm_mask(frame_tar_avg)
		_, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

		# saving to disk
		frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
		frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0))
		frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg", ".png")
		helper.imwrite_indexed(os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette)


def restrict_neighborhood(h, w):
	# We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
	mask = torch.zeros(h, w, h, w)
	for i in range(h):
		for j in range(w):
			for p in range(2 * args.size_mask_neighborhood + 1):
				for q in range(2 * args.size_mask_neighborhood + 1):
					if i - args.size_mask_neighborhood + p < 0 or i - args.size_mask_neighborhood + p >= h:
						continue
					if j - args.size_mask_neighborhood + q < 0 or j - args.size_mask_neighborhood + q >= w:
						continue
					mask[i, j, i - args.size_mask_neighborhood + p, j - args.size_mask_neighborhood + q] = 1

	mask = mask.reshape(h * w, h * w)
	return mask.cuda(non_blocking=True)


def label_propagation(args, model, frame_tar, seg, diffuse, mask_neighborhood=None):
	"""
	propagate segs of frames in list_frames to frame_tar
	"""
	## we only need to extract feature of the target frame
	feat_tar, h, w = extract_feature(model, frame_tar, return_h_w=True)
	seg = seg.cuda()

	return_feat_tar = feat_tar.T # dim x h*w

	aff, mask_neighborhood = helper.diffusion_image(feat_tar, args.size_mask_neighborhood, args.topk, 
        h, w, mask_neighborhood, infi_alpha=diffuse, sparse = args.sparse)

	C = seg.shape[1]
	seg = seg.reshape(C, -1)#.transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
	seg_tar = torch.mm(seg, aff)
	seg_tar = seg_tar.reshape(1, C, h, w)
	return seg_tar, return_feat_tar, mask_neighborhood
 

def extract_feature(model, frame, return_h_w=False):
	"""Extract one frame feature everytime."""
	out = model.get_intermediate_layers(frame.unsqueeze(0).cuda(), n=1)[0]
	out = out[:, 1:, :]  # we discard the [CLS] token
	h, w = int(frame.shape[1] / model.patch_embed.patch_size), int(frame.shape[2] / model.patch_embed.patch_size)
	dim = out.shape[-1]
	out = out[0].reshape(h, w, dim)
	out = out.reshape(-1, dim)
	if return_h_w:
		return out, h, w
	return out

def process_vid_main(args, model, color_palette, video_list, noise=0, diffuse=0):
	print(os.getpid(), "Start Noise: ", noise, "\tAlpha: ", diffuse, "\tGPU: ", args.gpu)
	log_path = os.path.join(args.log, "img_snt_" + str(noise) + "_" + str(diffuse) + ".txt")
	sys.stdout = open(log_path, "w")
	sys.stderr = sys.stdout
	print("Eval image sanity")
	print("Start Noise: ", noise, "\tAlpha: ", diffuse, "\tGPU: ", args.gpu)
	# log_file.write("Noise: " + str(noise) + "\tAlpha: " + str(diffuse) + "\n")
	for i, video_name in enumerate(video_list):
		video_name = video_name.strip()
		print(f'[{i}/{len(video_list)}] Begin to segmentate video {video_name}.')
		video_dir = os.path.join(args.data_path, "JPEGImages/480p/", video_name)
		frame_list = helper.read_frame_list(video_dir)
		seg_list = helper.get_seg_list(frame_list)
		eval_img_seg_sanity_davis(args, model, frame_list, video_dir, 
			seg_list, color_palette, noise, diffuse)
	sys.stdout.close()
	sys.stderr.close()
	sys.stdout = sys.__stdout__
	sys.stderr = sys.__stderr__
	print("Done Noise: ", noise, "\tAlpha: ", diffuse)


if __name__ == '__main__':
	parser = argparse.ArgumentParser('Evaluation with video object segmentation on DAVIS 2017')
	parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
	parser.add_argument('--arch', default='vit_small', type=str,
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
	parser.add_argument("--log", type=str, default="logs", help="Log file")
	parser.add_argument("--noise", type=float, default=0, help="Noise level to add to the segmentation")
	parser.add_argument("--gpu", type=str, default="0", help="GPU number to use")
	parser.add_argument("--sparse", type=int, default=1, help="Sparse or not")
	# parser.add_argument("--diffuse", type=float, default=0, help="Diffusion alpha")
	args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	print(os.getpid())
	print("git:\n  {}\n".format(utils.get_sha()))
	print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

	# building network
	model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
	print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
	model.cuda()
	utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
	for param in model.parameters():
		param.requires_grad = False
	model.eval()

	color_palette = []
	for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
		color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
	color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)

	video_list = open(os.path.join(args.data_path, "ImageSets/2017/val.txt")).readlines()
	noise_list = [args.noise]#[0]#, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
	diffuse_list = list(np.arange(1.0, 1.02, 0.1))
	
	mp.set_start_method('spawn')
	# all_processes = mp.Queue()
	for i, noise in enumerate(noise_list):
		for diffuse in diffuse_list:
			p = mp.Process(target=process_vid_main, args=(args, model, color_palette, video_list, noise, diffuse))
			p.start()
			# print(all_processes.get())
			# all_processes.append(p)
			# process_vid_main(args, model, color_palette, video_list, noise, diffuse)

	# for p in all_processes:
	# 	p.join()
