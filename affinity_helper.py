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

import utils
import vision_transformer as vits
from pdb import set_trace


def restrict_neighborhood(h, w, args):
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


# N = h*w, i.e. the number of patches in an image
# new_feature: N x d
# feature_stack: num_context x d x N
# diffusion_num: ---"k = number of diffusions wanted", "infinite"
def diffusion(feat_tar, feat_sources, diffusion_num, size_mask_neighborhood, topk, h, w, 
			  ncontext, mask_neighborhood, multiscale_rate, att_alpha = 0.1, infi_alpha = 1, 
			  sparse = True, multiscale = False):
	f_dim = feat_tar.shape[1]
	if diffusion_num == "infinite":
		overall_feat = torch.cat((feat_tar.unsqueeze(0),feat_sources))
		overall_feat = F.normalize(overall_feat,dim=2,p=2)
		overall_feat = overall_feat.reshape((ncontext+1)*h*w,f_dim)
		# Normalizing
		aff = torch.softmax(torch.matmul(overall_feat,overall_feat.T)/att_alpha,1)
		# Implementing S = (I-\alpha*A)^{-1}
		# inf_aff = torch.inverse(torch.eye(aff.shape[0]).cuda()-multiscale_rate*aff)
		aff = aff[:h*w,h*w:(ncontext+1)*h*w]
		aff = aff.transpose(1,0).reshape(ncontext,h*w,h*w).transpose(2,1)
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
		aff = inf_aff[:h*w,h*w:(ncontext+1)*h*w]
		aff = aff.transpose(1,0).reshape(ncontext,h*w,h*w).transpose(2,1)

	if size_mask_neighborhood > 0:
		if mask_neighborhood is None:
			mask_neighborhood = restrict_neighborhood(h, w, size_mask_neighborhood)
			mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
		aff *= mask_neighborhood
		
	aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
	if sparse:
		tk_val, _ = torch.topk(aff, dim=0, k=topk)
		tk_val_min, _ = torch.min(tk_val, dim=0)
		aff[aff < tk_val_min] = 0

	aff = aff / torch.sum(aff, keepdim=True, axis=0)
	return aff, mask_neighborhood


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
	# aff1 = copy.deepcopy(aff)
	# aff = torch.inverse(torch.eye(aff.shape[0]).cuda()-infi_alpha*aff)
	# aff = aff.transpose(1,0).reshape(ncontext,h*w,h*w).transpose(2,1)
	
	if size_mask_neighborhood > 0:
		if mask_neighborhood is None:
			mask_neighborhood = restrict_neighborhood(h, w, size_mask_neighborhood)
			# mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
		aff *= mask_neighborhood
		# aff1 *= mask_neighborhood

	aff = aff.transpose(1, 0).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
	# aff1 = aff1.transpose(1, 0).reshape(-1, h * w)
	if sparse:
		tk_val, _ = torch.topk(aff, dim=0, k=topk)
		tk_val_min, _ = torch.min(tk_val, dim=0)
		aff[aff < tk_val_min] = 0

	# if sparse:
	# 	tk_val, _ = torch.topk(aff1, dim=0, k=topk)
	# 	tk_val_min, _ = torch.min(tk_val, dim=0)
	# 	aff1[aff1 < tk_val_min] = 0

	aff = aff / torch.sum(aff, keepdim=True, axis=0)
	# aff1 = copy.deepcopy(aff)
	set_trace()
	pass
	aff = torch.inverse(torch.eye(aff.shape[0]).cuda()-infi_alpha*aff)
	aff = aff / torch.sum(aff, keepdim=True, axis=0)
	# aff1 = aff1 / torch.sum(aff1, keepdim=True, axis=0)
	return aff, mask_neighborhood

