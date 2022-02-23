import os
import copy
import glob
import queue
from urllib.request import urlopen
import argparse
import numpy as np
from tqdm import tqdm
import math
from pdb import set_trace

import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms

import utils
import vision_transformer as vits


def extract_feature(model, args, frame, return_h_w=False, layer=1, model_name='dino', patch_h=None, patch_w=None):
	"""Extract one frame feature everytime."""
	out, h, w, dim = get_intermediate_layers(model, frame, n=layer, model_name=model_name)

	out = out.reshape(h, w, dim)
	out = out.reshape(-1, dim)
	if return_h_w:
		return out, h, w
	return out


def get_intermediate_layers(model, x, n=1, model_name='dino'):
	if model_name == 'dino.vit':
		out = get_intermediate_layers_dino(model, x.unsqueeze(0).cuda(), n)[0][0, 1:, :]
		h, w = x.shape[1] // model.patch_size, x.shape[2] // model.patch_size
		dim = out.shape[-1]
		return out, h, w, dim
	if model_name == 'clip.vit':
		out = get_intermediate_layers_clip_vit(model, x.type(torch.HalfTensor).unsqueeze(0).cuda(), n)[0][0, 1:, :]
		h, w = x.shape[1] // model.patch_size, x.shape[2] // model.patch_size
		dim = out.shape[-1]
		return out, h, w, dim
	if model_name == 'vit_B_16_imagenet1k' or model_name == 'vit_small_patch16_224':
		out = get_intermediate_layers_vit(model, x.unsqueeze(0).cuda(), n)[0][0, 1:, :]
		h, w = x.shape[1] // model.patch_size, x.shape[2] // model.patch_size
		dim = out.shape[-1]
		return out, h, w, dim
	if model_name == 'vit_B_32_imagenet1k':
		out = get_intermediate_layers_vit(model, x.unsqueeze(0).cuda(), n)[0][0, 1:, :]
		h, w = x.shape[1] // model.patch_size, x.shape[2] // model.patch_size
		dim = out.shape[-1]
		return out, h, w, dim
	if model_name == 'mae':
		out = get_intermediate_layers_mae(model, x.unsqueeze(0).cuda(), n)[0][0, 1:, :]
		h, w = x.shape[1] // model.patch_size, x.shape[2] // model.patch_size
		dim = out.shape[-1]
		return out, h, w, dim
	if model_name == 'swin':
		out = get_intermediate_layers_swin(model, x.unsqueeze(0).cuda(), n)[0][0]
		dim = out.shape[-1]
		if n < 3:
			h, w = x.shape[1] // model.patch_size // 2, x.shape[2] // model.patch_size // 2
		if n == 3:
			h, w = x.shape[1] // model.patch_size, x.shape[2] // model.patch_size
		if n == 4:
			h, w = x.shape[1] // model.patch_size * 2, x.shape[2] // model.patch_size * 2
		return out, h, w, dim
	if model_name == 'convnext':
		out = get_intermediate_layers_convnext(model, x.unsqueeze(0).cuda(), n)[0][0].permute((1, 2, 0))
		dim = out.shape[-1]
		if n == 1:
			h, w = x.shape[1] // model.patch_size // 2, x.shape[2] // model.patch_size // 2
		if n == 2:
			h, w = x.shape[1] // model.patch_size, x.shape[2] // model.patch_size
		if n == 3:
			h, w = x.shape[1] // model.patch_size * 2, x.shape[2] // model.patch_size * 2
		if n == 4:
			h, w = x.shape[1] // model.patch_size * 4, x.shape[2] // model.patch_size * 4
		return out, h, w, dim
	if model_name == 'dino.conv':
		out = get_intermediate_layers_dino_conv(model, x.unsqueeze(0).cuda(), n)[0][0].permute((1, 2, 0))
		dim = out.shape[-1]
		if n == 1:
			h, w = x.shape[1] // model.patch_size // 2, x.shape[2] // model.patch_size // 2
		if n == 2:
			h, w = x.shape[1] // model.patch_size, x.shape[2] // model.patch_size
		if n == 3:
			h, w = x.shape[1] // model.patch_size * 2, x.shape[2] // model.patch_size * 2
		if n == 4:
			h, w = x.shape[1] // model.patch_size * 4, x.shape[2] // model.patch_size * 4
		return out, h, w, dim
	if model_name == 'clip.conv':
		out = get_intermediate_layers_clip_conv(model, x.unsqueeze(0).cuda(), n)[0][0].permute((1, 2, 0))
		dim = out.shape[-1]
		if n == 1:
			h, w = x.shape[1] // model.patch_size // 2, x.shape[2] // model.patch_size // 2
		if n == 2:
			h, w = x.shape[1] // model.patch_size, x.shape[2] // model.patch_size
		if n == 3:
			h, w = x.shape[1] // model.patch_size * 2, x.shape[2] // model.patch_size * 2
		if n == 4:
			h, w = x.shape[1] // model.patch_size * 4, x.shape[2] // model.patch_size * 4
		return out, h, w, dim
	if model_name == 'swav':
		out = get_intermediate_layers_swav(model, x.unsqueeze(0).cuda(), n)[0][0].permute((1, 2, 0))
		dim = out.shape[-1]
		if n == 1:
			h, w = x.shape[1] // model.patch_size // 2 + 1, x.shape[2] // model.patch_size // 2 + 1
		if n == 2:
			h, w = x.shape[1] // model.patch_size + 1, x.shape[2] // model.patch_size + 1
		if n == 3:
			h, w = x.shape[1] // model.patch_size * 2 + 1, x.shape[2] // model.patch_size * 2 + 1
		if n == 4:
			h, w = x.shape[1] // model.patch_size * 4 + 1, x.shape[2] // model.patch_size * 4 + 1
		return out, h, w, dim

	if model_name == 'swav_w2':
		out = get_intermediate_layers_swav(model, x.unsqueeze(0).cuda(), n)[0][0].permute((1, 2, 0))
		dim = out.shape[-1]
		if n == 1:
			h, w = x.shape[1] // model.patch_size // 2, x.shape[2] // model.patch_size // 2
		if n == 2:
			h, w = x.shape[1] // model.patch_size, x.shape[2] // model.patch_size
		if n == 3:
			h, w = x.shape[1] // model.patch_size * 2, x.shape[2] // model.patch_size * 2
		if n == 4:
			h, w = x.shape[1] // model.patch_size * 4, x.shape[2] // model.patch_size * 4
		return out, h, w, dim

	if 'resnet' in model_name or model_name == 'resnext':
		out = get_intermediate_layers_resnet(model, x.unsqueeze(0).cuda(), n)
		num_patches = patch_w*patch_h
		out = out[0]
		out = out.reshape(out.shape[0]*out.shape[1],out.shape[2])
		assert out.shape[0]>=num_patches
		out = out[:num_patches]
		h, w = patch_h, patch_w
		dim = out.shape[-1]
		return out, h, w, dim
	
	if model_name == 'deit':
		out = get_intermediate_layers_deit(model, x.unsqueeze(0).cuda(), n)[0]
		h, w = x.shape[1] // model.patch_size, x.shape[2] // model.patch_size
		dim = out.shape[-1]
		return out, h, w, dim

	if model_name == 'beit':
		out = get_intermediate_layers_beit(model, x.unsqueeze(0).cuda(), n)[0]
		h, w = x.shape[1] // model.patch_size, x.shape[2] // model.patch_size
		dim = out.shape[-1]
		return out, h, w, dim

	if model_name == 'mlp_mixer':
		out = get_intermediate_layers_mlp_mixer(model, x.unsqueeze(0).cuda(), n)[0, 1:, :]
		h, w = x.shape[1] // model.patch_size, x.shape[2] // model.patch_size
		dim = out.shape[-1]
		return out, h, w, dim


def interpolate_pos_encoding(model, positional_embedding, x, w, h):
	npatch = x.shape[1] - 1
	N = positional_embedding.shape[0] - 1
	if npatch == N and w == h:
		return positional_embedding.unsqueeze(0).to(x.dtype)
	class_pos_embed = positional_embedding.unsqueeze(0)[:, 0].to(x.dtype)
	patch_pos_embed = positional_embedding.unsqueeze(0)[:, 1:].to(x.dtype)
	dim = x.shape[-1]
	w0 = w // model.patch_size
	h0 = h // model.patch_size
	# we add a small number to avoid floating point error in the interpolation
	# see discussion at https://github.com/facebookresearch/dino/issues/8
	w0, h0 = w0 + 0.1, h0 + 0.1
	patch_pos_embed = F.interpolate(
		patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
		scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
		mode='bicubic',
	)
	assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
	patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
	return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


def get_intermediate_layers_dino(model, x, n=1):
	out = model(x)
	x = model.prepare_tokens(x)
	# we return the output tokens from the `n` last blocks
	output = []
	for i, blk in enumerate(model.blocks):
		x = blk(x)
		if len(model.blocks) - i <= n:
			output.append(model.norm(x))
	return output

def get_intermediate_layers_dino_conv(model, x, n=1):
	x = model.conv1(x)
	x = model.bn1(x)
	x = model.relu(x)
	x = model.maxpool(x)

	if n == 0:
		return [x]
	x = model.layer1(x)
	if n == 4:
		return [x]
	x = model.layer2(x)
	if n == 3:
		return [x]
	x = model.layer3(x)
	if n == 2:
		# set_trace()
		return [x]
	x = model.layer4(x)

	# x = model.avgpool(x)
	# x = torch.flatten(x, 1)
	# x = model.fc(x)

	return [x]


def get_intermediate_layers_clip_vit(model, x, n=1):
	B, nc, w, h = x.shape
	x = model.conv1(x)  # shape = [*, width, grid, grid]
	x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
	x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
	x = torch.cat([model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
	x = x + interpolate_pos_encoding(model, model.positional_embedding, x, w, h)
	x = model.ln_pre(x)

	x = x.permute(1, 0, 2)  # NLD -> LND
	output = []
	for i, blk in enumerate(model.transformer):
		x = blk(x)
		x1 = x.permute(1, 0, 2)  # LND -> NLD
		x1 = model.ln_post(x1)#[:, 0, :])
		# if model.proj is not None:
		#     x1 = x1 @ model.proj

		if len(model.transformer) - i <= n:
			output.append(x1)

	# set_trace()
	# x = model.transformer(x)
	# return x.unsqueeze(0)
	return output


def get_intermediate_layers_clip_conv(model, x, n=1):
	B, nc, w, h = x.shape
	def stem(x):
		for conv, bn in [(model.conv1, model.bn1), (model.conv2, model.bn2), (model.conv3, model.bn3)]:
			x = model.relu(bn(conv(x)))
		x = model.avgpool(x)
		return x

	x = x.type(model.conv1.weight.dtype)
	x = stem(x)
	if n == 0:
		return [x]
	x = model.layer1(x)
	if n == 4:
		return [x]
	x = model.layer2(x)
	if n == 3:
		return [x]
	x = model.layer3(x)
	if n == 2:
		# set_trace()
		return [x]
	x = model.layer4(x)
	# x = model.attnpool(x)

	return [x]


def get_intermediate_layers_vit(model, x, n=1):
	"""Breaks image into patches, applies transformer, applies MLP head.
	Args:
		x (tensor): `b,c,fh,fw`
	"""
	# b, c, fh, fw = x.shape
	# x = model.patch_embedding(x)  # b,d,gh,gw
	# x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
	# if hasattr(model, 'class_token'):
	# 	x = torch.cat((model.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
	# if hasattr(model, 'positional_embedding'):
	# 	x = x + interpolate_pos_encoding(model, model.positional_embedding.pos_embedding.squeeze(0), x, fh, fw)  # b,gh*gw+1,d 
	# # set_trace()
	# # x = model.transformer(x)  # b,gh*gw+1,d
	# output = []
	# for i, blk in enumerate(model.transformer): 
	# 	x = blk(x, None)
	# 	x1 = copy.deepcopy(x)
	# 	if hasattr(model, 'pre_logits'):
	# 		x1 = model.pre_logits(x)
	# 		# x1 = torch.tanh(x1)
	# 		# x1 = model.norm(x1)
	# 	# if hasattr(model, 'fc'):
	# 	#     x1 = model.norm(x1)[:, 0]  # b,d
	# 	#     x1 = model.fc(x1)  # b,num_classes
	# 	if len(model.transformer) - i <= n:
	# 		output.append(x1)
	# return output

	x = model.patch_embed(x)
	cls_token = model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
	x = torch.cat((cls_token, x), dim=1)
	x = model.pos_drop(x + model.pos_embed)
	output = []
	for i, blk in enumerate(model.blocks): 
		x = blk(x)
		x1 = copy.deepcopy(x)
		x1 = model.norm(x1)
		if len(model.blocks) - i <= n:
			output.append(x1)
	return output


def get_intermediate_layers_swin(model, x, n=1):
	B, nc, w, h = x.shape
	x = model.patch_embed(x)
	output = []
	for i, blk in enumerate(model.layers): 
		x = blk(x)
		x1 = copy.deepcopy(x)
		# if i > 1:
		# 	x1 = model.norm(x1)  # B L C
		if len(model.layers) - i <= n:
			output.append(x1)
	return output
	# return [x]
	# x = model.patch_embed.proj(x)
	# if model.patch_embed.flatten:
	# 	x = x.flatten(2).transpose(1, 2)
	# x = model.patch_embed.norm(x)
	# if model.absolute_pos_embed is not None:
	# 	x = x + interpolate_pos_encoding(model, model.absolute_pos_embed, x, w, h)
	# pass


def interpolate_pos_encoding_mae(model, positional_embedding, w, h):
	npatch = w*h
	N = positional_embedding.shape[0] - 1
	if npatch == N and w == h:
		return positional_embedding.unsqueeze(0)#.to(x.dtype)
	class_pos_embed = positional_embedding.unsqueeze(0)[:, 0]#.to(x.dtype)
	patch_pos_embed = positional_embedding.unsqueeze(0)[:, 1:]#.to(x.dtype)
	dim = positional_embedding.shape[-1]
	w0 = w
	h0 = h
	# we add a small number to avoid floating point error in the interpolation
	# see discussion at https://github.com/facebookresearch/dino/issues/8
	w0, h0 = w0 + 0.1, h0 + 0.1
	patch_pos_embed = F.interpolate(
		patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
		scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
		mode='bicubic',
	)
	assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
	patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
	return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


def get_intermediate_layers_mae(model, x, n=1):
	# embed patches
	B, nc, w, h = x.shape
	x = model.patch_embed(x)

	# # add pos embed w/o cls token
	# x = x + interpolate_pos_encoding_mae(model, model.pos_embed[0, 1:, :], x, w, h)
	x = x + model.pos_embed[:, 1:, :]

	# # masking: length -> length * mask_ratio
	# x, mask, ids_restore = model.random_masking(x, 0.75)

	# # append cls token
	cls_token = model.cls_token + model.pos_embed[:, :1, :]
	cls_tokens = cls_token.expand(x.shape[0], -1, -1)
	x = torch.cat((cls_tokens, x), dim=1)

	# # apply Transformer blocks
	output = []
	for i, blk in enumerate(model.blocks): 
		x = blk(x)
		x1 = copy.deepcopy(x)
		x1 = model.norm(x1)
		if len(model.blocks) - i <= n:
			output.append(x1)
	# set_trace()
	return output
	# return [x]


def get_intermediate_layers_convnext(model, x, n=1):
	# embed patches
	B, nc, w, h = x.shape
	# set_trace()
	x = model.stem(x)
	output = []
	for i, blk in enumerate(model.stages):
		x = blk(x)
		x1 = copy.deepcopy(x)
		x1 = model.norm_pre(x1)
		if len(model.stages) - i <= n:
			output.append(x1)
	return output


def get_intermediate_layers_swav(model, x, n=1):
	# embed patches
	B, nc, w, h = x.shape
	x = model.padding(x)

	x = model.conv1(x)
	x = model.bn1(x)
	x = model.relu(x)
	x = model.maxpool(x)
	x = model.layer1(x)
	if n == 4:
		return [x]
	x = model.layer2(x)
	if n == 3:
		return [x]
	x = model.layer3(x)
	if n == 2:
		return [x]
	x = model.layer4(x)

	return [x]


def get_intermediate_layers_resnet(model, x, n=1):
	model = nn.Sequential(*list(model.children())[:-n])
	output = model(x)
	#output = output.reshape(output.shape[0],output.shape[1],-1)
	output = torch.tensor(np.transpose(output.cpu().numpy(),(0,2,3,1))).cuda()
	return output


def get_intermediate_layers_beit(model, x, n=1):

	class BeitRelativePositionBias(nn.Module):
		def __init__(self, window_size):
			super().__init__()
			self.window_size = window_size
			self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
			self.relative_position_bias_table = nn.Parameter(
				torch.zeros(self.num_relative_distance, 12)
			)  # 2*Wh-1 * 2*Ww-1, nH
			# cls to token & token 2 cls & cls to cls

			# get pair-wise relative position index for each token inside the window
			coords_h = torch.arange(window_size[0])
			coords_w = torch.arange(window_size[1])
			coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
			coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
			relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
			relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
			relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
			relative_coords[:, :, 1] += window_size[1] - 1
			relative_coords[:, :, 0] *= 2 * window_size[1] - 1
			relative_position_index = torch.zeros(
				size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
			)
			relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
			relative_position_index[0, 0:] = self.num_relative_distance - 3
			relative_position_index[0:, 0] = self.num_relative_distance - 2
			relative_position_index[0, 0] = self.num_relative_distance - 1

			self.register_buffer("relative_position_index", relative_position_index)

		def forward(self):
			relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
				self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1
			)  # Wh*Ww,Wh*Ww,nH

			return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww



	def embeddings(model,x):
		embeddings = model.embeddings.patch_embeddings.projection(x).flatten(2).transpose(1, 2)
		batch_size, seq_len, _ = embeddings.size()
		cls_tokens = model.embeddings.cls_token.expand(batch_size, -1, -1)
		embeddings = torch.cat((cls_tokens, embeddings), dim=1)
		return embeddings

	image_size = [x.shape[2],x.shape[3]]
	patch_size = [16,16]
	patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
	
	hidden_states = embeddings(model,x)
	count = 0
	for i, layer_module in enumerate(model.encoder.layer):

		layer_module.attention.attention.relative_position_bias = BeitRelativePositionBias(window_size=patch_shape).cuda()
		layer_outputs = layer_module(hidden_states)

		hidden_states = layer_outputs[0]
		count += 1
		if count==len(model.encoder.layer)-n-1:
			break
	return hidden_states


def get_intermediate_layers_mlp_mixer(model, x, n=1):
	x = model.stem(x)
	x = x.flatten(2)
	x = x.transpose(-1, -2)
	count = 0
	for block in model.layer:
		x = block(x)
		count+=1
		if count==len(model.layer)-n-1:
			break
	return x


def get_intermediate_layers_deit(model,x, n = 1):
	def patch_embed(model,x):
		B, C, H, W = x.shape
		x = model.patch_embed.proj(x)
		x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
		return x

	def forward_features(model,x,n):
		B, nc, w, h = x.shape
		x = patch_embed(model,x)
		cls_token = model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
		x = torch.cat((cls_token, x), dim=1)
		#x = x + model.pos_embed
		x = x + interpolate_pos_encoding(model, nn.Parameter(model.pos_embed.squeeze(0)), x, w, h)
		for i in range(0,len(model.blocks)-n):
			x = model.blocks[i](x)
		x = model.norm(x)
		return x
		#return x[0,1:]

	x = forward_features(model,x,n)
	return x