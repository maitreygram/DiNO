import re
import sys
from pdb import set_trace

results_JF = {}
results_J = {}
results_F = {}

file_pth = "logs/res_vid_seg_layers_all.txt"
file = open(file_pth)
data = file.read()
rows = data.split("\n")
res = [i for i,r in enumerate(rows) if "Per-sequence" in r]
for r in res:
	layer = int(re.search('video_(.+?)_mask_', rows[r]).group(1))
	mask = int(re.search('_mask_(.+?)_topk_', rows[r]).group(1))
	topk = int(re.search('_topk_(.+?)/per-sequence_results', rows[r]).group(1))
	row_res = rows[r+3].split()
	# print(layer, mask, topk)
	# print(rows[r+2])
	# print(rows[r+3].split())
	if layer not in results_JF:
		results_JF[layer] = {}
		results_J[layer] = {}
		results_F[layer] = {}

	if mask not in results_JF[layer]:
		results_JF[layer][mask] = {}
		results_J[layer][mask] = {}
		results_F[layer][mask] = {}
	
	if topk in results_JF[layer][mask]:
		sys.exit("exist")
	
	results_JF[layer][mask][topk] = float(row_res[0])
	results_J[layer][mask][topk] = float(row_res[1])
	results_F[layer][mask][topk] = float(row_res[4])

layers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
masks=[0, 2, 5, 8, 12, 15]
topks=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

print("J&F-mean")
for l in layers:
	print("Layer_number:", 13-l)
	print("masks\\topks", end="\t")
	for t in topks:
		print(t, end="\t")
	print()
	for m in masks:
		print(m, end="\t\t")
		for t in topks:
			if t in results_JF[l][m]:
				print("{:.4f}".format(results_JF[l][m][t]), end="\t")
			else:
				print("aerror", end="\t")
		print()
	print()

print("\n\n\n")
print("J-mean")
for l in layers:
	print("Layer_number:", 13-l)
	print("masks\\topks", end="\t")
	for t in topks:
		print(t, end="\t")
	print()
	for m in masks:
		print(m, end="\t\t")
		for t in topks:
			if t in results_J[l][m]:
				print("{:.4f}".format(results_J[l][m][t]), end="\t")
			else:
				print("aerror", end="\t")
		print()
	print()

print("\n\n\n")
print("F-mean")
for l in layers:
	print("Layer_number:", 13-l)
	print("masks\\topks", end="\t")
	for t in topks:
		print(t, end="\t")
	print()
	for m in masks:
		print(m, end="\t\t")
		for t in topks:
			if t in results_F[l][m]:
				print("{:.4f}".format(results_F[l][m][t]), end="\t")
			else:
				print("aerror", end="\t")
		print()
	print()

# set_trace()
# pass
# print(results_JF)