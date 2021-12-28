layers=(1 2 3 4 5 6 7 8 9 10 11 12)
masks=(12 15)
topks=(1 2 4 8 16 32 64 128 256 512 1680)
# layers=(2)
# masks=(5)
# topks=(16)
# gpu=0

# while getopts l:g: flag
# do
#     case "${flag}" in
#         l) layer=${OPTARG};;
#         g) gpu=${OPTARG};;
#     esac
# done

for l in ${layers[@]}; do
	for m in ${masks[@]}; do
		for k in ${topks[@]}; do
			echo "Layer : $l    Mask : $m    topk : $k"
			res_pth="/playpen-storage/maitrey/temp/video_${l}_mask_${m}_topk_${k}/"
			echo $res_pth
			python3 /playpen-storage/maitrey/davis-2017/davis2017-evaluation/evaluation_method.py --task semi-supervised \
			--results_path $res_pth --davis_path /playpen-storage/maitrey/davis-2017/davis-2017/DAVIS/
			echo ""
		done
	done
done
