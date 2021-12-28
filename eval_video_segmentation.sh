masks=(0 2 5 8 12 15)
topks=(1 2 4 8 16 32 64 128 256 512 1680)
# masks=(5)
# topks=(16)
layer=1
gpu=0

while getopts l:g: flag
do
    case "${flag}" in
        l) layer=${OPTARG};;
        g) gpu=${OPTARG};;
    esac
done

for m in ${masks[@]}; do
	for k in ${topks[@]}; do
		echo "GPU : $gpu    Layer : $layer    Mask : $m    topk : $k"
		python3 eval_video_segmentation.py --data_path /playpen-storage/maitrey/davis-2017/davis-2017/DAVIS/ --output_dir /playpen-storage/maitrey/temp/ \
		--layer $layer --size_mask_neighborhood $m --topk $k --gpu $gpu
		echo ""
	done
done
