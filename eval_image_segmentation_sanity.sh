echo "alpha : 0.55"
python3 /playpen-storage/maitrey/davis-2017/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path /playpen-storage/maitrey/temp/_0_0.9/ --davis_path /playpen-storage/maitrey/davis-2017/davis-2017/DAVIS/
echo ""
echo "alpha : 0.6"
python3 /playpen-storage/maitrey/davis-2017/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path /playpen-storage/maitrey/temp/_0.01_0.9/ --davis_path /playpen-storage/maitrey/davis-2017/davis-2017/DAVIS/
echo ""
echo "alpha : 0.65"
python3 /playpen-storage/maitrey/davis-2017/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path /playpen-storage/maitrey/temp/_0.02_0.9/ --davis_path /playpen-storage/maitrey/davis-2017/davis-2017/DAVIS/
echo ""
echo "alpha : 0.7"
python3 /playpen-storage/maitrey/davis-2017/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path /playpen-storage/maitrey/temp/_0.05_0.9/ --davis_path /playpen-storage/maitrey/davis-2017/davis-2017/DAVIS/
echo ""
echo "alpha : 0.75"
python3 /playpen-storage/maitrey/davis-2017/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path /playpen-storage/maitrey/temp/_0.1_0.9/ --davis_path /playpen-storage/maitrey/davis-2017/davis-2017/DAVIS/
echo ""
echo "alpha : 0.8"
python3 /playpen-storage/maitrey/davis-2017/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path /playpen-storage/maitrey/temp/_0.2_0.9/ --davis_path /playpen-storage/maitrey/davis-2017/davis-2017/DAVIS/
echo ""
echo "alpha : 0.85"
python3 /playpen-storage/maitrey/davis-2017/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path /playpen-storage/maitrey/temp/_0.5_0.9/ --davis_path /playpen-storage/maitrey/davis-2017/davis-2017/DAVIS/
echo ""
echo "alpha : 0.9"
python3 /playpen-storage/maitrey/davis-2017/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path /playpen-storage/maitrey/temp/_1_0.9/ --davis_path /playpen-storage/maitrey/davis-2017/davis-2017/DAVIS/
echo ""
echo "alpha : 0.1"