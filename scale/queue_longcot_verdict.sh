#!/bin/bash
G=$1
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets HF_HUB_CACHE=/home/omin/.cache/huggingface/hub PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/omin/nested_ssm/scale
for spec "fresh|" "v4|--c 4 --pb 32" "v4|--c 2 --pb 32"; do :; done 2>/dev/null
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py fresh --ckpt nemo9b_rot_longcot.pt --tasks minerva_math gsm8k --limit 100 --tag lc_fresh
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --c 4 --pb 32 --ckpt nemo9b_rot_longcot.pt --tasks minerva_math gsm8k --limit 100 --tag lc_v4c4
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --c 2 --pb 32 --ckpt nemo9b_rot_longcot.pt --tasks minerva_math gsm8k --limit 100 --tag lc_v4c2
echo LONGCOT VERDICT DONE
