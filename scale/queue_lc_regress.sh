#!/bin/bash
G=$1
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets HF_HUB_CACHE=/home/omin/.cache/huggingface/hub PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/omin/nested_ssm/scale
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py fresh --ckpt nemo9b_rot_longcot.pt --limit 300 --tag lcreg_fresh
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --c 4 --pb 32 --ckpt nemo9b_rot_longcot.pt --limit 300 --tag lcreg_v4c4
echo LCREG DONE
