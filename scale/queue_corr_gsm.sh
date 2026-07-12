#!/bin/bash
G=$1
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets HF_HUB_CACHE=/home/omin/.cache/huggingface/hub PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/omin/nested_ssm/scale
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --c 32 --pb 32 --ckpt nemo9b_rot_longcot.pt --tasks gsm8k --limit 200 --tag corr_c32stale
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --c 32 --pb 32 --corr 1 --ckpt nemo9b_rot_longcot.pt --tasks gsm8k --limit 200 --tag corr_c32corr
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py fresh --ckpt nemo9b_rot_longcot.pt --tasks gsm8k --limit 200 --tag corr_fresh
echo CORR GSM DONE
