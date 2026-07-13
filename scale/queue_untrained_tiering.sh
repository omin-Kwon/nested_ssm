#!/bin/bash
G=$1
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets HF_HUB_CACHE=/home/omin/.cache/huggingface/hub PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/omin/nested_ssm/scale
T="--tasks gsm8k --limit 200"
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --pb 32 --c 4 --coldfp32 1 $T --ckpt nemo9b_identity.pt --tag unt_fp32
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --pb 32 --c 4 $T --ckpt nemo9b_identity.pt --tag unt_bf16
echo UNTRAINED TIERING DONE
