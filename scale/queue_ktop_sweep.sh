#!/bin/bash
G=$1
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets HF_HUB_CACHE=/home/omin/.cache/huggingface/hub PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/omin/nested_ssm/scale
CK=nemo9b_rot_longcot2.pt
T="--tasks gsm8k --limit 200 --coldfp32 1"
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --pb 32 --c 4 $T --ckpt $CK --tag kt_full
for K in 32 16 8; do
  CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --pb 32 --c 4 --ktop $K $T --ckpt $CK --tag kt_$K
done
echo KTOP SWEEP DONE
