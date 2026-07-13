#!/bin/bash
until grep -q "FINAL" /home/omin/nested_ssm/scale/logs/longcot2_train.log 2>/dev/null; do sleep 60; done
sleep 30
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets HF_HUB_CACHE=/home/omin/.cache/huggingface/hub PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/omin/nested_ssm/scale
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py fresh --ckpt nemo9b_rot_longcot2.pt --tasks gsm8k --limit 500 --tag lc2_fresh
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --c 4 --pb 32 --ckpt nemo9b_rot_longcot2.pt --tasks gsm8k --limit 500 --tag lc2_v4c4
echo LC2 VERDICT DONE
