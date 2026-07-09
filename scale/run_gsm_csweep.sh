#!/bin/bash
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets
export HF_HUB_CACHE=/home/omin/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=$1
cd /home/omin/nested_ssm/scale
# c-dial: does lower c recover GSM8K? plus pb=64 at c16
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --tasks gsm8k --limit 150 --c 4  --pb 32 --tag gsm_c4
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --tasks gsm8k --limit 150 --c 8  --pb 32 --tag gsm_c8
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --tasks gsm8k --limit 150 --c 16 --pb 64 --tag gsm_pb64
echo "GSM CSWEEP ALL DONE"
