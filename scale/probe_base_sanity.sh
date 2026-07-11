#!/bin/bash
G=$1
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets
export HF_HUB_CACHE=/home/omin/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ALLOW_CODE_EVAL=1
cd /home/omin/nested_ssm/scale
B="--model nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py raw $B --tasks niah_single_1 --limit 100 --maxlen 4096 --tag base_niah
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py raw $B --tasks gsm8k --limit 150 --tag base_gsm
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py raw $B --tasks humaneval --limit 100 --tag base_code
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py raw $B --tasks minerva_math --limit 100 --tag base_math
echo "BASE SANITY DONE"
