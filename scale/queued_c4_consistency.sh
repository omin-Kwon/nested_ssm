#!/bin/bash
G=$1
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets
export HF_HUB_CACHE=/home/omin/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/omin/nested_ssm/scale
# (1) raw 9B GSM8K (missing 3-way cell)
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_raw_gsm.py
# (2) recall 6 at c=4 (consistency with GSM8K's c=4 headline)
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 \
  --tasks fda swde squad_completion triviaqa nq_open drop --limit 300 --c 4 --pb 32 --tag recall_c4
echo "C4 CONSISTENCY ALL DONE"
