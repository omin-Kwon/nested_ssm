#!/bin/bash
G=$1; MODE=$2   # MODE = fresh | v4c4
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets
export HF_HUB_CACHE=/home/omin/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ALLOW_CODE_EVAL=1
cd /home/omin/nested_ssm/scale
if [ "$MODE" = "v4c4" ]; then ARM="v4"; EXTRA="--c 4 --pb 32"; TAG=v4c4; else ARM="fresh"; EXTRA=""; TAG=fresh; fi
# (A) RULER remaining tasks @ 4096
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py $ARM $EXTRA \
  --tasks niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue ruler_cwe ruler_fwe ruler_vt \
  --limit 100 --maxlen 4096 --tag ruler2_$TAG
# (B) long-output decode-heavy: math CoT + code gen
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py $ARM $EXTRA \
  --tasks minerva_math --limit 100 --tag math_$TAG
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py $ARM $EXTRA \
  --tasks humaneval --limit 100 --tag code_$TAG
echo "GAPFILL $MODE DONE"
