#!/bin/bash
G=$1
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets
export HF_HUB_CACHE=/home/omin/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/omin/nested_ssm/scale
for spec in "raw|" "fresh|" "v4|--c 4 --pb 32"; do
  MODE=${spec%%|*}; EXTRA=${spec##*|}
  [ "$MODE" = "v4" ] && TAG=v4c4 || TAG=$MODE
  CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py $MODE $EXTRA --tasks minerva_math --limit 100 --tag math_$TAG
done
echo "MATH 3ARM DONE"
