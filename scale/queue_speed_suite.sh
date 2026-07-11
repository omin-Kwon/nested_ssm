#!/bin/bash
G=$1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets
export HF_HUB_CACHE=/home/omin/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/omin/nested_ssm/scale
echo "== [1/2] state-op bench B=256,448 (incl fp8 arm, fixed flush sampling) =="
CUDA_VISIBLE_DEVICES=$G python3 bench_v4_decode.py --Bs 256 448 --cs 4 8 16 64
echo "== [2/2] E2E decode bench B=256 (raw/fresh/v4c16/v4c4) =="
for ARM in raw fresh v4c16 v4c4 v4c16fp8 v4c4fp8; do
  CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 e2e_decode_bench.py $ARM --B 256 --gen 128
done
echo "SPEED SUITE DONE"
