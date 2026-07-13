#!/bin/bash
# wait for a GPU with >=45GB free, then run the selfspec B=256 bench there
while true; do
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2 < 138000 {print $1; exit}')
  [ -n "$G" ] && break
  sleep 120
done
echo "GPU $G window opened for selfspec bench"
cd /home/omin/nested_ssm/scale
CUDA_VISIBLE_DEVICES=$G HF_HUB_CACHE=/NHNHOME/ARC/arclab/shared/hub ~/nemo_env/bin/python3 bench_selfspec_e2e.py --B 256
