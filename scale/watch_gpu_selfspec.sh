#!/bin/bash
# stable-window watcher: >=60GB free on the same GPU across two checks 60s apart
while true; do
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2 < 120000 {print $1; exit}')
  if [ -n "$G" ]; then
    sleep 60
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)
    if [ "$U" -lt 120000 ]; then
      echo "GPU $G stable window (used=${U}MiB) — launching selfspec bench"
      cd /home/omin/nested_ssm/scale
      CUDA_VISIBLE_DEVICES=$G HF_HUB_CACHE=/NHNHOME/ARC/arclab/shared/hub PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/nemo_env/bin/python3 bench_selfspec_e2e.py --B 256 && break
      echo "bench failed (window lost?) — rearming"
    fi
  fi
  sleep 120
done
