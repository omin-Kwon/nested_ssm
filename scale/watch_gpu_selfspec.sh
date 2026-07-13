#!/bin/bash
# stable-window watcher: selfspec bench (15min) then GSM-lossless training chain
while true; do
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2 < 120000 {print $1; exit}')
  if [ -n "$G" ]; then
    sleep 60
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)
    if [ "$U" -lt 120000 ]; then
      echo "GPU $G stable window (used=${U}MiB) — bench then training"
      cd /home/omin/nested_ssm/scale
      CUDA_VISIBLE_DEVICES=$G HF_HUB_CACHE=/NHNHOME/ARC/arclab/shared/hub PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/nemo_env/bin/python3 bench_selfspec_e2e.py --B 256
      echo "bench exit=$? — launching longcot2 training on GPU $G"
      CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_CACHE=/NHNHOME/ARC/arclab/shared/hub ~/nemo_env/bin/python3 nemotron_retrofit.py \
        --steps 1200 --batch 1 --seqlen 4096 --lr 1e-3 --cosine \
        --v4aware --cs_menu 2 4 4 8 8 16 32 --tune_decay --data mixed_train_nemo.npy \
        --resume nemo9b_rot_longcot.pt --log_every 100 --save_every 200 \
        --save nemo9b_rot_longcot2.pt > /home/omin/nested_ssm/scale/logs/longcot2_train.log 2>&1
      echo "training exit=$?"
      break
    fi
  fi
  sleep 120
done
