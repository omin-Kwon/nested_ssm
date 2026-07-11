#!/bin/bash
G=$1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/omin/nested_ssm/scale
PY=~/nemo_env/bin/python3
COMMON="--batch 1 --seqlen 4096 --lr 2e-3 --cosine --v4aware --tune_decay \
  --cs_menu 2 4 8 16 32 --data mixed_train_nemo.npy --resume nemo9b_rot_p4long.pt"
# smoke 4 steps -> /tmp; abort if it doesn't save
echo "[retrain] SMOKE seqlen4096 b1"
CUDA_VISIBLE_DEVICES=$G $PY nemotron_retrofit.py $COMMON --steps 4 --log_every 2 --save /tmp/smoke_lc.pt
if [ ! -f /tmp/smoke_lc.pt ]; then echo "[retrain] SMOKE FAILED (OOM?) — aborting"; exit 1; fi
echo "[retrain] SMOKE OK -> full 1500 steps"
CUDA_VISIBLE_DEVICES=$G $PY nemotron_retrofit.py $COMMON --steps 1500 --log_every 300 --save nemo9b_rot_longcot.pt
echo "RETRAIN LONGCOT DONE"
