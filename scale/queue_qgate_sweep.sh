#!/bin/bash
until grep -q "saved nemo9b_rot_qreg2.pt" /home/omin/nested_ssm/scale/logs/qreg_train.log 2>/dev/null; do sleep 30; done
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets HF_HUB_CACHE=/home/omin/.cache/huggingface/hub PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/omin/nested_ssm/scale
CK=nemo9b_rot_qreg2.pt
CUDA_VISIBLE_DEVICES=0 ~/nemo_env/bin/python3 run_recall_native.py fresh --ckpt $CK --tasks gsm8k --limit 100 --tag qg_fresh
CUDA_VISIBLE_DEVICES=0 ~/nemo_env/bin/python3 run_recall_native.py v4 --c 4 --pb 32 --ckpt $CK --tasks gsm8k --limit 100 --tag qg_v4
for TAU in 0.3 0.45 0.6; do
  CUDA_VISIBLE_DEVICES=0 ~/nemo_env/bin/python3 run_recall_native.py v4 --c 4 --pb 32 --qgate $TAU --ckpt $CK --tasks gsm8k --limit 100 --tag qg_tau$TAU
done
echo QGATE SWEEP DONE
