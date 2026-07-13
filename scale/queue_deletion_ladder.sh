#!/bin/bash
# deletion-vs-lazy ladder: (a) fresh (b) GHOST-lite perm+delete (c) trainedR+delete (d) trainedR+lazy
G=$1
export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets HF_HUB_CACHE=/home/omin/.cache/huggingface/hub PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/omin/nested_ssm/scale
T="--tasks gsm8k fda swde --limit 200"
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py fresh --ckpt nemo9b_rot_longcot.pt $T --tag del_fresh
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --pb 32 --c 4 --coldoff 1 --ckpt nemo9b_ghost_perm.pt $T --tag del_ghost
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --pb 32 --c 4 --coldoff 1 --ckpt nemo9b_rot_longcot.pt $T --tag del_rotdel
CUDA_VISIBLE_DEVICES=$G ~/nemo_env/bin/python3 run_recall_native.py v4 --pb 32 --c 4 --ckpt nemo9b_rot_longcot.pt $T --tag del_lazy
echo DELETION LADDER DONE
