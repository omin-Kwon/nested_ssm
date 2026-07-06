#!/bin/bash
# wait for 120M pair to finish, then run state-bound motivation benches on GPU 3
tail --pid=2700183 -f /dev/null 2>/dev/null
tail --pid=2700184 -f /dev/null 2>/dev/null
sleep 20
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
cd /home/omin/TTT-PNM/scale
CUDA_VISIBLE_DEVICES=3 python3 bench_state_bound.py --fam gdn --axis heads \
  --budgets 80 160 --ms 1 2 4 8 16 32 > bench_gdn.log 2>&1
CUDA_VISIBLE_DEVICES=3 python3 bench_state_bound.py --fam gla --axis heads \
  --budgets 80 160 --ms 1 2 4 8 16 32 > bench_gla.log 2>&1
CUDA_VISIBLE_DEVICES=3 python3 bench_state_bound.py --fam gdn --axis dk \
  --budgets 80 --ms 1 2 4 8 16 > bench_gdn_dk.log 2>&1
python3 plot_state_bound.py >> bench_gdn.log 2>&1
echo DONE > bench_chain.done
