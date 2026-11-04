#!/bin/bash
# Day 6c: extra TRAINING seeds for the memoryless GNNs (K = 0,1,2,3), behaviour cloning then DAgger, no test evaluation here
# (d6_main_eval.py does that). Resume-safe: a model whose runs/d5_K*_s*.pt exists is skipped. Takes about 2 hours per seed.
# Usage (from src/, venv active):  caffeinate -i bash d6_more_seeds.sh 1 2        -> trains seeds 1 and 2
for seed in "$@"; do
  for K in 3 2 1 0; do
    if [ -f "../runs/d5_K${K}_s${seed}.pt" ]; then echo "K=$K seed=$seed already done, skipping"; continue; fi
    echo "=== K=$K seed=$seed : behaviour cloning ==="
    python d4_train.py --K $K --seed $seed --epochs 30 --eval 0 2>&1 | tee -a ../logs/d6_seeds.log
    echo "=== K=$K seed=$seed : DAgger ==="
    python d5_dagger.py --K $K --seed $seed --rounds 3 --eps 240 --epochs 10 --dev 5 --eval 0 2>&1 | tee -a ../logs/d6_seeds.log
  done
done
echo "ALL DONE"
