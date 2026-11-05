#!/bin/bash
# Day 10a: extra TRAINING seeds for the latched-token GNN (K=2). BC then DAgger. Resume-safe.
# Usage (from src/, venv active):  caffeinate -i bash d10_latched_seeds.sh 1 2
for seed in "$@"; do
  if [ -f "../runs/d5_K2_latched_s${seed}.pt" ]; then echo "latched seed=$seed already done, skipping"; continue; fi
  echo "=== latched K=2 seed=$seed : behaviour cloning ==="
  python d4_train.py --K 2 --latched --seed $seed --epochs 30 --eval 0 2>&1 | tee -a ../logs/d10_latched.log
  echo "=== latched K=2 seed=$seed : DAgger ==="
  python d5_dagger.py --K 2 --latched --seed $seed --rounds 3 --eps 240 --epochs 10 --dev 5 --eval 0 2>&1 | tee -a ../logs/d10_latched.log
done
echo "ALL DONE"
