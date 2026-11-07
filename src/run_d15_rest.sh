#!/bin/bash
set -e
cd "$(dirname "$0")"
caffeinate -i python d12_eval.py d15_filter_dist 2>&1 | tee ../results/log_d15_filter_dist.txt
caffeinate -i python d15_cutoff.py --models d5_K2_s0 d5_K2_s1 d5_K2_s2 d5_K2_s3 d5_K2_s4 d5_K2_latched_s0 d5_K2_latched_s1 d5_K2_latched_s2 \
  --cuts 0 100 200 300 500 700 1100 --variants flagoff flagon --out ../results/d15_cutoff_primary.csv 2>&1 | tee ../results/log_d15_cutoff_primary.txt
caffeinate -i python d15_cutoff.py --models d5_K1_s0 d5_K1_s1 d5_K1_s2 d5_K3_s0 d5_K3_s1 d5_K3_s2 \
  --cuts 0 300 700 1100 --variants flagoff flagon --out ../results/d15_cutoff_secondary.csv 2>&1 | tee ../results/log_d15_cutoff_secondary.txt
