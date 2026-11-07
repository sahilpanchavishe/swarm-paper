#!/bin/bash
set -e
cd "$(dirname "$0")"
for s in d15_A_conn1 d15_A_comp2_same d15_A_comp2_split d15_A_comp3_same d15_change; do
  caffeinate -i python d12_eval.py $s 2>&1 | tee ../results/log_$s.txt
done
