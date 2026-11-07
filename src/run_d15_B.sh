#!/bin/bash
set -e
cd "$(dirname "$0")"
caffeinate -i python d12_eval.py d15_filter_dist 2>&1 | tee ../results/log_d15_filter_dist.txt
