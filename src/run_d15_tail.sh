#!/bin/bash
cd "$(dirname "$0")"
echo "waiting for run_d15_rest.sh ..."
while pgrep -f run_d15_rest.sh > /dev/null; do sleep 30; done
grep -q saved ../results/log_d15_cutoff_secondary.txt || { echo "CHAIN INCOMPLETE: no 'saved' in log_d15_cutoff_secondary.txt"; exit 1; }
python d15_stats_B.py 2>&1 | tee ../results/log_stats_B.txt
python d15_stats_C.py 2>&1 | tee ../results/log_stats_C.txt
caffeinate -i python d12_eval.py d15_frontier 2>&1 | tee ../results/log_d15_frontier.txt
caffeinate -i python d15_drift.py 2>&1 | tee ../results/log_d15_drift.txt
caffeinate -i python d15_replay.py 2>&1 | tee ../results/log_d15_replay.txt
python - <<'PY'
import pandas as pd
d = pd.read_csv("../results/d15_drift.csv"); TS = [0, 100, 200, 300, 500, 700, 1099]
pd.DataFrame({"t": TS, "uninformed_radius_mm": [1000 * d[f"rad_un_{t}"].mean() for t in TS], "observer_radius_mm": [1000 * d[f"rad_ob_{t}"].mean() for t in TS],
              "mean_components": [d[f"comp_{t}"].mean() for t in TS], "pct_connected": [100 * (d[f"comp_{t}"] == 1).mean() for t in TS]}).to_csv("../results/d15_drift_summary.csv", index=False)
PY
python d15_tables.py 2>&1 | tee ../results/log_tables.txt
echo "TAIL DONE"
