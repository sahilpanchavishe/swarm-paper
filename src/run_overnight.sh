#!/bin/bash
cd "$(dirname "$0")/.." && source .venv/bin/activate
L=results/log_overnight.txt; exec > >(tee -a $L) 2>&1
echo "== waiting for d15_pred_conf"; while pgrep -f "d12_eval.py d15_pred_conf" >/dev/null; do sleep 30; done
grep -q saved results/log_d15_pred_conf.txt || (cd src && python d12_eval.py d15_pred_conf 2>&1 | tee -a ../results/log_d15_pred_conf.txt)
echo "== P1-P3 scoring"
python - <<'PY'
import pandas as pd
d = pd.read_csv('results/d12_d15_pred_conf.csv'); d['cf'] = d.success.astype(bool) & ~d.collision.astype(bool)
g = d.groupby(['policy','delay','obs_pct']).agg(n=('cf','size'), success=('success','mean'), contact=('collision','mean'), cf=('cf','mean')).round(3)
print(g.to_string()); g.to_csv('results/d15_P_quick.csv')
PY
git add results/d12_d15_pred_conf.csv results/d15_P_quick.csv results/log_d15_pred_conf.txt
git commit -m "Day 15: delay-aware filter confirmatory results (P1-P3)"; git pull --rebase --autostash; git push
echo "== no-memory relay under observer errors (exploratory addition)"
python - <<'PY'
p='src/d12_common.py'; s=open(p).read()
old='elif pol["mem"] == "none": know = obs if K == 0 else reach(A, obs, K); tok = np.full(N, ci)'
new='elif pol["mem"] == "none": know = obs_e if K == 0 else reach(A, obs_e, K); tok = cn'
assert s.count(old)==1; open(p,'w').write(s.replace(old,new))
p='src/d15_suites.py'; s=open(p).read(); old='if __name__ == "__main__":'
assert s.count(old)==1 and "d15_obs_nomem" not in s
open(p,'w').write(s.replace(old,'SUITES15["d15_obs_nomem"] = dict(SUITES15["d15_obs"], policies=["relay K2", "relayD K2"])\n\n'+old))
PY
python -m py_compile src/d12_common.py || exit 1
cat >> 01_protocol/protocol_day15_addendum.md <<'EOT'

## 14. Exploratory addition (written 9 Nov 2026 after the observer results were seen): memoryless relays under observer errors
Suite d15_obs_nomem runs relay K2 and relayD K2 (no register memory) on the d15_obs conditions, seeds 110000-110024, to test the O1 sentence left untested in section 12b. Memoryless relays now read the imperfect observer command; behaviour is unchanged when the keys are unset. Exploratory, not confirmatory.
EOT
git add src/d12_common.py src/d15_suites.py 01_protocol/protocol_day15_addendum.md; git commit -m "Day 15: memoryless relay under observer errors, specified before running"; git push
(cd src && python d12_eval.py d15_obs_nomem 2>&1 | tee ../results/log_d15_obs_nomem.txt)
python - <<'PY'
import pandas as pd
d = pd.read_csv('results/d12_d15_obs_nomem.csv')
d['cond'] = d.apply(lambda r: 'contra' if r.contra else f'miss {r.miss}' if r.miss>0 else f'wrong {r.wrong}' if r.wrong>0 else 'clean', axis=1)
g = d.groupby(['cond','policy','obs_pct']).agg(n=('success','size'), success=('success','mean'), contact=('collision','mean')).round(3)
print(g.to_string()); g.to_csv('results/d15_Onomem_descriptive.csv')
PY
echo "== tables"
cat >> src/d15_tables.py <<'EOT'

# ---- added 9 Nov: observers, delay filter, probe, failure classes, confirmatory B and D
for n, st in (("d15_Bconf_descriptive.csv","t15_10_Bconf_descriptive"), ("d15_O_descriptive.csv","t15_12_observers"), ("d15_Onomem_descriptive.csv","t15_13_observers_nomem_EXPLORATORY"),
              ("d15_P_quick.csv","t15_14_delay_filter"), ("d15_probe.csv","t15_15_probe_EXPLORATORY"), ("d15_failure_classes.csv","t15_16_failure_classes_EXPLORATORY")):
    x = load(n)
    if x is not None: save(x, st)
x = load("d15_Bconf_paired.csv")
if x is not None: save(x[[c for c in PC if c in x]], "t15_11_Bconf_paired")
x = load("d12_d15_change_conf_summary.csv")
if x is not None: save(x, "t15_17_change_confirmatory")
EOT
python src/d15_tables.py
python src/check_claims.py || echo "claims check has failures or rows to add"
cat >> README.md <<'EOT'

## 4. Studies added on 8-9 Nov (protocol sections 9-14)
| Study | Suite / script | Seeds | Status |
|---|---|---|---|
| D replication | `d15_change_conf` | 90000-90024 | confirmatory |
| B replication | `d15_filter_dist_conf` | 100000-100024 | confirmatory |
| Observer errors | `d15_obs`, `d15_obs_nomem` | 110000-110024 | confirmatory / exploratory (nomem) |
| Delay-aware filter | `d15_pred_conf`, `d15_pred.py` | 120000-120024 | confirmatory (pad chosen on dev seeds 5000-5009) |
| Failure classification, learned probe | `d15_classify.py`, `d15_probe.py` | earlier runs | exploratory |
Stats: `d15_stats_Bconf.py`, `d15_stats_O.py`. Tables: `python src/d15_tables.py`.
EOT
git add src/d15_tables.py README.md results/d12_d15_obs_nomem.csv results/d15_Onomem_descriptive.csv results/log_d15_obs_nomem.txt results/log_overnight.txt paper/tables/t15_* paper/figs/f15_*.pdf
git commit -m "Day 15: tables, README and exploratory no-memory observer run"; git pull --rebase --autostash; git push
echo "OVERNIGHT DONE"
