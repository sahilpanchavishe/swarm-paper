#!/bin/bash
# Overnight queue, batch 2 (section 17: provenance and confidence). Start it now; it waits for batch 1 (results/.overnight2_done).
#   caffeinate -i src/run_overnight3.sh
# Needs in src/: patch_d17.py, d17_fusion.py, d17_stats.py (this script is src/run_overnight3.sh).
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
L=results/log_overnight3.txt; exec > >(tee -a "$L") 2>&1
echo "== batch 2 started $(date); waiting for batch 1"
while [ ! -f results/.overnight2_done ]; do sleep 60; done
echo "== batch 1 finished, continuing $(date)"

echo "== gates"
for f in src/patch_d17.py src/d17_fusion.py src/d17_stats.py; do [ -f "$f" ] || { echo "GATE FAILED: $f missing"; exit 1; }; done
if [ "$(grep -c 'TODO(you)' 01_protocol/protocol_day15_addendum.md)" != "0" ]; then echo "GATE FAILED: TODO(you) in protocol"; exit 1; fi
git log --oneline | grep -q "provenance and confidence study" || { echo "GATE FAILED: section 17 commit not found"; exit 1; }
git pull --rebase --autostash -q || { echo "GATE FAILED: pull failed"; exit 1; }

echo "== patch and checks"
python src/patch_d17.py || { echo "PATCH FAILED, nothing changed"; exit 1; }
python -m py_compile src/d12_common.py src/d15_suites.py src/d17_fusion.py || { echo "COMPILE FAILED"; git checkout -- src/d12_common.py src/d15_suites.py; exit 1; }
(cd src && python d17_fusion.py) | tee /tmp/d17_selftest.txt; grep -q PASS /tmp/d17_selftest.txt || { echo "GATE FAILED: fusion self-test"; exit 1; }
for s in d17_relay d17_gnn; do
  pl=$(cd src && python d12_eval.py $s --plan 2>&1 | tail -1); echo "$pl"
  echo "$pl" | grep -q "12 units, 0 already done" || { echo "GATE FAILED: plan for $s is not 12 units, 0 done"; exit 1; }
done

echo "== regression smoke (the first-token relay with the new code must reproduce section 15 results on the same scenarios)"
rm -f /tmp/d17_smoke.csv
(cd src && python d12_eval.py d17_relay --n 3 --seed0 130000 --workers 2 --out /tmp/d17_smoke.csv 2>&1 | tail -2)
python - <<'PY'
import sys
import pandas as pd
d = pd.read_csv('/tmp/d17_smoke.csv'); r = pd.read_csv('results/d12_d15_obs_safe.csv')
a = d[(d.fusion == 'first') & (~d.contra) & (d.policy == 'relay K2+mem')]
b = r[(r.policy == 'relay K2+mem') & (r.safety == True) & (r.miss == 0) & (r.wrong == 0) & (~r.contra)]
j = a.merge(b[['formation', 'seed', 'obs_pct', 'final_err']], on=['formation', 'seed', 'obs_pct'], suffixes=('', '_ref'))
m = round(1000 * (j.final_err - j.final_err_ref).abs().median(), 4) if len(j) else None
print("matched rows:", len(j), "| median |d final_err| mm:", m)
print((d.assign(cf=d.success & ~d.collision).groupby(['fusion', 'contra', 'conf_bad']).cf.mean() * 100).round(0).to_string())
sys.exit(0 if len(j) > 0 and m is not None and m < 0.001 else 1)
PY
[ $? -eq 0 ] || { echo "GATE FAILED: regression check (new code changed old behaviour). Reverting src edits."; git checkout -- src/d12_common.py src/d15_suites.py; exit 1; }

echo "== commit the code before running"
git add src/d12_common.py src/d15_suites.py src/d17_fusion.py src/d17_stats.py src/patch_d17.py src/run_overnight3.sh
git commit -m "Day 15: code for section 17 (count-by-source and max-confidence relays), committed before the run" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01P4ASwddA8Kn1WewzqMdH1H"
git push || echo "PUSH FAILED: the commit exists locally with its timestamp; push it in the morning"

echo "== runs"
for s in d17_relay d17_gnn; do
  echo "-- $s $(date)"; (cd src && python d12_eval.py $s 2>&1 | tee -a ../results/log_$s.txt | tail -3)
  f=results/d12_$s.csv; [ -f "$f" ] && echo "$s rows: $(( $(wc -l < $f) - 1 )) (expected 1200)" || echo "$s: NO CSV"
done

echo "== statistics (V1-V3 scored at the end)"
python src/d17_stats.py 10000 2>&1 | tee results/log_stats_d17.txt | tail -40

echo "== commit results (explicit paths)"
for f in results/d12_d17_relay.csv results/d12_d17_gnn.csv results/log_d17_relay.txt results/log_d17_gnn.txt results/d17_descriptive.csv results/d17_paired.csv results/log_stats_d17.txt; do
  [ -e "$f" ] && git add -- "$f"
done
git status --short
git commit -m "Day 15: provenance and confidence study results (section 17, V1-V3)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01P4ASwddA8Kn1WewzqMdH1H"
git pull --rebase --autostash && git push
echo "== OVERNIGHT3 DONE $(date)"; touch results/.overnight3_done
