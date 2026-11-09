#!/bin/bash
# Overnight queue, batch 1. Run from anywhere:  caffeinate -i src/run_overnight2.sh
# Gates first (stops without running anything if the pre-registration is not in place), then: arms A and B, three-arm stats,
# higher-precision paired stats, tables, facts sheet, claims check, commit, release check (NO tag).
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
L=results/log_overnight2.txt; exec > >(tee -a "$L") 2>&1
echo "== start $(date)"

echo "== gates"
if [ "$(grep -c 'TODO(you)' 01_protocol/protocol_day15_addendum.md)" != "0" ]; then echo "GATE FAILED: TODO(you) still in the protocol. Fix, commit, push, restart."; exit 1; fi
git fetch -q
if [ -n "$(git log origin/main..HEAD --oneline)" ]; then echo "GATE FAILED: unpushed commits. Push first, restart."; exit 1; fi
git log --oneline | grep -q "three-arm delay study" || { echo "GATE FAILED: section 16 commit not found. Commit the pre-registration, restart."; exit 1; }
echo "gates passed"

echo "== arms A and B"
for s in d15_arm_nofilter d15_arm_stale; do
  echo "-- $s $(date)"; (cd src && python d12_eval.py $s 2>&1 | tee -a ../results/log_$s.txt | tail -3)
  f=results/d12_$s.csv; [ -f "$f" ] && echo "$s rows: $(( $(wc -l < $f) - 1 )) (expected 4000)" || echo "$s: NO CSV"
done

echo "== three-arm statistics (T1-T5 scored at the end)"
python src/d15_stats_3arm.py 10000 2>&1 | tee results/log_stats_3arm.txt | tail -14

echo "== paired statistics again with 20000 draws (overwrites the 4000-draw CSVs; use the new numbers in the text)"
for sc in d15_stats_Osafe d15_stats_final d15_stats_D; do
  if grep -q "B = 4000" src/$sc.py; then
    sed 's/B = 4000/B = 20000/' src/$sc.py > src/_hi_$sc.py
    python src/_hi_$sc.py > results/log_hi_$sc.txt 2>&1 && echo "$sc: rerun ok" || echo "$sc: rerun FAILED (see results/log_hi_$sc.txt)"
    rm -f src/_hi_$sc.py
  else echo "$sc: 'B = 4000' not found, skipped"; fi
done

echo "== tables and facts sheet"
if ! grep -q "t15_23_three_arm" src/d15_tables.py; then
cat >> src/d15_tables.py <<'EOT'

# ---- added 10 Nov: safety-on observers, Study D paired, three-arm delay study
for n, st in (("d15_Osafe_paired.csv", "t15_20_obs_safe_paired"), ("d15_Dfinal_paired.csv", "t15_21_D_paired"), ("d15_Dcf_paired.csv", "t15_22_D_cf_paired_POSTHOC"),
              ("d15_3arm_descriptive.csv", "t15_23_three_arm"), ("d15_3arm_paired.csv", "t15_24_three_arm_paired")):
    x = load(n)
    if x is not None: save(x, st)
EOT
fi
python src/d15_tables.py 2>&1 | tail -4
python - <<'PY'
import glob, pathlib
out = ["# Facts sheet (generated from paper/tables/t15_*.md; write the prose yourself)\n"]
for f in sorted(glob.glob("paper/tables/t15_*.md")): out += [f"\n## {pathlib.Path(f).stem}\n", open(f).read()]
open("paper/facts_d15.md", "w").write("\n".join(out)); print("facts sheet rebuilt")
PY
python src/check_claims.py 2>&1 | tail -3

echo "== commit (explicit paths only)"
for f in results/d12_d15_arm_nofilter.csv results/d12_d15_arm_stale.csv results/log_d15_arm_nofilter.txt results/log_d15_arm_stale.txt \
         results/d15_3arm_descriptive.csv results/d15_3arm_paired.csv results/log_stats_3arm.txt results/d15_Osafe*.csv results/d15_Dfinal_paired.csv \
         results/d15_Dcf_paired.csv results/d15_Pfinal_paired.csv results/d15_Ofinal_paired.csv results/log_hi_*.txt src/d15_tables.py paper/facts_d15.md paper/tables/t15_*; do
  [ -e "$f" ] && git add -- "$f"
done
git status --short
git commit -m "Day 15: matched three-arm delay study results, higher-precision paired statistics, tables" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01P4ASwddA8Kn1WewzqMdH1H"
git pull --rebase --autostash && git push

echo "== release check (does not tag)"
src/release_check.sh 2>&1 | tail -8
echo "== OVERNIGHT2 DONE $(date)"; touch results/.overnight2_done
