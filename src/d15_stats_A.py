"""Day 15 Study A: disconnected starts. Collision-free success, paired comparisons per scenario and observability, Holm over the whole family.
Run: python src/d15_stats_A.py   Writes results/d15_A_paired.csv and results/d15_A_descriptive.csv"""
import sys, glob
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6_analysis as A6
import d12_stats as S12
ROOT = S12.ROOT
fs = [f for f in sorted(glob.glob(str(ROOT / "results" / "*d15_A_*.csv"))) if "summary" not in f and "paired" not in f and "descriptive" not in f]
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
d["success"] = d.success.astype(bool) & ~d.collision.astype(bool)           # collision-free success (primary metric)
desc = d.groupby(["scenario", "obs_pct", "policy"]).agg(episodes=("success", "size"), cf_success_pct=("success", lambda x: 100 * x.mean()),
                                                        contact_pct=("collision", lambda x: 100 * x.mean()), informed_final=("informed_final", "mean")).reset_index()
desc.to_csv(ROOT / "results" / "d15_A_descriptive.csv", index=False)
CMP = [("GNN K2 vs relayD K2+mem", "GNN K2", "relayD K2+mem"), ("latched GNN K2 vs relayD K2+mem", "GNN K2+latched", "relayD K2+mem"),
       ("relayD K2+mem vs hold-still relay K2+mem", "relayD K2+mem", "relay K2+mem"), ("GNN K2 vs memoryless relay K2", "GNN K2", "relay K2"),
       ("latched GNN K2 vs memoryless GNN K2", "GNN K2+latched", "GNN K2")]
rows = []
for sc in sorted(d.scenario.unique()):
    for pct in (25, 50):
        for lab, a, b in CMP:
            r = S12.compare(d[d.scenario == sc], f"{sc}: {lab} ({pct}% obs)", a, b, pct)
            if r: rows.append(r)
P = pd.DataFrame(rows); P["p_success_holm"] = A6.holm(P.p_success_max)
P.to_csv(ROOT / "results" / "d15_A_paired.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(desc.round(1).to_string(index=False)); print()
print(P[["comparison", "dsuccess_pct", "ci_lo", "ci_hi", "seeds_a_better", "p_success_max", "p_success_holm"]].round(4).to_string(index=False))
