"""Day 15 Study B: plain safety filter under link loss, delay and noise (margins failed on dev, protocol 3b and 3c). Run: python src/d15_stats_B.py"""
import sys, glob
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6_analysis as A6
import d12_stats as S12
ROOT = S12.ROOT
fs = [f for f in sorted(glob.glob(str(ROOT / "results" / "*d15_filter_dist_conf.csv"))) if "summary" not in f and "paired" not in f and "descriptive" not in f]
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
d["cf"] = d.success.astype(bool) & ~d.collision.astype(bool)
d["cond"] = [f"loss {l}" if l > 0 else f"delay {int(dl)}" if dl > 0 else f"noise {int(round(1000 * n))} mm" for l, dl, n in zip(d.loss, d.delay, d.noise)]
desc = d.groupby(["cond", "obs_pct", "policy"]).agg(episodes=("cf", "size"), success_pct=("success", lambda x: 100 * x.mean()),
        contact_pct=("collision", lambda x: 100 * x.mean()), cf_success_pct=("cf", lambda x: 100 * x.mean()),
        median_clearance_mm=("min_clearance", lambda x: 1000 * x.median())).reset_index()
desc.to_csv(ROOT / "results" / "d15_Bconf_descriptive.csv", index=False)
d["success"] = d.cf
CMP = [("GNN K2 vs relayD K2+mem", "GNN K2", "relayD K2+mem"), ("latched GNN K2 vs relayD K2+mem", "GNN K2+latched", "relayD K2+mem"),
       ("GNN K3 vs GNN K2", "GNN K3", "GNN K2"), ("latched GNN K2 vs memoryless GNN K2", "GNN K2+latched", "GNN K2")]
rows = []
for cond in sorted(d.cond.unique()):
    for pct in (25, 50):
        for lab, a, b in CMP:
            r = S12.compare(d[d.cond == cond], f"{cond}: {lab} ({pct}% obs)", a, b, pct)
            if r: rows.append(r)
P = pd.DataFrame(rows); P["p_success_holm"] = A6.holm(P.p_success_max); P.to_csv(ROOT / "results" / "d15_Bconf_paired.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
print(desc.round(1).to_string(index=False)); print()
print(P[["comparison", "dsuccess_pct", "ci_lo", "ci_hi", "seeds_a_better", "p_success_max", "p_success_holm"]].round(4).to_string(index=False))
