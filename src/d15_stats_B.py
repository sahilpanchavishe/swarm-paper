"""Day 15 Study B: safety filter under link loss, delay and noise; margin rule vs plain filter. Run: python src/d15_stats_B.py"""
import sys, glob
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6_analysis as A6
import d12_stats as S12
ROOT = S12.ROOT
fs = [f for f in sorted(glob.glob(str(ROOT / "results" / "*d15_filter_dist*.csv"))) if "summary" not in f and "paired" not in f and "descriptive" not in f]
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
d["cf"] = d.success.astype(bool) & ~d.collision.astype(bool)
d["filt"] = ["margin" if x > 0.1001 else "plain" for x in d.dsafe]
desc = d.groupby(["loss", "delay", "noise", "filt", "obs_pct", "policy"]).agg(
    episodes=("cf", "size"), success_pct=("success", lambda x: 100 * x.mean()), contact_pct=("collision", lambda x: 100 * x.mean()),
    cf_success_pct=("cf", lambda x: 100 * x.mean()), median_clearance_mm=("min_clearance", lambda x: 1000 * x.median())).reset_index()
desc.to_csv(ROOT / "results" / "d15_B_descriptive.csv", index=False)
d["success"] = d.cf; d["label"] = d.policy; d["policy"] = d.policy + " +" + d.filt
rows = []
for col, val in (("delay", 5), ("delay", 10), ("noise", 0.005), ("noise", 0.01)):
    sub = d[d[col] == val]
    for pol in ("GNN K2", "GNN K3", "GNN K2+latched", "relayD K2+mem"):
        for pct in (25, 50):
            r = S12.compare(sub, f"margin vs plain filter: {pol}, {col} {val} ({pct}% obs)", pol + " +margin", pol + " +plain", pct)
            if r: rows.append(r)
P = pd.DataFrame(rows); P["p_success_holm"] = A6.holm(P.p_success_max)
P.to_csv(ROOT / "results" / "d15_B_paired.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
print(desc.round(1).to_string(index=False)); print()
print(P[["comparison", "dsuccess_pct", "ci_lo", "ci_hi", "seeds_a_better", "p_success_max", "p_success_holm"]].round(4).to_string(index=False))
