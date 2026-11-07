"""Day 15 Study C statistics (descriptive): success after token removal, with the position-only (nearest-template) readout. Run: python src/d15_stats_C.py [csv ...]"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
files = sys.argv[1:] or sorted(glob.glob(str(ROOT / "results" / "d15_cutoff_*.csv")))
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
d["policy"] = d.apply(lambda r: f"GNN K{r.K}" + ("+latched" if r.latched else ""), axis=1)
rows = []
for (pol, var, tc), x in d.groupby(["policy", "variant", "t_cut"]):
    ps = x.groupby("train_seed").success.mean() * 100; c = x.near_cut_ok
    rows.append(dict(policy=pol, variant=var, t_cut=tc, train_seeds=x.train_seed.nunique(), n=len(x), success_pct=100 * x.success.mean(), seed_min=ps.min(), seed_max=ps.max(),
                     err_at_cut_mm=1000 * x.err_at_cut.mean(), template_ok_at_cut_pct=100 * c.mean(),
                     success_if_template_ok=100 * x.success[c].mean() if c.any() else np.nan, success_if_template_wrong=100 * x.success[~c].mean() if (~c).any() else np.nan))
R = pd.DataFrame(rows); R.to_csv(ROOT / "results" / "d15_C_descriptive.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(R.round(1).to_string(index=False))
print("\nDefault-formation check (flagoff): which formation do runs end nearest to, and do failures go there?")
for pol, x in d[d.variant == "flagoff"].groupby("policy"):
    x0 = x[x.t_cut == 0]
    if len(x0) == 0: continue
    mode = x0.near_final.mode()[0]; fails = x[(x.t_cut > 0) & ~x.success]
    print(f"{pol}: at t_cut=0, {100 * (x0.near_final == mode).mean():.0f}% end nearest to '{mode}'; "
          f"of {len(fails)} failures at t_cut>0, {100 * (fails.near_final == mode).mean() if len(fails) else float('nan'):.0f}% end nearest to it")
