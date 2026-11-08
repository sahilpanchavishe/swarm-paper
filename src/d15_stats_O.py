"""Day 15 observer study (O1-O3): success and contact by condition, policy and observability, with the range over training seeds. Run: python src/d15_stats_O.py [csv]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "d12_d15_obs.csv")
for c in ("miss", "wrong", "contra"):
    if c not in d: d[c] = np.nan
d["cond"] = d.apply(lambda r: "contra" if r.contra else f"miss {r.miss}" if r.miss > 0 else f"wrong {r.wrong}" if r.wrong > 0 else "clean", axis=1)
rows = []
for (c, pol, o), x in d.groupby(["cond", "policy", "obs_pct"]):
    ps = x.groupby(x.train_seed.fillna(-1)).success.mean() * 100
    rows.append(dict(cond=c, policy=pol, obs_pct=o, episodes=len(x), success_pct=100 * x.success.mean(), seed_min=ps.min(), seed_max=ps.max(), contact_pct=100 * x.collision.mean()))
R = pd.DataFrame(rows); R.to_csv(ROOT / "results" / "d15_O_descriptive.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200); print(R.round(1).to_string(index=False))
print("\nDrop from clean (points):")
cl = R[R.cond == "clean"].set_index(["policy", "obs_pct"]).success_pct
R["drop_from_clean"] = [cl[(p, o)] - s for p, o, s in zip(R.policy, R.obs_pct, R.success_pct)]
print(R[R.cond != "clean"].pivot_table(index=["cond", "obs_pct"], columns="policy", values="drop_from_clean").round(1).to_string())
