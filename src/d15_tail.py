"""Day 15 (EXPLORATORY): who fails in the filtered wide box? Reads results/d12_widebox_safe.csv only."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(ROOT / "results" / "d12_widebox_safe.csv"); d = d[d.safety == True]; f = d[~d.success.astype(bool)]
pd.set_option("display.width", 220)
print(f"{len(f)} failing of {len(d)} episodes")
print(f.groupby(["policy", "obs_pct"]).agg(n_fail=("seed", "size"), share_not_all_informed=("informed_final", lambda x: (x < 1).mean()),
      median_init_comp=("init_comp", "median"), median_final_err_mm=("final_err", lambda x: 1000 * x.median()),
      median_max_err_mm=("max_err", lambda x: 1000 * x.median())).round(2).to_string())
print(f.groupby(["policy", "formation"]).size().unstack(fill_value=0).to_string())
g = f[f.policy.str.startswith("GNN")].groupby(["policy", "obs_pct", "seed", "formation"]).train_seed.nunique()
print("\nepisodes failing in k of the 3 training seeds (k=3 means the start itself is hard):"); print(g.groupby("policy").value_counts().sort_index().to_string())
