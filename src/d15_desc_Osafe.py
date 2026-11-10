"""Descriptive table for the section 15 safety-on observer study, from the raw per-run file.
Run from the repo root:  python src/d15_desc_Osafe.py
Writes results/d15_Osafe_descriptive.csv: per (safety, miss, wrong, contra, contra_n, contra_rand, obs_pct, policy) cell, percent of runs with
success, contact (contact_steps > 0) and collision-free success (success and contact_steps == 0). This definition reproduces the paired CSV exactly."""
import pandas as pd

d = pd.read_csv("results/d12_d15_obs_safe.csv")
ok = d.success.astype(bool)
d["success_pct"] = 100 * ok
d["contact_pct"] = 100 * (d.contact_steps > 0)
d["cf_pct"] = 100 * (ok & (d.contact_steps == 0))
keys = ["safety", "miss", "wrong", "contra", "contra_n", "contra_rand", "obs_pct", "policy"]
g = (d.groupby(keys, dropna=False)
       .agg(n=("success", "size"), success_pct=("success_pct", "mean"), contact_pct=("contact_pct", "mean"), cf_pct=("cf_pct", "mean"))
       .round(3).reset_index())
g.to_csv("results/d15_Osafe_descriptive.csv", index=False)
print("wrote results/d15_Osafe_descriptive.csv,", len(g), "cells")
