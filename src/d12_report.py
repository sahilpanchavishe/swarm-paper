"""Day 12: summary table for any d11 results CSV (all numbers from the raw episode rows).
  python d12_report.py ../results/d12_confirm.csv            # writes <csv stem>_summary.csv next to it and prints it
Columns: success %, collision-free success % (success AND no contact), contact %, near-contact steps (clearance < 0.09 m, mean per episode),
mean and 95th-percentile of the MAX per-robot final error, median minimum clearance, plus a 95 % interval for success that resamples BOTH
test scenarios (formation, seed) and training seeds (two-level bootstrap), so it is not the over-narrow pooled Wilson interval.
One row per policy x observability x condition. Contact is measured at centre distance < 0.074 m (the physics engine does not let robots go much below that)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

COND = ["start_half", "loss", "delay", "noise", "safety", "dsafe", "gamma", "change"]

def boot_ci(x, nboot=2000, seed=0):
    """x: DataFrame with columns formation, seed, train_seed, value. Two-level bootstrap of the mean."""
    W = x.pivot_table(index=["formation", "seed"], columns=x.train_seed.fillna(-1), values="value", aggfunc="mean").values.astype(float)
    rng = np.random.default_rng(seed); n, m = W.shape; out = np.empty(nboot)
    for r in range(nboot): out[r] = W[rng.integers(0, n, n)][:, rng.integers(0, m, m)].mean()
    return np.percentile(out, [2.5, 97.5])

def summarize(d):
    d = d.copy(); d["cfs"] = d.success & ~d.collision; keys = ["policy", "obs_pct"] + [c for c in COND if c in d]; rows = []
    for k, x in d.groupby(keys, dropna=False):
        lo, hi = boot_ci(x.assign(value=x.success.astype(float)))
        rows.append(dict(zip(keys, k), episodes=len(x), models=x.train_seed.nunique() if x.train_seed.notna().any() else 0, success_pct=100 * x.success.mean(),
                         ci_lo=100 * lo, ci_hi=100 * hi, collision_free_success_pct=100 * x.cfs.mean(), contact_pct=100 * x.collision.mean(),
                         near_steps_mean=x.near_steps.mean(), mean_final_err_mm=1000 * x.final_err.mean(), max_err_mean_mm=1000 * x.max_err.mean(),
                         max_err_p95_mm=1000 * x.max_err.quantile(0.95), median_min_clearance_mm=1000 * x.min_clearance.median()))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    p = Path(sys.argv[1]); d = pd.read_csv(p); T = summarize(d)
    T.to_csv(p.with_name(p.stem + "_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
    print(T.round(2).to_string(index=False)); print(f"\nsaved {p.with_name(p.stem + '_summary.csv')}")
