"""Study D (command change, suite d15_change_conf): paired comparisons, joint-seed two-level bootstrap, Holm per metric.
Metrics: success (as in the study) and collision-free success (post hoc, exploratory). Conditions: safety flag x observability."""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = 4000


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def mat(x, metric):
    x = x.assign(key=x.formation.astype(str) + "|" + x.seed.astype(str), ts=x.train_seed.fillna(-1).astype(float))
    return x.pivot_table(index="key", columns="ts", values=metric, aggfunc="mean")


def boot(A, Bm, rng):
    idx = A.index.intersection(Bm.index)
    A, Bm = A.loc[idx].values, Bm.loc[idx].values
    n, ka, kb = len(idx), A.shape[1], Bm.shape[1]
    d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n); ja = rng.integers(0, ka, ka); jb = ja if kb == ka else rng.integers(0, kb, kb)
        d[b] = A[i][:, ja].mean() - Bm[i][:, jb].mean()
    p = min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / B))
    return 100 * (A.mean() - Bm.mean()), 100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5), p, len(idx)


PAIRS = [("GNN K2+latched/ver", "relay K2+ver"), ("GNN K2+latched/ver", "GNN K2"), ("GNN K2", "relay K2+ver"),
         ("GNN K2+latched/ver", "GNN K2+latched/naive"), ("GNN K3", "GNN K2"), ("relay K3+ver", "relay K2+ver"),
         ("GNN K2", "GNN K1")]

d = pd.read_csv(ROOT / "results" / "d12_d15_change_conf.csv")
d["success"] = d.success.astype(bool)
d["cf"] = d.success & ~d.collision.astype(bool)
rng = np.random.default_rng(0)
for metric, out in (("success", "d15_Dfinal_paired.csv"), ("cf", "d15_Dcf_paired.csv")):
    print("\n=== metric:", metric)
    print((d.groupby(["safety", "obs_pct", "policy"])[metric].mean() * 100).round(1).unstack("policy").to_string())
    rows = []
    for (s, o), x in d.groupby(["safety", "obs_pct"]):
        pols = set(x.policy)
        for a, b in PAIRS:
            if a in pols and b in pols:
                dd, lo, hi, p, n = boot(mat(x[x.policy == a], metric), mat(x[x.policy == b], metric), rng)
                rows.append(dict(safety=s, obs_pct=o, comparison=f"{a} vs {b}", n_scen=n, d_pct=dd, ci_lo=lo, ci_hi=hi, p_boot=p))
    R = pd.DataFrame(rows); R["p_holm"] = holm(R.p_boot)
    R.to_csv(ROOT / "results" / out, index=False)
    print(out, len(R), "comparisons; Holm<0.05:", int((R.p_holm < 0.05).sum()))
    print(R.round(2).to_string(index=False))
