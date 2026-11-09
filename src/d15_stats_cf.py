"""Day 15 addendum analysis (post hoc, exploratory): observer study re-analysed on collision-free success.
Same two-level joint-seed bootstrap and Holm correction as d15_stats_final.py. Primary (pre-specified) metric of the
observer study was formation success; this adds collision-free success because contact differs sharply between policies."""
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
    x = x.assign(key=x.formation + "|" + x.seed.astype(str) + "|" + x.obs_pct.astype(str), ts=x.train_seed.fillna(-1))
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
    return 100 * (A.mean() - Bm.mean()), 100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5), p


def run(d, pairs, metric, out):
    rng = np.random.default_rng(0); rows = []
    for c, xc in d.groupby("cond"):
        for o, xo in xc.groupby("obs_pct"):
            pols = set(xo.policy)
            for a, b in pairs:
                if a in pols and b in pols:
                    dd, lo, hi, p = boot(mat(xo[xo.policy == a], metric), mat(xo[xo.policy == b], metric), rng)
                    rows.append(dict(cond=c, obs_pct=o, comparison=f"{a} vs {b}", d_pct=dd, ci_lo=lo, ci_hi=hi, p_boot=p))
    R = pd.DataFrame(rows); R["p_holm"] = holm(R.p_boot)
    R.to_csv(ROOT / "results" / out, index=False)
    print(out, len(R), "comparisons; Holm<0.05:", int((R.p_holm < 0.05).sum()))
    print(R[R.p_holm < 0.05].round(2).to_string(index=False))


o = pd.read_csv(ROOT / "results" / "d12_d15_obs.csv")
o["cf"] = o.success.astype(bool) & ~o.collision.astype(bool)
o["cond"] = o.apply(lambda r: "contra" if r.contra else f"miss {r.miss}" if r.miss > 0
                    else f"wrong {r.wrong}" if r.wrong > 0 else "clean", axis=1)
PAIRS = [("GNN K2", "relayD K2+mem"), ("GNN K2+latched", "relayD K2+mem"), ("GNN K2+latched", "GNN K2"),
         ("relay K2+mem", "relayD K2+mem")]
print(o.groupby(["cond", "obs_pct", "policy"]).cf.mean().mul(100).round(1).unstack("policy").to_string())
run(o, PAIRS, "cf", "d15_Ocf_paired.csv")
