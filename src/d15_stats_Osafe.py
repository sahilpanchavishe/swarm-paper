"""Day 15 section 15: analysis of d15_obs_safe (matched filter off/on, randomised persistent observers).
Primary metric collision-free success (cf). Scores predictions S1-S5 exactly as committed in the addendum, then paired
comparisons (joint-seed two-level bootstrap, B=4000, Holm over the whole family)."""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = 4000

d = pd.read_csv(ROOT / "results" / "d12_d15_obs_safe.csv")
d["success"] = d.success.astype(bool); d["contact"] = d.collision.astype(bool); d["cf"] = d.success & ~d.contact


def lab(r):
    if r.contra and r.contra_n == 1: return "contra1"
    if r.contra: return "contraHalf"
    if r.miss > 0: return f"miss{r.miss}"
    if r.wrong > 0: return f"wrong{r.wrong}"
    return "clean"


d["cond"] = d.apply(lab, axis=1)
d["filt"] = d.safety.astype(bool)
print("rows", len(d), "| per (filter, cond, obs, policy) n =", d.groupby(["filt", "cond", "obs_pct", "policy"]).size().unique())

T = d.groupby(["filt", "cond", "obs_pct", "policy"])[["cf", "success", "contact"]].mean().mul(100).round(1)
for m in ("cf", "success", "contact"):
    print(f"\n=== {m} (%)"); print(T[m].unstack("policy").to_string())


def cf(filt, cond, obs, pol): return T.loc[(filt, cond, obs, pol), "cf"]


print("\n=== Predictions (filter on)")
pol = ["GNN K2", "GNN K2+latched", "relayD K2+mem", "relay K2+mem"]
s1 = [(c, o, p, T.loc[(True, c, o, p), "contact"]) for c in ("clean", "miss0.6", "wrong0.3") for o in (25, 50) for p in pol]
bad = [x for x in s1 if x[3] > 2]
print("S1 contact <=2% all policies, clean/miss0.6/wrong0.3:", "PASS" if not bad else f"FAIL {bad}")
s2 = [(o, cf(True, "wrong0.3", o, "GNN K2") - cf(True, "wrong0.3", o, p)) for o in (25, 50) for p in ("GNN K2+latched", "relay K2+mem")]
print("S2 K2 exceeds latched and relay K2+mem by >=20 at wrong0.3:", "PASS" if all(v >= 20 for _, v in s2) else "FAIL", s2)
s3m = [cf(True, "miss0.6", 25, p) for p in ("GNN K2+latched", "relayD K2+mem", "relay K2+mem")]; s3k = cf(True, "miss0.6", 25, "GNN K2")
print("S3 miss0.6@25: memory >=95 and K2 <=80:", "PASS" if min(s3m) >= 95 and s3k <= 80 else "FAIL", s3m, s3k)
s4 = max(cf(True, "contra1", 50, p) for p in pol)
print("S4 contra1@50: no policy reaches 50:", "PASS" if s4 < 50 else "FAIL", "max", s4)
s5 = max(cf(True, "contraHalf", o, p) for o in (25, 50) for p in pol)
print("S5 contraHalf: all <=10:", "PASS" if s5 <= 10 else "FAIL", "max", s5)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def mat(x, metric):
    x = x.assign(key=x.formation.astype(str) + "|" + x.seed.astype(str), ts=x.train_seed.fillna(-1).astype(float))
    return x.pivot_table(index="key", columns="ts", values=metric, aggfunc="mean")


def boot(A, Bm, rng):
    idx = A.index.intersection(Bm.index); A, Bm = A.loc[idx].values, Bm.loc[idx].values
    n, k = len(idx), A.shape[1]; dd = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n); ja = rng.integers(0, A.shape[1], A.shape[1]); jb = ja if Bm.shape[1] == A.shape[1] else rng.integers(0, Bm.shape[1], Bm.shape[1]); dd[b] = A[i][:, ja].mean() - Bm[i][:, jb].mean()
    p = min(1.0, max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1 / B))
    return 100 * (A.mean() - Bm.mean()), 100 * np.percentile(dd, 2.5), 100 * np.percentile(dd, 97.5), p


rng = np.random.default_rng(0); rows = []
PAIRS = [("GNN K2", "relayD K2+mem"), ("GNN K2+latched", "relayD K2+mem"), ("GNN K2+latched", "GNN K2"), ("relay K2+mem", "relayD K2+mem")]
for (c, o), x in d.groupby(["cond", "obs_pct"]):
    for f in (False, True):
        xf = x[x.filt == f]
        for a, b in PAIRS:
            r = boot(mat(xf[xf.policy == a], "cf"), mat(xf[xf.policy == b], "cf"), rng)
            rows.append(dict(cond=c, obs_pct=o, filter=f, comparison=f"{a} vs {b}", d_pct=r[0], ci_lo=r[1], ci_hi=r[2], p_boot=r[3]))
    for pn in pol:   # effect of the filter on identical scenarios
        xp = x[x.policy == pn]
        r = boot(mat(xp[xp.filt], "cf"), mat(xp[~xp.filt], "cf"), rng)
        rows.append(dict(cond=c, obs_pct=o, filter="on-off", comparison=f"{pn}: filter on vs off", d_pct=r[0], ci_lo=r[1], ci_hi=r[2], p_boot=r[3]))
R = pd.DataFrame(rows); R["p_holm"] = holm(R.p_boot)
R.to_csv(ROOT / "results" / "d15_Osafe_paired.csv", index=False)
print("\nd15_Osafe_paired.csv", len(R), "comparisons; Holm<0.05:", int((R.p_holm < 0.05).sum()))
print(R[R.p_holm < 0.05].round(2).to_string(index=False))
