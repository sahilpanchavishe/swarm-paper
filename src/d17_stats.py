"""Section 17: provenance and confidence study. Reads results/d12_d17_relay.csv and results/d12_d17_gnn.csv, prints descriptive tables, paired comparisons
(joint bootstrap, Holm over the whole study) and scores V1-V3. Thresholds below MUST match the text committed in protocol section 17.
Run from the repo root:  python src/d17_stats.py [draws]"""
import sys
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
V1_MAX = 20.0; V2_MAXCONF_C2_MAX = 20.0; V2_MAXCONF_C3_MIN = 70.0; V2_MAXCONF_C1_MAX = 20.0; V3_PROV_MIN = 60.0; V3_PROV_C0_MIN = 95.0


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def mat(x, metric):
    x = x.assign(key=x.formation + "|" + x.seed.astype(str) + "|" + x.obs_pct.astype(str), ts=x.train_seed.fillna(-1))
    return x.pivot_table(index="key", columns="ts", values=metric, aggfunc="mean")


def boot(A, Bm, rng):
    idx = A.index.intersection(Bm.index); A, Bm = A.loc[idx].values, Bm.loc[idx].values
    n, ka, kb = len(idx), A.shape[1], Bm.shape[1]; d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n); ja = rng.integers(0, ka, ka); jb = ja if kb == ka else rng.integers(0, kb, kb)
        d[b] = A[i][:, ja].mean() - Bm[i][:, jb].mean()
    pt = A.mean() - Bm.mean(); p = min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / B))
    return 100 * pt, 100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5), p


d = pd.concat([pd.read_csv(ROOT / "results" / "d12_d17_relay.csv"), pd.read_csv(ROOT / "results" / "d12_d17_gnn.csv")], ignore_index=True)
d["success"] = d.success.astype(bool); d["collision"] = d.collision.astype(bool); d["cf"] = d.success & ~d.collision
d["contra"] = d.contra.astype(bool)
d["cond"] = np.where(~d.contra, "C0", np.where(d.conf_bad.round(2) == 0.9, "C2", np.where(d.conf_bad.round(2) == 0.3, "C3", "C1")))
d["rule"] = np.where(d.policy.str.startswith("GNN"), "GNN K2+latched", "relay " + d.fusion.map(dict(first="first-token", maxconf="max-confidence", prov="count-by-source")))
cnt = d.groupby(["rule", "cond"]).size().unstack(); assert not cnt.isna().any().any(), "missing cell (unfinished run?)\n" + str(cnt)

desc = d.groupby(["rule", "cond"]).agg(n=("cf", "size"), success=("success", "mean"), contact=("collision", "mean"), cf=("cf", "mean"),
                                       t_converge_median_s=("t_converge_s", "median")).reset_index()
for c in ("success", "contact", "cf"): desc[c] = 100 * desc[c]
desc.to_csv(ROOT / "results" / "d17_descriptive.csv", index=False)
pd.set_option("display.width", 200)
print("=== collision-free success (%)"); print(desc.pivot(index="rule", columns="cond", values="cf").round(1).to_string())
print("=== success (%)"); print(desc.pivot(index="rule", columns="cond", values="success").round(1).to_string())
print("=== contact (%)"); print(desc.pivot(index="rule", columns="cond", values="contact").round(1).to_string())

PAIRS = [("relay count-by-source", "relay first-token"), ("relay count-by-source", "relay max-confidence"), ("relay max-confidence", "relay first-token"),
         ("relay count-by-source", "GNN K2+latched")]
rng = np.random.default_rng(0); rows = []
for c in ("C0", "C1", "C2", "C3"):
    x = d[d.cond == c]
    for a, b in PAIRS:
        dd, lo, hi, p = boot(mat(x[x.rule == a], "cf"), mat(x[x.rule == b], "cf"), rng)
        rows.append(dict(cond=c, comparison=f"{a} vs {b}", d_cf_pct=dd, ci_lo=lo, ci_hi=hi, p_boot=p))
P = pd.DataFrame(rows); P["p_holm"] = holm(P.p_boot); P.to_csv(ROOT / "results" / "d17_paired.csv", index=False)
print(f"\n=== paired on collision-free success: {len(P)} comparisons, {B} draws, Holm<0.05: {(P.p_holm < 0.05).sum()} (p floor 1/{B})"); print(P.round(3).to_string(index=False))


def cf(rule, cond): return float(desc[(desc.rule == rule) & (desc.cond == cond)].cf.iloc[0])
print("\n=== predictions (section 17)")
v1 = [(r, c, round(cf(r, c), 1)) for r in ("relay first-token", "GNN K2+latched") for c in ("C1", "C2", "C3") if cf(r, c) > V1_MAX]
print("V1", "PASS" if not v1 else f"FAIL {v1}")
v2 = []
if cf("relay max-confidence", "C2") > V2_MAXCONF_C2_MAX: v2.append(("C2 too high", round(cf("relay max-confidence", "C2"), 1)))
if cf("relay max-confidence", "C3") < V2_MAXCONF_C3_MIN: v2.append(("C3 too low", round(cf("relay max-confidence", "C3"), 1)))
if cf("relay max-confidence", "C1") > V2_MAXCONF_C1_MAX: v2.append(("C1 too high", round(cf("relay max-confidence", "C1"), 1)))
print("V2", "PASS" if not v2 else f"FAIL {v2}")
v3 = [(c, round(cf("relay count-by-source", c), 1)) for c in ("C1", "C2", "C3") if cf("relay count-by-source", c) < V3_PROV_MIN]
if cf("relay count-by-source", "C0") < V3_PROV_C0_MIN: v3.append(("C0", round(cf("relay count-by-source", "C0"), 1)))
print("V3", "PASS" if not v3 else f"FAIL {v3}")
