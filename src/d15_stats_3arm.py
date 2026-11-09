"""Section 16: matched three-arm delay study. Arm A = no filter (d15_arm_nofilter), arm B = filter on delayed positions (d15_arm_stale),
arm C = predictor filter (d15_pred_conf). Same seeds 120000-120024, model seeds, policies, delays, observability.
Primary metric: collision-free success (cf). Scores T1-T5 against the thresholds below; THESE MUST MATCH THE TEXT YOU COMMITTED in protocol section 16.
Paired comparisons: joint bootstrap over scenarios and training seeds (same seed indices when both arms have the same number), two-sided p, Holm over the whole study.
Run from the repo root:  python src/d15_stats_3arm.py [draws]   (default 10000 draws)"""
import sys
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
# ---- thresholds from protocol section 16 (edit here if your committed wording differs)
T1 = dict(contact_min_gnn=40.0, contact_max_latched=20.0, contact_max_relay=5.0)
T2 = dict(contact_min=50.0, cf_gap_max=10.0)
T3 = dict(cf_gain_min=50.0)
T4 = dict(cf_gain_gnn=30.0, relay_band=10.0)
T4_GNN = ["GNN K2", "GNN K3"]   # memoryless GNNs only if protocol section 16a is committed; use GNN (all three) if you keep T4 as originally written
T5 = dict(success_cost_max=10.0)
GNN = ["GNN K2", "GNN K2+latched", "GNN K3"]; ALL = GNN + ["relayD K2+mem"]


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


files = dict(A="d12_d15_arm_nofilter.csv", B="d12_d15_arm_stale.csv", C="d12_d15_pred_conf.csv")
frames = []
for arm, f in files.items():
    x = pd.read_csv(ROOT / "results" / f); x["arm"] = arm; frames.append(x)
d = pd.concat(frames, ignore_index=True)
d["success"] = d.success.astype(bool); d["collision"] = d.collision.astype(bool); d["cf"] = d.success & ~d.collision
cnt = d.groupby(["arm", "policy", "delay", "obs_pct"]).size().unstack("arm")
assert not cnt.isna().any().any(), "an arm is missing a cell (unfinished run?)\n" + str(cnt[cnt.isna().any(axis=1)])
assert (cnt.nunique(axis=1) == 1).all(), "arms have different episode counts in some cells\n" + str(cnt)

agg = dict(n=("cf", "size"), success=("success", "mean"), contact=("collision", "mean"), cf=("cf", "mean"))
if "min_clearance" in d: agg["min_clr_median_mm"] = ("min_clearance", lambda s: 1000 * s.median())
if "final_err" in d: agg["final_err_median_mm"] = ("final_err", lambda s: 1000 * s.median())
desc = d.groupby(["arm", "policy", "delay", "obs_pct"]).agg(**agg).reset_index()
for c in ("success", "contact", "cf"): desc[c] = 100 * desc[c]
desc.to_csv(ROOT / "results" / "d15_3arm_descriptive.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
print("=== descriptive (percent)"); print(desc.round(1).to_string(index=False))

rng = np.random.default_rng(0); rows = []
for (pol, dl, o), x in d.groupby(["policy", "delay", "obs_pct"]):
    for metric in ("cf", "success"):
        for a, b in (("B", "A"), ("C", "B"), ("C", "A")):
            dd, lo, hi, p = boot(mat(x[x.arm == a], metric), mat(x[x.arm == b], metric), rng)
            rows.append(dict(policy=pol, delay=dl, obs_pct=o, metric=metric, comparison=f"{a} vs {b}", d_pct=dd, ci_lo=lo, ci_hi=hi, p_boot=p))
P = pd.DataFrame(rows); P["p_holm"] = holm(P.p_boot); P.to_csv(ROOT / "results" / "d15_3arm_paired.csv", index=False)
print(f"\n=== paired: {len(P)} comparisons, {B} draws, Holm<0.05: {(P.p_holm < 0.05).sum()} (p floor 1/{B}; report p as <= the printed Holm value)")
print(P[P.metric == "cf"].round(3).to_string(index=False))


def cell(arm, pol, dl, o, col): return desc[(desc.arm == arm) & (desc.policy == pol) & (desc.delay == dl) & (desc.obs_pct == o)][col].iloc[0]
def pair(pol, dl, o, metric, cmp_): return P[(P.policy == pol) & (P.delay == dl) & (P.obs_pct == o) & (P.metric == metric) & (P.comparison == cmp_)].iloc[0]

print("\n=== predictions (arm A and B are fresh; arm C was seen before section 16 was written)")
cells = [(dl, o) for dl in (5, 10) for o in (25, 50)]
bad = [(p, dl, o, round(cell("A", p, dl, o, "contact"), 1)) for p in ("GNN K2", "GNN K3") for dl, o in cells if cell("A", p, dl, o, "contact") < T1["contact_min_gnn"]]
bad += [(p, dl, o, round(cell("A", p, dl, o, "contact"), 1)) for p in ("GNN K2+latched",) for dl, o in cells if cell("A", p, dl, o, "contact") >= T1["contact_max_latched"]]
bad += [(p, dl, o, round(cell("A", p, dl, o, "contact"), 1)) for p in ("relayD K2+mem",) for dl, o in cells if cell("A", p, dl, o, "contact") > T1["contact_max_relay"]]
print("T1", "PASS" if not bad else f"FAIL {bad}")
bad = [(p, o, round(cell("B", p, 10, o, "contact"), 1)) for p in ALL for o in (25, 50) if cell("B", p, 10, o, "contact") < T2["contact_min"]]
gap = [(o, round(abs(cell("B", "GNN K2", 10, o, "cf") - cell("A", "GNN K2", 10, o, "cf")), 1)) for o in (25, 50)]
bad2 = [g for g in gap if g[1] > T2["cf_gap_max"]]
print("T2", "PASS" if not bad and not bad2 else f"FAIL contact below {T2['contact_min']}: {bad}; |cf B-A| for GNN K2 delay 10 above {T2['cf_gap_max']}: {bad2}", "| gaps", gap)
res = [(p, o, round(r.d_pct, 1), round(r.p_holm, 4)) for p in GNN for o in (25, 50) for r in [pair(p, 10, o, "cf", "C vs B")]]
bad = [r for r in res if r[2] < T3["cf_gain_min"] or r[3] >= 0.05]
print("T3", "PASS" if not bad else f"FAIL {bad}", "| all:", res)
res = [(p, dl, o, round(pair(p, dl, o, "cf", "C vs A").d_pct, 1)) for p in T4_GNN for dl, o in cells]
bad = [r for r in res if r[3] < T4["cf_gain_gnn"]]
rel = [(dl, o, round(pair("relayD K2+mem", dl, o, "cf", "C vs A").d_pct, 1)) for dl, o in cells]
bad2 = [r for r in rel if abs(r[2]) > T4["relay_band"]]
print("   (not scored in T4, C vs A cf:", [(p, dl, o, round(pair(p, dl, o, "cf", "C vs A").d_pct, 1)) for p in GNN if p not in T4_GNN for dl, o in cells], ")")
print("T4", "PASS" if not bad and not bad2 else f"FAIL GNN gains below {T4['cf_gain_gnn']}: {bad}; relayD outside +-{T4['relay_band']}: {bad2}", "| relayD C-A:", rel)
res = [(p, dl, o, round(cell("A", p, dl, o, "success") - cell("C", p, dl, o, "success"), 1)) for p in ALL for dl, o in cells]
bad = [r for r in res if r[3] > T5["success_cost_max"]]
print("T5", "PASS" if not bad else f"FAIL {bad}", "| max cost:", max(r[3] for r in res))
