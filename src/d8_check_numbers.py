"""Day 8: recompute the numbers quoted in paper/results_discussion_draft.md from the raw results and fail loudly if any differ.
Run after every change to the results:  python d8_check_numbers.py      (reads only ../results/*.csv)
If a check fails, fix the TEXT (never the data) and update the expected value here."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
S = pd.read_csv(R / "d6_table_success.csv"); P = pd.read_csv(R / "d6_table_paired.csv")
E = pd.read_csv(R / "d6_main_eval.csv"); B = pd.read_csv(R / "d3_baselines.csv"); A = pd.read_csv(R / "d7_token_ablation.csv")
T = pd.read_csv(R / "token_coverage_vs_K.csv"); T = T[np.isclose(T.start_half, 0.5)]
G = E[E.method == "DAgger"].copy(); G["pol"] = G.apply(lambda r: f"GNN K{r.K}" + ("+latched" if r.latched else ""), axis=1)
bad = []; n = 0

def chk(name, got, want, tol=0.051):
    global n; n += 1
    ok = abs(float(got) - float(want)) <= tol
    if not ok: bad.append(f"{name}: computed {float(got):.3f}, text says {want}")

def s(pol, pct, col="success_pct"): return S[(S.policy == pol) & (S.obs_pct == pct)].iloc[0][col]
def p(prefix, col): return P[P.comparison.str.startswith(prefix)].iloc[0][col]

# 3.1 reference points
chk("oracle success", s("oracle", 100), 100.0); chk("oracle CI lo", s("oracle", 100, "ci95_lo"), 98.1)
o = B[B.policy == "oracle"]; chk("oracle mean t_conv", o.t_converge_s.mean(), 8.7); chk("oracle max t_conv", o.t_converge_s.max(), 14.8)
for k, w25, w50 in ((1, 39.0, 85.5), (2, 85.5, 97.0), (3, 89.0, 98.5), (7, 89.5, 98.5)):
    chk(f"relay K{k} 25", s(f"relay K{k}", 25), w25); chk(f"relay K{k} 50", s(f"relay K{k}", 50), w50)
for k in (1, 2, 3):
    chk(f"relay K{k}+mem 25", s(f"relay K{k}+mem", 25), 100.0); chk(f"relay K{k}+mem 50", s(f"relay K{k}+mem", 50), 99.5)
chk("relay K0 25", s("relay K0", 25), 0.0); chk("relay K0 50", s("relay K0", 50), 0.0)
for obs, k, mean, allr in ((2, 1, 76.3, 12.8), (2, 2, 92.8, 65.0), (2, 3, 96.0, 80.8), (2, 6, 96.7, None),
                           (4, 1, 94.6, 64.6), (4, 2, 98.3, 88.6), (4, 3, 98.5, 90.4)):
    r = T[(T.observers == obs) & (T.K == k)].iloc[0]; chk(f"coverage {obs} obs K{k}", r.mean_pct_reached, mean)
    if allr is not None: chk(f"all reached {obs} obs K{k}", r.pct_episodes_all_reached, allr)

# 3.2 primary result
chk("K2 25", s("GNN K2", 25), 97.5); chk("K2 25 lo", s("GNN K2", 25, "ci95_lo"), 95.9); chk("K2 25 hi", s("GNN K2", 25, "ci95_hi"), 98.5)
chk("K1 25", s("GNN K1", 25), 72.3); chk("K1 25 lo", s("GNN K1", 25, "ci95_lo"), 68.6); chk("K1 25 hi", s("GNN K1", 25, "ci95_hi"), 75.8)
chk("primary diff", p("PRIMARY", "dsuccess_pct"), 25.2); chk("primary lo", p("PRIMARY", "dsuccess_ci_lo"), 20.8); chk("primary hi", p("PRIMARY", "dsuccess_ci_hi"), 29.7)
chk("primary holm p<0.0001", p("PRIMARY", "p_success_holm") < 1e-4, 1, 0); chk("primary final err mm", -1000 * p("PRIMARY", "dfinal_err_m"), 22.0)
ps = 100 * G.groupby(["pol", "obs_pct", "train_seed"]).success.mean()
for sd, w in ((0, 75.5), (1, 66.5), (2, 75.0)): chk(f"K1 seed{sd} 25", ps[("GNN K1", 25, sd)], w)
for sd in (0, 1, 2): chk(f"K2 seed{sd} 25", ps[("GNN K2", 25, sd)], 97.5)
chk("K1 seed range", ps[("GNN K1", 25, 0)] - ps[("GNN K1", 25, 1)], 9.0)
chk("K2 vs K1 50 diff", p("K=2 vs K=1 (50% obs)", "dsuccess_pct"), 19.2); chk("K2 vs K1 50 lo", p("K=2 vs K=1 (50% obs)", "dsuccess_ci_lo"), 15.3)
chk("K2 vs K1 50 hi", p("K=2 vs K=1 (50% obs)", "dsuccess_ci_hi"), 23.2); chk("K2 50", s("GNN K2", 50), 98.2); chk("K1 50", s("GNN K1", 50), 79.0)
chk("K0 25", s("GNN K0", 25), 0.0); chk("K0 50", s("GNN K0", 50), 0.2)

# 3.3 capacity
chk("K1 100", s("GNN K1", 100), 83.5); chk("K1 100 lo", s("GNN K1", 100, "ci95_lo"), 80.3); chk("K1 100 hi", s("GNN K1", 100, "ci95_hi"), 86.3)
chk("K2 100", s("GNN K2", 100), 99.5); chk("K3 100", s("GNN K3", 100), 99.8)
c = "capacity control: K=2 vs K=1 (100% obs)"
chk("cap diff", p(c, "dsuccess_pct"), 16.0); chk("cap lo", p(c, "dsuccess_ci_lo"), 12.7); chk("cap hi", p(c, "dsuccess_ci_hi"), 19.5)
chk("residual ~9", p("PRIMARY", "dsuccess_pct") - p(c, "dsuccess_pct"), 9.2)

# 3.4 GNN vs relay
for lab, d, lo, hi in (("GNN vs memoryless relay, K=2 (25", 12.0, 7.2, 17.3), ("GNN vs memoryless relay, K=3 (25", 10.0, 5.8, 14.7),
                       ("GNN vs memoryless relay, K=1 (25", 33.3, 25.2, 41.3), ("GNN vs memoryless relay, K=2 (50", 1.2, -1.3, 4.0),
                       ("GNN vs memoryless relay, K=3 (50", 1.3, -0.2, 3.3), ("GNN vs memoryless relay, K=1 (50", -6.5, -12.8, 0.0)):
    chk(lab + " diff", p(lab, "dsuccess_pct"), d); chk(lab + " lo", p(lab, "dsuccess_ci_lo"), lo); chk(lab + " hi", p(lab, "dsuccess_ci_hi"), hi)
chk("K2 vs relay 25 holm p(final err)=0.011", p("GNN vs memoryless relay, K=2 (25", "p_final_err_holm"), 0.011, 0.0006)
chk("K3 vs relay 25 p success<=0.0001", p("GNN vs memoryless relay, K=3 (25", "p_success_holm"), 0.0001, 0.00006)
chk("K3 vs relay 25 p final err 0.20", p("GNN vs memoryless relay, K=3 (25", "p_final_err_holm"), 0.20, 0.006)
chk("K1 vs relay 50 p success 0.22", p("GNN vs memoryless relay, K=1 (50", "p_success_holm"), 0.22, 0.006)
chk("K1 vs relay 50 p final 0.027", p("GNN vs memoryless relay, K=1 (50", "p_final_err_holm"), 0.027, 0.0006)
chk("latched vs relay mem 50 diff", p("latched GNN vs relay with memory, K=2 (50", "dsuccess_pct"), 0.5)
chk("latched 25", s("GNN K2+latched", 25), 100.0); chk("latched contact", s("GNN K2+latched", 25, "contact_pct"), 4.0)

# 3.5 ablation (seed 0, 80 episodes per cell)
a = A[A.obs_pct == 25].groupby(["model", "mode"]).agg(t=("success_true", "mean"), sh=("success_shown", "mean"), no=("nonobs_follow_shown", "mean")) * 100
for m, blank, norm, nonobs in (("d5_K1_s0", 1.2, 70.0, 89.8), ("d5_K2_s0", 2.5, 98.8, 99.6), ("d5_K3_s0", 8.8, 97.5, 99.0), ("d5_K2_latched_s0", 0.0, 100.0, 100.0)):
    chk(f"{m} blank", a.loc[(m, "blank"), "t"], blank); chk(f"{m} normal", a.loc[(m, "normal"), "t"], norm)
    chk(f"{m} wrong vs shown", a.loc[(m, "wrong"), "sh"], norm); chk(f"{m} wrong vs true", a.loc[(m, "wrong"), "t"], 0.0)
    chk(f"{m} nonobs follow", a.loc[(m, "wrong"), "no"], nonobs)
for m, w in (("d5_K1_s0", 22.1), ("d5_K2_s0", 21.2), ("d5_K3_s0", 24.8)): chk(f"{m} blank chance", a.loc[(m, "blank"), "no"], w)
a100 = A[A.obs_pct == 100].groupby(["model", "mode"]).success_shown.mean() * 100
chk("K1 100 wrong shown", a100[("d5_K1_s0", "wrong")], 86.2); chk("K2 100 wrong shown", a100[("d5_K2_s0", "wrong")], 100.0)
chk("n per cell", A.groupby(["model", "obs_pct", "mode"]).size().min(), 80, 0)

# 3.6 BC vs DAgger
Bc = E.copy(); Bc["pol"] = Bc.apply(lambda r: f"K{r.K}" + ("+latched" if r.latched else ""), axis=1)
tab = Bc.groupby(["pol", "method", "obs_pct"]).agg(s=("success", "mean"), c=("collision", "mean")) * 100
for pol, bc, dg in (("K1", 39.3, 72.3), ("K2", 75.0, 97.5), ("K3", 94.2, 99.0)):
    chk(f"{pol} BC 25", tab.loc[(pol, "BC", 25), "s"], bc); chk(f"{pol} DAgger 25", tab.loc[(pol, "DAgger", 25), "s"], dg)
chk("K2 contact BC", tab.loc[("K2", "BC", 25), "c"], 86.8); chk("K2 contact DAgger", tab.loc[("K2", "DAgger", 25), "c"], 51.7)
chk("K3 contact BC", tab.loc[("K3", "BC", 25), "c"], 72.7); chk("K3 contact DAgger", tab.loc[("K3", "DAgger", 25), "c"], 13.3)
chk("lat contact BC", tab.loc[("K2+latched", "BC", 25), "c"], 12.0); chk("lat contact DAgger", tab.loc[("K2+latched", "DAgger", 25), "c"], 4.0)
chk("K0 BC 100", tab.loc[("K0", "BC", 100), "s"], 36.3); chk("K0 DAgger 100", tab.loc[("K0", "DAgger", 100), "s"], 17.0)
chk("K0 BC 25", tab.loc[("K0", "BC", 25), "s"], 0.0); chk("K0 DAgger 25", tab.loc[("K0", "DAgger", 25), "s"], 0.0)

# 3.7 contact, 3.8 formations
for pol, v25 in (("GNN K1", 87.3), ("GNN K2", 51.7), ("GNN K3", 13.3)): chk(f"{pol} contact 25", s(pol, 25, "contact_pct"), v25)
chk("K2 contact 50", s("GNN K2", 50, "contact_pct"), 55.2); chk("K2 contact 100", s("GNN K2", 100, "contact_pct"), 52.2)
chk("max overlap % (all GNN)", (G.min_clearance < 0.069).mean() * 100, 0.0, 0); chk("min clearance mm", 1000 * G.min_clearance.min(), 72.5)
for pol, w in (("GNN K1", 25.6), ("GNN K2", 3.5), ("GNN K3", 2.0), ("GNN K2+latched", 1.2)): chk(f"{pol} final err", s(pol, 25, "mean_final_err_m") * 1000, w)
g25 = G[G.obs_pct == 25]; f = 100 * g25.groupby(["pol", "formation"]).success.mean(); cf = 100 * g25.groupby(["pol", "formation"]).collision.mean()
for form, w in (("circle", 98.0), ("cluster", 95.3), ("line", 96.7), ("split", 100.0)): chk(f"K2 {form}", f[("GNN K2", form)], w)
for form, w in (("circle", 73.3), ("cluster", 54.0), ("line", 80.7), ("split", 81.3)): chk(f"K1 {form}", f[("GNN K1", form)], w)
chk("K2 contact split", cf[("GNN K2", "split")], 37.3); chk("K2 contact cluster", cf[("GNN K2", "cluster")], 66.0)
rk2 = B[(B.obs_pct == 25) & (B.policy == "K2")].groupby("formation").success.mean() * 100
chk("relay K2 split", rk2["split"], 60.0); chk("relay K2 cluster", rk2["cluster"], 100.0)

# 4 discussion
chk("K3 vs K2 25 diff", p("K=3 vs K=2 (25", "dsuccess_pct"), 1.5); chk("K3 vs K2 25 holm success 0.031", p("K=3 vs K=2 (25", "p_success_holm"), 0.031, 0.0006)
chk("K3 vs K2 25 holm final 0.44", p("K=3 vs K=2 (25", "p_final_err_holm"), 0.44, 0.006)

print(f"{n} numbers checked")
if bad:
    print("MISMATCHES (fix the text, not the data):"); [print("  " + b) for b in bad]; raise SystemExit(1)
print("all quoted numbers match the results files")
