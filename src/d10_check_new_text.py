"""Day 10c: number checks for the text added in Sections 2.11, 3.9 and 3.10 and the rewritten Discussion paragraph, for numbers that
d8_check_numbers.py and d9_check_numbers.py do not already check. Reads only ../results/*.csv (no MuJoCo needed).
Same rule as the other checkers: if a check fails, fix the TEXT (never the data) and update the expected value here.
Run:  cd src && python d10_check_new_text.py"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent; R = ROOT / "results"
D = pd.read_csv(R / "d9_relay_default.csv"); DP = pd.read_csv(R / "d9_relay_default_paired.csv"); C = pd.read_csv(R / "d9_token_cutoff.csv")
E = pd.read_csv(R / "d6_main_eval.csv"); B = pd.read_csv(R / "d3_baselines.csv")
bad = []; n = 0

def chk(name, got, want, tol=0.051):
    global n; n += 1
    if abs(float(got) - float(want)) > tol: bad.append(f"{name}: computed {float(got):.3f}, text says {want}")

def dp(prefix, col): return DP[DP.comparison.str.startswith(prefix)].iloc[0][col]
def cut(K, variant, tc, seed=None, form=None):
    x = C[(C.K == K) & (~C.latched) & (C.variant == variant) & (C.t_cut == tc)]
    if seed is not None: x = x[x.train_seed == seed]
    if form: x = x[x.formation == form]
    return 100 * x.success.mean()

# Section 3.9
chk("relayD K0 at 50 %", 100 * D[(D.policy == "relayD K0") & (D.obs_pct == 50)].success.mean(), 0.0)
chk("number of d9 comparisons (Holm family)", len(DP), 12, 0)
chk("n pairs", DP.n_pairs.min(), 200, 0)
chk("relayD vs hold K2 25: final error mm", 1000 * dp("relayD vs relay (hold), K=2 (25", "dfinal_err_m"), 5.9)
chk("relayD vs hold K2 25: Holm p final error", dp("relayD vs relay (hold), K=2 (25", "p_final_err_holm"), 0.0011, 0.00006)
chk("relayD vs hold K2 25: Holm p success", dp("relayD vs relay (hold), K=2 (25", "p_success_holm"), 0.069, 0.0006)
chk("GNN vs relayD K3 25: Holm p final error", dp("GNN vs relayD, K=3 (25", "p_final_err_holm"), 0.21, 0.006)
chk("K1 gap split: 33.3 - 22.3", 33.3 - dp("GNN vs relayD, K=1 (25", "dsuccess_pct"), 11.0)
chk("K2 advantage over hold relay (12.0) vs default-slot relay (16.0)", dp("GNN vs relayD, K=2 (25", "dsuccess_pct"), 16.0)
G = E[(E.method == "DAgger") & (~E.latched) & (E.obs_pct == 25)]
chk("GNN K2 episodes per formation at 25 %", G[G.K == 2].groupby("formation").size().min(), 150, 0)
chk("relay K2 episodes per formation at 25 %", B[(B.policy == "K2") & (B.obs_pct == 25)].groupby("formation").size().min(), 50, 0)

# Section 3.10
chk("K2 flagoff t300 / t0 ratio ~3", cut(2, "flagoff", 300) / cut(2, "flagoff", 0), 3.0, 0.06)
chk("K2 flagoff t700 / t0 ratio ~3", cut(2, "flagoff", 700) / cut(2, "flagoff", 0), 3.0, 0.06)
for sd, w0, w700 in ((0, 25.0, 75.0), (1, 23.8, 70.0), (2, 23.8, 73.8)):
    chk(f"K2 seed {sd} flagoff t0", cut(2, "flagoff", 0, sd), w0); chk(f"K2 seed {sd} flagoff t700", cut(2, "flagoff", 700, sd), w700)
k2 = C[(C.K == 2) & (~C.latched) & (C.variant == "flagoff") & (C.t_cut == 700) & (C.formation == "cluster")]
chk("cluster error at cut (mm)", 1000 * k2.err_at_cut.mean(), 3.9); chk("cluster final error (mm)", 1000 * k2.final_err.mean(), 431.1)
chk("K2 flagon t0", cut(2, "flagon", 0), 2.1); chk("K2 flagon t300", cut(2, "flagon", 300), 5.8); chk("K2 flagon t700", cut(2, "flagon", 700), 5.8)
chk("K1 flagon t0", cut(1, "flagon", 0), 1.2); chk("K1 flagon t700", cut(1, "flagon", 700), 7.5)
chk("K3 flagon t0", cut(3, "flagon", 0), 8.8); chk("K3 flagon t700", cut(3, "flagon", 700), 20.0)
chk("K2 flagoff t300 mean over 240", cut(2, "flagoff", 300), 71.7); chk("K2 control", cut(2, "flagon", 1100), 97.1)
chk("worse-after-cut share (K2, t700, flagoff)", 100 * C[(C.K == 2) & (~C.latched) & (C.variant == "flagoff") & (C.t_cut == 700)].worse_after_cut.mean(), 25.8)
chk("cut-off steps in seconds: 300 steps", 300 * 0.032, 9.6); chk("cut-off steps in seconds: 700 steps", 700 * 0.032, 22.4)
chk("cut-off networks: K2 seeds", C[(C.K == 2) & (~C.latched)].train_seed.nunique(), 3, 0)
chk("cut-off networks: K1 seeds", C[(C.K == 1) & (~C.latched)].train_seed.nunique(), 1, 0); chk("cut-off networks: K3 seeds", C[(C.K == 3) & (~C.latched)].train_seed.nunique(), 1, 0)
chk("cut-off test observability only 25 %", C.obs_pct.nunique(), 1, 0); chk("cut-off test: 20 seeds", C.seed.nunique(), 20, 0)
chk("flag-off table n K1", len(C[(C.K == 1) & (~C.latched) & (C.variant == "flagoff") & (C.t_cut == 300)]), 80, 0)
chk("flag-off table n K2", len(C[(C.K == 2) & (~C.latched) & (C.variant == "flagoff") & (C.t_cut == 300)]), 240, 0)

print(f"{n} numbers checked")
if bad:
    print("MISMATCHES (fix the text, not the data):"); [print("  " + b) for b in bad]; raise SystemExit(1)
print("all quoted numbers match the results files")
