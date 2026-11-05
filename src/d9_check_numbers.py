"""Day 9c: number checks for the exploratory Day 9 results, plus a scan for numbers in the draft that NO check covers.
Same rule as d8_check_numbers.py: if a check fails, fix the TEXT (never the data) and update the expected value here.
Part 1  recomputes the numbers for (a) the fairer relay (d9_relay_default.py) and (b) the token cut-off test (d9_token_cutoff.py) and the per-formation comparison.
Part 2  lists every number in paper/results_discussion_draft.md that does not appear as a literal in d8_check_numbers.py or d9_check_numbers.py.
        This is a HEURISTIC: a number counts as covered if the same digits appear as a whole number anywhere in the two checker files, even for a different quantity, so a
        number NOT listed is not proof it was checked, and a listed one is a definite gap. It does not fail the run: review each listed line, then add a chk() for it.
Run (venv active):  cd src && python d8_check_numbers.py && python d9_check_numbers.py"""
import re
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; R = ROOT / "results"
D = pd.read_csv(R / "d9_relay_default.csv"); DP = pd.read_csv(R / "d9_relay_default_paired.csv"); C = pd.read_csv(R / "d9_token_cutoff.csv")
B = pd.read_csv(R / "d3_baselines.csv"); E = pd.read_csv(R / "d6_main_eval.csv"); G = E[(E.method == "DAgger") & (~E.latched)]
bad = []; n = 0

def chk(name, got, want, tol=0.051):
    global n; n += 1
    if abs(float(got) - float(want)) > tol: bad.append(f"{name}: computed {float(got):.3f}, text says {want}")

def rd(K, pct): return 100 * D[(D.policy == f"relayD K{K}") & (D.obs_pct == pct)].success.mean()
def dp(prefix, col): return DP[DP.comparison.str.startswith(prefix)].iloc[0][col]
def cut(K, variant, tc, form=None):
    x = C[(C.K == K) & (~C.latched) & (C.variant == variant) & (C.t_cut == tc)]
    if form: x = x[x.formation == form]
    return 100 * x.success.mean()

# (a) fairer relay: uninformed robots head to the mean of their four candidate slots (exploratory)
for K, w25, w50 in ((1, 50.0, 92.0), (2, 81.5, 99.0), (3, 90.0, 99.0), (7, 90.0, 99.0)): chk(f"relayD K{K} 25", rd(K, 25), w25); chk(f"relayD K{K} 50", rd(K, 50), w50)
chk("relayD K0 25", rd(0, 25), 0.0); chk("relayD contact (all)", 100 * D.collision.mean(), 0.0, 0)
for lab, d, lo, hi, p in (("GNN vs relayD, K=1 (25", 22.3, 14.2, 30.2, 0.0001), ("GNN vs relayD, K=2 (25", 16.0, 10.5, 21.7, 0.0000), ("GNN vs relayD, K=3 (25", 9.0, 5.0, 13.5, 0.0002),
                          ("GNN vs relayD, K=1 (50", -13.0, -18.7, -7.3, 0.0002), ("GNN vs relayD, K=2 (50", -0.8, -2.5, 1.2, 0.63), ("GNN vs relayD, K=3 (50", 0.8, -0.3, 2.5, 0.63),
                          ("relayD vs relay (hold), K=1 (25", 11.0, 4.5, 17.5, 0.012), ("relayD vs relay (hold), K=2 (25", -4.0, -7.0, -1.0, 0.069)):
    chk(lab + " diff", dp(lab, "dsuccess_pct"), d); chk(lab + " lo", dp(lab, "dsuccess_ci_lo"), lo); chk(lab + " hi", dp(lab, "dsuccess_ci_hi"), hi)
    chk(lab + " holm p", dp(lab, "p_success_holm"), p, 0.0006 if p < 0.01 else 0.006)

# (b) token cut-off at 25 % observability. flagoff = token and flag removed for every robot (clean "nobody told me"); flagon = Day 7 style blank (flag stays on, out of distribution)
for K, (c, o0, o300, o700, f0, f700) in {1: (70.0, 17.5, 55.0, 61.3, 1.2, 7.5), 2: (97.1, 24.2, 71.7, 72.9, 2.1, 5.8), 3: (97.5, 23.8, 73.8, 73.8, 8.8, 20.0)}.items():
    chk(f"K{K} control (never removed)", cut(K, "flagon", 1100), c, 0.06); chk(f"K{K} flagoff t0", cut(K, "flagoff", 0), o0, 0.06)
    chk(f"K{K} flagoff t300", cut(K, "flagoff", 300), o300, 0.06); chk(f"K{K} flagoff t700", cut(K, "flagoff", 700), o700, 0.06)
    chk(f"K{K} flagon t0", cut(K, "flagon", 0), f0, 0.06); chk(f"K{K} flagon t700", cut(K, "flagon", 700), f700, 0.06)
for form, a0, a700 in (("circle", 25.0, 100.0), ("line", 16.7, 91.7), ("cluster", 0.0, 0.0), ("split", 55.0, 100.0)):
    chk(f"K2 flagoff t0 {form}", cut(2, "flagoff", 0, form), a0, 0.06); chk(f"K2 flagoff t700 {form}", cut(2, "flagoff", 700, form), a700, 0.06)
chk("K2 worse after cut t700 (%)", 100 * C[(C.K == 2) & (~C.latched) & (C.variant == "flagoff") & (C.t_cut == 700)].worse_after_cut.mean(), 25.8)
chk("K2 error at cut t700 (mm)", 1000 * C[(C.K == 2) & (~C.latched) & (C.variant == "flagoff") & (C.t_cut == 700)].err_at_cut.mean(), 4.0, 0.6)
chk("train seeds K2", C[(C.K == 2) & (~C.latched)].train_seed.nunique(), 3, 0); chk("unique test episodes per cell", C[(C.K == 2) & (C.t_cut == 700) & (C.variant == "flagoff") & (C.train_seed == 0)].shape[0], 80, 0)

# GNN minus memoryless relay by formation (25 % observability): the advantage sits where positional memory works
def by_form(pol_df, K): return 100 * pol_df[(pol_df.obs_pct == 25)].groupby("formation").success.mean()
for K, want in ((2, {"circle": 10.0, "line": 2.7, "cluster": -4.7, "split": 40.0}), (1, {"circle": 53.3, "line": 38.7, "cluster": -22.0, "split": 63.3})):
    gn = by_form(G[G.K == K], K); rl = by_form(B[B.policy == f"K{K}"], K)
    for form, w in want.items(): chk(f"GNN K{K} minus relay K{K}, {form}", gn[form] - rl[form], w)

# (c) Methods constants against the CODE (so the Methods text cannot drift from what was run)
import sys, inspect; sys.path.insert(0, str(HERE))
import swarm_sim as SS, mj_env as M
xml = M.build_xml()
for nm, got, want in (("R_WHEEL", M.R_WHEEL, 0.0205), ("AXLE", M.AXLE, 0.052), ("ROBOT_R", M.ROBOT_R, 0.037), ("wheel speed limit", M.W_WHEEL_MAX, 6.28), ("v_max", M.V_MAX, 0.1287),
                      ("torque cap", M.TAU_MAX, 0.004), ("control dt", M.DT_CTRL, 0.032), ("physics dt", M.DT_PHYS, 0.004), ("substeps", M.SUBSTEPS, 8), ("arena side", 2 * M.ARENA, 2.2),
                      ("R_comm", SS.R_COMM, 0.50), ("contact distance = 2 x radius", 2 * M.ROBOT_R, 0.074)):
    chk(f"const {nm}", got, want, 5e-5)
sig = lambda f: {k: v.default for k, v in inspect.signature(f).parameters.items() if v.default is not inspect.Parameter.empty}
for k, w in (("k_att", 2.8), ("k_rep", 5e-3), ("r_inf", 0.12), ("k_tan", 0.5)): chk(f"expert {k}", sig(SS.expert_velocity_vector)[k], w, 1e-9)
chk("heading gain k_w", sig(SS.to_unicycle)["k_w"], 4.0, 1e-9); chk("chassis friction default", sig(M.build_xml)["chassis_mu"], 0.3, 1e-9)
chk("min separation", sig(SS.random_start)["min_sep"], 0.15, 1e-9)
chk("actuator gain kv", float(re.search(r'kv="([\d.]+)"', xml).group(1)), 0.05, 1e-9); chk("chassis mass", float(re.search(r'name="chassis0"[^>]*mass="([\d.]+)"', xml).group(1)), 0.15, 1e-9)
chk("wheel friction", float(re.search(r'<body name="wl0".*?friction="([\d.]+) ', xml, re.S).group(1)), 1.5, 1e-9); chk("floor friction", float(re.search(r'name="floor"[^>]*friction="([\d.]+) ', xml).group(1)), 1.0, 1e-9)
d3src = (HERE / "d3_baselines.py").read_text()
chk("horizon steps", int(re.search(r"T_STEPS, START_HALF, SEED0 = (\d+), ([\d.]+), (\d+)", d3src).group(1)), 1100, 0); chk("horizon seconds", 1100 * M.DT_CTRL, 35.2)
chk("start half-width (box side = 1.0 m)", 2 * float(re.search(r"T_STEPS, START_HALF, SEED0 = (\d+), ([\d.]+), (\d+)", d3src).group(2)), 1.0)
assert "errs[-1] < 0.025" in d3src and "errs[-1] < 0.05" in d3src, "success thresholds in d3_baselines.py are not 0.025 / 0.05"
V = pd.read_csv(R / "model_validation.csv")
for col, want in (("path_error_pct", (-3.5, -2.3)), ("yaw_error_pct", (-3.2, -8.0))):
    vals = V[col].dropna().round(1).tolist()
    for w in want: chk(f"model validation {col} has {w}", min(abs(v - w) for v in vals), 0.0, 0.051)
per_model = E.groupby("model").size(); chk("episodes per trained model", per_model.min(), 600, 0); chk("episodes per trained model (max)", per_model.max(), 600, 0)
chk("episodes per observability level", E.groupby(["model", "obs_pct"]).size().min(), 200, 0)

print(f"{n} numbers checked")
if bad:
    print("MISMATCHES (fix the text, not the data):"); [print("  " + b) for b in bad]

# Part 2: numbers in the draft that no check mentions (heuristic)
src = (HERE / "d8_check_numbers.py").read_text() + (HERE / "d9_check_numbers.py").read_text()
draft = (ROOT / "paper" / "results_discussion_draft.md").read_text().splitlines()
pat = re.compile(r"(?<![\w.\-])(\d+\.\d+|\d+)(?= ?%| points| episodes| mm\b| m\b)")
unchecked = []
for i, line in enumerate(draft, 1):
    if line.lstrip().startswith("#"): continue
    for m in pat.finditer(line):
        tok = m.group(1)
        covered = re.search(r"(?<![\d.])" + re.escape(tok) + r"(?![\d])", src) is not None                  # whole-number match, so 5.0 does not count as covered by 25.0
        if not covered and tok not in {"0", "1", "2", "3", "4", "5", "8", "100", "50", "25", "10", "20", "30"}:        # observability levels, K values, seed counts
            unchecked.append((i, tok, line.strip()[max(0, m.start() - 55): m.end() + 35].replace("\n", " ")))
print(f"\n=== {len(unchecked)} numbers in paper/results_discussion_draft.md that no check mentions (review each by hand, then add a chk()) ===")
for i, tok, ctx in unchecked: print(f"  line {i:3d}  {tok:>7}   ...{ctx}...")
if bad: raise SystemExit(1)
print("\nall d9 numbers match the results files")
