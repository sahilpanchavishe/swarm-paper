"""Day 15: paper tables and figures for the new studies (A, B, C, D, drift, frontier, replay). p-values print as <0.0001. Skips any study whose result file is missing.
Run: python src/d15_tables.py   Writes paper/tables/t15_*.csv/.md and paper/figs/f15_*.pdf"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent; R = ROOT / "results"; TAB = ROOT / "paper" / "tables"; FIG = ROOT / "paper" / "figs"; TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
def fp(x): return x if isinstance(x, str) or x != x else ("<0.0001" if x < 1e-4 else f"{x:.4f}")
def save(df, stem):
    df = df.copy()
    for c in df.columns:
        if str(c).startswith("p_"): df[c] = df[c].map(fp)
    num = df.select_dtypes("number").columns; df[num] = df[num].round(1); df.to_csv(TAB / f"{stem}.csv", index=False); cols = list(df.columns)
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    (TAB / f"{stem}.md").write_text("\n".join(L) + "\n"); print("wrote", stem, len(df), "rows")
def load(n):
    p = R / n
    if not p.exists(): print("skip (missing):", n); return None
    return pd.read_csv(p)
PC = ["comparison", "dsuccess_pct", "ci_lo", "ci_hi", "seeds_a_better", "p_success_max", "p_success_holm"]
d = load("d15_A_descriptive.csv");  save(d, "t15_1_A_descriptive") if d is not None else None
d = load("d15_A_paired.csv");       save(d[PC], "t15_2_A_paired") if d is not None else None
d = load("d15_B_descriptive.csv");  save(d, "t15_3_B_descriptive") if d is not None else None
d = load("d15_B_paired.csv");       save(d[PC], "t15_4_B_paired") if d is not None else None
C = load("d15_C_descriptive.csv");  save(C, "t15_5_C_descriptive") if C is not None else None
d = load("d12_d15_change_summary.csv")
if d is not None: save(d[[c for c in ["policy", "obs_pct", "safety", "episodes", "success_pct", "ci_lo", "ci_hi", "collision_free_success_pct", "contact_pct", "max_err_p95_mm"] if c in d]], "t15_6_change_EXPLORATORY")
d = load("d15_drift_summary.csv");  save(d, "t15_7_drift_EXPLORATORY") if d is not None else None
F = load("d12_d15_frontier.csv")
if F is not None:
    F["cf"] = F.success.astype(bool) & ~F.collision.astype(bool)
    G = F.groupby(["start_half", "policy", "obs_pct"]).agg(episodes=("cf", "size"), mean_init_components=("init_comp", "mean"), success_pct=("success", lambda x: 100 * x.mean()), contact_pct=("collision", lambda x: 100 * x.mean()), cf_success_pct=("cf", lambda x: 100 * x.mean())).reset_index()
    save(G, "t15_8_frontier_EXPLORATORY")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for a_, pct in zip(ax, (25, 50)):
        for pol, x in G[G.obs_pct == pct].groupby("policy"): a_.plot(x.start_half, x.cf_success_pct, "o-", label=pol)
        a_.set_title(f"{pct}% observers"); a_.set_xlabel("start box half-width (m)"); a_.axvline(1.1, ls=":", c="gray")
    ax[0].set_ylabel("collision-free success (%)"); ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "f15_frontier.pdf"); plt.close(fig)
d = load("d15_replay.csv");         save(d, "t15_9_replay_EXPLORATORY") if d is not None else None
if C is not None:
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for a_, var in zip(ax, ("flagoff", "flagon")):
        for pol, x in C[C.variant == var].groupby("policy"):
            x = x.sort_values("t_cut"); a_.errorbar(x.t_cut, x.success_pct, yerr=[x.success_pct - x.seed_min, x.seed_max - x.success_pct], fmt="o-", capsize=2, label=pol)
        a_.set_title(var); a_.set_xlabel("token removed at control step")
    ax[0].set_ylabel("success (%)"); ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "f15_cutoff.pdf"); plt.close(fig)
B = load("d15_B_descriptive.csv")
if B is not None:
    x = B[B.obs_pct == 25].pivot_table(index="cond", columns="policy", values="contact_pct"); ax = x.plot.bar(figsize=(8, 3)); ax.set_ylabel("episodes with contact (%)")
    ax.figure.tight_layout(); ax.figure.savefig(FIG / "f15_filter_disturbance.pdf"); plt.close("all")
