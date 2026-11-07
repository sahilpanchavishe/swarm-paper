"""Day 14: every paper table and the three new figures, rebuilt from the committed Day 12 / Day 14 raw CSVs only.
No simulation, no training, no number typed by hand.   Run from src/ (or anywhere):   python d14_tables_figures.py

Tables -> paper/tables/ (each as .csv and .md), all with TWO-LEVEL bootstrap 95 % intervals (scenarios and training seeds resampled):
  t14_1_confirmatory   headline: success, contact, collision-free success, clearance, max per-robot error. Seeds 20000-20049.   CONFIRMATORY
  t14_2_capacity       K=1 / K=2 vs the three capacity controls (wide1, shared2, mlpwide) and K=0.                              CONFIRMATORY
  t14_3_safety         same models with and without the pre-selected filter (dsafe 0.10, gamma 20), seeds 20000-20049.         CONFIRMATORY
  t14_4_widebox        1.8 m start box: unfiltered, filtered, strong relay (relayD).                                           EXPLORATORY (seeds 30000+)
  t14_5_robustness     link loss, sensing delay, position noise.                                                               EXPLORATORY
  t14_6_change         mid-episode command change.                                                                             EXPLORATORY
  t14_7_paired         paired tests copied from d12_paired_confirm_capacity.csv, d14_safety_paired.csv, d14_widebox_*_paired.csv
Figures -> paper/figs/:  figQ_capacity, figR_filter_clearance, figS_robustness  (.png and .pdf)
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES, TAB, FIG = ROOT / "results", ROOT / "paper" / "tables", ROOT / "paper" / "figs"
TAB.mkdir(exist_ok=True, parents=True); FIG.mkdir(exist_ok=True, parents=True)
CONTACT_M = 0.074      # centre distance below two robot radii = contact
NBOOT, SEED = 5000, 0
OBS = (100, 50, 25)
BASE = ["loss", "delay", "noise", "start_half", "safety", "change"]


def load(name):
    d = pd.read_csv(RES / name)
    d["train_seed"] = d.train_seed.fillna(-1).astype(int)
    d["collision"] = d.collision.astype(bool); d["success"] = d.success.astype(bool)
    d["cf"] = d.success & ~d.collision
    return d


def boot_ci(x, col, rng):
    """Two-level bootstrap of the mean of x[col]: resample (formation, seed) scenarios and training seeds (relays: one pseudo-seed)."""
    W = x.pivot_table(index=["formation", "seed"], columns="train_seed", values=col, aggfunc="mean").values.astype(float)
    n, m = W.shape
    out = np.array([W[rng.integers(0, n, n)][:, rng.integers(0, m, m)].mean() for _ in range(NBOOT)])
    return np.percentile(out, [2.5, 97.5])


def summ(x, rng, **tags):
    lo, hi = boot_ci(x.assign(v=x.success.astype(float)), "v", rng)
    clo, chi = boot_ci(x.assign(v=x.cf.astype(float)), "v", rng)
    return dict(**tags, episodes=len(x), training_seeds=int((x.train_seed >= 0).any() and x.train_seed[x.train_seed >= 0].nunique()),
                success_pct=100 * x.success.mean(), succ_lo=100 * lo, succ_hi=100 * hi,
                contact_pct=100 * x.collision.mean(), cf_success_pct=100 * x.cf.mean(), cf_lo=100 * clo, cf_hi=100 * chi,
                median_clearance_mm=1000 * x.min_clearance.median(),
                max_err_mean_mm=1000 * x.max_err.mean(), max_err_p95_mm=1000 * x.max_err.quantile(.95))


def save(df, stem, caption):
    df = df.round(2); df.to_csv(TAB / f"{stem}.csv", index=False)
    cols = list(df.columns)
    md = [f"**{caption}**\n", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    (TAB / f"{stem}.md").write_text("\n".join(md) + "\n"); print(f"wrote {stem} ({len(df)} rows)")


def order(df, pols, key="policy"):
    df = df.copy(); df["_o"] = df[key].map({p: i for i, p in enumerate(pols)}); return df.sort_values(["_o", "obs_pct"], ascending=[True, False]).drop(columns="_o")


rng = np.random.default_rng(SEED)

# ---- 1 confirmatory headline -------------------------------------------------------------------------------------------------
C = load("d12_confirm.csv")
P1 = ["oracle", "relay K2", "relay K2+mem", "relay K3+mem", "relay K7+mem", "relayD K2", "GNN K0", "GNN K1", "GNN K2", "GNN K3", "GNN K2+latched"]
rows = [summ(C[(C.policy == p) & (C.obs_pct == o)], rng, policy=p, obs_pct=o) for p in P1 for o in OBS if len(C[(C.policy == p) & (C.obs_pct == o)])]
T1 = order(pd.DataFrame(rows), P1)
save(T1, "t14_1_confirmatory", "Table 1. Confirmatory results, fresh scenarios (seeds 20000-20049), no safety filter. Success = final error below 25 mm. "
     "Intervals: two-level bootstrap over scenarios and training seeds (GNN K=1,2,3: 5 training seeds; K=0 and latched: 3; relays and oracle: none).")

# ---- 2 capacity controls -----------------------------------------------------------------------------------------------------
K = load("d12_capacity.csv"); CC = pd.concat([C[C.policy.isin(["GNN K0", "GNN K1", "GNN K2"])], K])
CC = CC[CC.train_seed.isin([0, 1, 2])]       # controls exist for training seeds 0-2 only: compare on the same three seeds
P2 = ["GNN K0", "GNN K0-wide", "GNN K1", "GNN K1-wide", "GNN K2-shared", "GNN K2"]
rows = [summ(CC[(CC.policy == p) & (CC.obs_pct == o)], rng, policy=p, obs_pct=o) for p in P2 for o in OBS]
labels = {"GNN K0": "K=0 (23k params)", "GNN K0-wide": "K=0 wide, no messages (65k)", "GNN K1": "K=1 (44k)", "GNN K1-wide": "K=1 wide (65k)",
          "GNN K2-shared": "K=2 shared weights (44k)", "GNN K2": "K=2 (65k)"}
T2 = order(pd.DataFrame(rows), P2); T2.insert(1, "model", T2.policy.map(labels))
save(T2, "t14_2_capacity", "Table 2. Capacity controls on training seeds 0-2 (all models, same recipe, seeds 20000-20049). Wide K=1 matches K=2's parameter count; "
     "shared K=2 matches K=1's; wide K=0 has no message passing.")

# ---- 3 safety filter -----------------------------------------------------------------------------------------------------------
S = load("d12_safety_final.csv")
P3 = ["GNN K1", "GNN K2", "GNN K3", "GNN K2+latched", "relay K2+mem"]
rows = []
for p in P3:
    for o in OBS:
        for lab, src in (("no filter", C), ("filter 0.10/20", S)):
            x = src[(src.policy == p) & (src.obs_pct == o)]
            if len(x): rows.append(summ(x, rng, policy=p, obs_pct=o, filter=lab))
T3 = pd.DataFrame(rows); T3["_o"] = T3.policy.map({p: i for i, p in enumerate(P3)}); T3 = T3.sort_values(["_o", "obs_pct", "filter"], ascending=[True, False, False]).drop(columns="_o")
save(T3, "t14_3_safety", "Table 3. Effect of the decentralised safety filter (dsafe 0.10 m, gamma 20, chosen on development seeds 5000-5009 by a rule fixed before the run), "
     "seeds 20000-20049. cf_success = success with no robot contact. Empirical filter, no formal guarantee, simulation with accurate positions.")

# ---- 4 wide box ----------------------------------------------------------------------------------------------------------------
W = pd.concat([load("d12_widebox.csv").assign(filt="no filter"), load("d12_widebox_relayD.csv").assign(filt="no filter"),
               load("d12_widebox_safe.csv").assign(filt="filter 0.10/20")], ignore_index=True)
rows = []
for (p, f, o), x in W.groupby(["policy", "filt", "obs_pct"]):
    rows.append(summ(x, rng, policy=p, obs_pct=o, filter=f))
T4 = pd.DataFrame(rows).sort_values(["policy", "filter", "obs_pct"], ascending=[True, False, False])
save(T4, "t14_4_widebox", "Table 4 (EXPLORATORY). 1.8 m start box, seeds 30000-30024, 25 scenarios x 4 formations, GNN training seeds 0-2. "
     "relayD = relay that sends uninformed robots to the default slot, with latest-wins token memory; it is given the formation geometry.")

# ---- 5 robustness --------------------------------------------------------------------------------------------------------------
R = load("d12_robust.csv")
rows = []
for (p, o, ls, dl, ns), x in R.groupby(["policy", "obs_pct", "loss", "delay", "noise"]):
    rows.append(summ(x, rng, policy=p, obs_pct=o, link_loss=ls, delay_steps=dl, noise_m=ns))
T5 = pd.DataFrame(rows).sort_values(["policy", "obs_pct", "link_loss", "delay_steps", "noise_m"], ascending=[True, False, True, True, True])
save(T5, "t14_5_robustness", "Table 5 (EXPLORATORY). Link loss (per step), sensing delay (control steps) and position noise, one perturbation at a time. No filter.")

# ---- 6 command change ----------------------------------------------------------------------------------------------------------
G = load("d12_change.csv"); rows = []
for (p, o), x in G.groupby(["policy", "obs_pct"]):
    rows.append(summ(x, rng, policy=p, obs_pct=o))
T6 = pd.DataFrame(rows).sort_values(["policy", "obs_pct"], ascending=[True, False])
save(T6, "t14_6_change", "Table 6 (EXPLORATORY). Command changes at 17.6 s, horizon 52.8 s; success = reaching the NEW formation. "
     "naive = a stale token is never overwritten; ver = version-aware token register.")

# ---- 7 paired tests, copied -----------------------------------------------------------------------------------------------------
keep = ["comparison", "dsuccess_pct", "ci_lo", "ci_hi", "per_seed_dsuccess_pct", "seeds_a_better", "p_success_max", "p_success_holm"]
parts = []
for f, lab in (("d12_paired_confirm_capacity.csv", "confirmatory, success"), ("d14_safety_paired.csv", "confirmatory, collision-free success"),
               ("d14_widebox_paired.csv", "EXPLORATORY wide box, unfiltered"), ("d14_widebox_safe_paired.csv", "EXPLORATORY wide box, filtered")):
    q = pd.read_csv(RES / f)
    if "metric" in q: q["family"] = lab + " / " + q.metric
    else: q["family"] = lab
    parts.append(q[["family"] + [c for c in keep if c in q]])
T7 = pd.concat(parts, ignore_index=True)
save(T7, "t14_7_paired", "Table 7. Paired comparisons (exact McNemar per training seed, max p over seeds, Holm over the family). "
     "Call a difference significant only if Holm p < 0.05 AND every training seed points the same way (seeds_a_better counts seeds where the first model wins).")

# ---- figures -------------------------------------------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
COL = {"GNN K0": "#999999", "GNN K0-wide": "#bbbbbb", "GNN K1": "#d55e00", "GNN K1-wide": "#e69f00", "GNN K2-shared": "#009e73", "GNN K2": "#0072b2"}


def fig_save(fig, stem): fig.savefig(FIG / f"{stem}.png", bbox_inches="tight"); fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight"); plt.close(fig); print("wrote", stem)


# Q capacity
fig, ax = plt.subplots(figsize=(6.4, 3.2)); w = 0.13
for j, p in enumerate(P2):
    for i, o in enumerate(OBS):
        r = T2[(T2.policy == p) & (T2.obs_pct == o)].iloc[0]
        ax.bar(i + (j - 2.5) * w, r.success_pct, w, color=COL[p], label=labels[p] if i == 0 else None,
               yerr=[[r.success_pct - r.succ_lo], [r.succ_hi - r.success_pct]], error_kw=dict(lw=.8, capsize=1.5))
ax.set_xticks(range(3)); ax.set_xticklabels([f"{o}% observability" for o in OBS]); ax.set_ylabel("success (%)"); ax.set_ylim(0, 105)
ax.set_title("Communication depth, not parameter count (training seeds 0-2; 95% bootstrap CI)", fontsize=9)
ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12)); fig_save(fig, "figQ_capacity")

# R clearance with / without filter, 25 % obs
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
for ax, p in zip(axs, ["GNN K2", "GNN K3"]):
    for src, lab, c in ((C, "no filter", "#d55e00"), (S, "filter 0.10/20", "#0072b2")):
        v = np.sort(1000 * src[(src.policy == p) & (src.obs_pct == 25)].min_clearance.values)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, label=lab)
    ax.axvline(1000 * CONTACT_M, color="k", ls=":", lw=.8); ax.text(1000 * CONTACT_M + 1, .05, "contact", fontsize=7)
    ax.set_title(p + ", 25% observability", fontsize=9); ax.set_xlabel("minimum centre distance in episode (mm)")
axs[0].set_ylabel("fraction of episodes"); axs[0].legend(frameon=False, fontsize=7, loc="lower right"); fig_save(fig, "figR_filter_clearance")

# S robustness: link loss and delay at 25 % obs
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
PS = ["GNN K2", "GNN K2+latched", "GNN K3", "relay K2", "relay K2+mem", "relayD K2"]
colS = dict(zip(PS, ["#0072b2", "#56b4e9", "#009e73", "#d55e00", "#e69f00", "#cc79a7"]))
for ax, (axis, xs, xl) in zip(axs, (("link_loss", [0, .1, .3, .5], "link loss per step"), ("delay_steps", [0, 5, 10], "sensing delay (control steps)"))):
    for p in PS:
        t = T5[(T5.policy == p) & (T5.obs_pct == 25)]
        for k in ("link_loss", "delay_steps", "noise_m"):
            if k != axis: t = t[t[k] == 0]
        t = t.sort_values(axis)
        if len(t): ax.plot(t[axis], t.success_pct, "o", ms=3, color=colS[p], label=p, ls="--" if p.startswith("relay") else "-")
    ax.set_xlabel(xl); ax.set_xticks(xs)
axs[0].set_ylabel("success (%)  [25% obs, exploratory]"); axs[1].legend(frameon=False, fontsize=7, loc="lower left"); fig_save(fig, "figS_robustness")
print("\nAll done. Every number above comes from results/*.csv.")
