"""Day 16: the main-text figures of the AME manuscript in one consistent style, from the committed raw CSVs only.

  python d16_paper_figures.py

Writes paper/figs/ame_fig2_confirmatory, ame_fig3_capacity, ame_fig4_clearance, ame_fig5_cutoff, ame_fig6_stress (.png 300 dpi + .pdf),
sized for the AME text width (12.6 cm). Intervals: two-level bootstrap over test scenarios and training seeds (2000 resamples, seed 0);
Fig. 5 shows the range over training seeds from results/d15_C_descriptive.csv."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent; RES = ROOT / "results"; FIG = ROOT / "paper" / "figs"
W = 12.6 / 2.54  # AME text width in inches
BLUE, ORANGE, GREEN, YELLOW, GREY, DARK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9a9a", "#4a4a4a"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True})
RNG = np.random.default_rng(0); NB = 2000


def boot(x, col):
    """Two-level bootstrap of a mean (percent) over (formation, seed) scenarios and training seeds."""
    x = x.copy(); x["ts"] = x.train_seed.fillna(-1)
    M = x.pivot_table(index=["formation", "seed"], columns="ts", values=col, aggfunc="mean").values.astype(float)
    n, m = M.shape; b = np.empty(NB)
    for r in range(NB): b[r] = M[RNG.integers(0, n, n)][:, RNG.integers(0, m, m)].mean()
    return 100 * M.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


def save(fig, stem):
    for ext in ("png", "pdf"): fig.savefig(FIG / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote", stem)


def cf(d): d = d.copy(); d["cf"] = (d.success & ~d.collision).astype(float); d["succ"] = d.success.astype(float); return d


C = cf(pd.read_csv(RES / "d12_confirm.csv")); S = cf(pd.read_csv(RES / "d12_safety_final.csv")); Q = cf(pd.read_csv(RES / "d12_capacity.csv"))

# ---------------- Fig. 2: confirmatory success and collision-free success ----------------
fig, ax = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
for obs, col, mk in ((100, GREY, "s"), (50, BLUE, "o"), (25, ORANGE, "D")):
    m, lo, hi = zip(*[boot(C[(C.policy == f"GNN K{k}") & (C.obs_pct == obs)], "succ") for k in range(4)])
    ax[0].errorbar(range(4), m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], color=col, marker=mk, ms=3.5, lw=1.6, capsize=2,
                   label=f"GNN, {obs} % obs.")
for pol, ls, lab in (("relay K2", "--", "relay K = 2, no memory (25 %)"), ("relay K2+mem", ":", "relay K = 2 + memory (25 %)")):
    ax[0].axhline(100 * C[(C.policy == pol) & (C.obs_pct == 25)].succ.mean(), color=DARK, ls=ls, lw=1, label=lab)
ax[0].set(xticks=range(4), xlabel="message-passing layers K", ylabel="success (%)", ylim=(-3, 104), title="(a) Success, no safety filter")
ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6, ncol=1)
pols = [("GNN K1", "K=1"), ("GNN K2", "K=2"), ("GNN K3", "K=3"), ("GNN K2+latched", "K=2\nlatch"), ("relay K2+mem", "relay\n+mem")]
xs = np.arange(len(pols)); w = 0.38
for j, (D, col, lab) in enumerate(((C, YELLOW, "without filter"), (S, GREEN, "with filter"))):
    st = [boot(D[(D.policy == p) & (D.obs_pct == 25)], "cf") for p, _ in pols]
    m = np.array([s[0] for s in st]); lo = np.array([s[1] for s in st]); hi = np.array([s[2] for s in st])
    ax[1].bar(xs + (j - 0.5) * w, m, w * 0.95, color=col, label=lab, yerr=[m - lo, hi - m], capsize=1.8, error_kw={"lw": 0.8})
ax[1].set(xticks=xs, xticklabels=[l for _, l in pols], ylabel="collision-free success (%)", ylim=(0, 104),
          title="(b) Collision-free success, 25 % observability")
ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6.3, ncol=2); ax[1].grid(axis="x", visible=False)
fig.tight_layout(w_pad=1.2); save(fig, "ame_fig2_confirmatory")

# ---------------- Fig. 3: capacity controls ----------------
CC = pd.concat([C[C.train_seed.isin([0, 1, 2])], Q])
arms = [("GNN K0", "K = 0 (23 k)", "#c8c8c8"), ("GNN K0-wide", "K = 0, wide (65 k)", GREY), ("GNN K1", "K = 1 (44 k)", ORANGE),
        ("GNN K1-wide", "K = 1, wide (65 k)", YELLOW), ("GNN K2-shared", "K = 2, shared weights (44 k)", GREEN), ("GNN K2", "K = 2 (65 k)", BLUE)]
fig, ax = plt.subplots(figsize=(W, 2.3)); obsl = (100, 50, 25); w = 0.13
for j, (p, lab, col) in enumerate(arms):
    st = [boot(CC[(CC.policy == p) & (CC.obs_pct == o)], "succ") for o in obsl]
    m = np.array([s[0] for s in st]); lo = np.array([s[1] for s in st]); hi = np.array([s[2] for s in st])
    ax.bar(np.arange(3) + (j - 2.5) * w, m, w * 0.92, color=col, label=lab, yerr=[m - lo, hi - m], capsize=1.5, error_kw={"lw": 0.7})
ax.set(xticks=range(3), xticklabels=[f"{o} % observability" for o in obsl], ylabel="success (%)", ylim=(0, 104))
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=6.5)
save(fig, "ame_fig3_capacity")

# ---------------- Fig. 4: clearance without and with the filter ----------------
fig, ax = plt.subplots(1, 2, figsize=(W, 2.1), sharey=True)
for a, p in zip(ax, ("GNN K2", "GNN K3")):
    for D, col, lab in ((C, ORANGE, "no filter"), (S, BLUE, "safety filter")):
        v = np.sort(1000 * D[(D.policy == p) & (D.obs_pct == 25)].min_clearance.values)
        a.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=col, lw=1.6, label=lab)
    a.axvline(74, color=DARK, ls=":", lw=1); a.text(75, 0.05, "contact\n74 mm", fontsize=6.5, color=DARK)
    a.set(xlim=(70, 125), xticks=[70, 80, 90, 100, 110, 120], title=p.replace("GNN K", "GNN K = ") + ", 25 % observability")
ax[0].set_ylabel("fraction of episodes"); ax[1].legend(frameon=False, loc="lower right", fontsize=6.5)
fig.supxlabel("minimum centre distance between any two robots in the episode (mm)", fontsize=8, y=0.02)
fig.tight_layout(w_pad=1.0); save(fig, "ame_fig4_clearance")

# ---------------- Fig. 5: Study C token removal ----------------
D5 = pd.read_csv(RES / "d15_C_descriptive.csv")
fig, ax = plt.subplots(1, 2, figsize=(W, 2.2), sharey=True)
for a, var, ttl in zip(ax, ("flagoff", "flagon"), ("(a) Token and flag removed", "(b) Flag kept on, token zeroed")):
    for p, col, mk in (("GNN K1", ORANGE, "o"), ("GNN K2", BLUE, "D"), ("GNN K3", GREEN, "s"), ("GNN K2+latched", DARK, "^")):
        x = D5[(D5.policy == p) & (D5.variant == var)].sort_values("t_cut")
        if x.empty: continue
        a.errorbar(x.t_cut, x.success_pct, yerr=[x.success_pct - x.seed_min, x.seed_max - x.success_pct], color=col, marker=mk, ms=3,
                   lw=1.4, capsize=1.8, label=p.replace("GNN K", "GNN K = ").replace("+latched", ", latched"))
    a.axhline(25, color=GREY, ls=":", lw=1); a.set(title=ttl, ylim=(-3, 104))
ax[0].set_ylabel("success (%)"); ax[0].text(560, 27, "chance (1 in 4)", fontsize=6.3, color=DARK)
ax[1].legend(frameon=False, loc="upper left", fontsize=6.3)
fig.supxlabel("control step at which the token is removed", fontsize=8, y=0.02)
fig.tight_layout(w_pad=1.0); save(fig, "ame_fig5_cutoff")

# ---------------- Fig. 6: stress tests ----------------
B = pd.read_csv(RES / "d15_Bconf_descriptive.csv"); P = cf(pd.read_csv(RES / "d12_d15_pred_conf.csv")); O = cf(pd.read_csv(RES / "d12_d15_obs.csv"))
fig, ax = plt.subplots(2, 1, figsize=(W, 4.6))
pp = [("GNN K2", "GNN K = 2"), ("GNN K3", "GNN K = 3"), ("GNN K2+latched", "GNN K = 2\nlatched"), ("relayD K2+mem", "relayD\n+ memory")]
xs = np.arange(len(pp)); w = 0.2
series = [("plain, 5 steps", lambda p: B[(B.cond == "delay 5") & (B.obs_pct == 25) & (B.policy == p)].contact_pct.iloc[0], "#f2b38f"),
          ("plain, 10 steps", lambda p: B[(B.cond == "delay 10") & (B.obs_pct == 25) & (B.policy == p)].contact_pct.iloc[0], ORANGE),
          ("delay-aware, 5 steps", lambda p: 100 * P[(P.delay == 5) & (P.obs_pct == 25) & (P.policy == p)].collision.mean(), "#9fc3ee"),
          ("delay-aware, 10 steps", lambda p: 100 * P[(P.delay == 10) & (P.obs_pct == 25) & (P.policy == p)].collision.mean(), BLUE)]
for j, (lab, f, col) in enumerate(series):
    v = [f(p) for p, _ in pp]
    ax[0].bar(xs + (j - 1.5) * w, v, w * 0.92, color=col, label=lab)
    for xi, vi in zip(xs + (j - 1.5) * w, v):
        if vi < 0.5: ax[0].text(xi, 1.5, "0", ha="center", fontsize=6.3, color=col)
ax[0].set(xticks=xs, xticklabels=[l for _, l in pp], ylabel="episodes with contact (%)", ylim=(0, 100),
          title="(a) Sensing delay, 25 % observability")
ax[0].legend(frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="filter, delay", title_fontsize=6.5); ax[0].grid(axis="x", visible=False)
def cond(r):
    if r.contra: return "contra"
    if r.miss > 0: return f"miss {r.miss}"
    if r.wrong > 0: return f"wrong {r.wrong}"
    return "clean"
O["cond"] = O.apply(cond, axis=1)
conds = [("clean", "no errors"), ("miss 0.6", "missed\np = 0.6"), ("wrong 0.1", "wrong\np = 0.1"), ("wrong 0.3", "wrong\np = 0.3"), ("contra", "contradicting\nobserver")]
xs = np.arange(len(conds)); w = 0.26
for j, (p, lab, col) in enumerate((("GNN K2", "GNN K = 2 (no memory)", BLUE), ("GNN K2+latched", "GNN K = 2, latched", DARK),
                                   ("relayD K2+mem", "relayD + memory", ORANGE))):
    v = [100 * O[(O.cond == c) & (O.obs_pct == 25) & (O.policy == p)].succ.mean() for c, _ in conds]
    ax[1].bar(xs + (j - 1) * w, v, w * 0.92, color=col, label=lab)
ax[1].tick_params(axis="x", labelsize=6.8); ax[1].set(xticks=xs, xticklabels=[l for _, l in conds], ylabel="success (%)", ylim=(0, 104), title="(b) Observer errors, 25 % observability")
ax[1].legend(frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(1.0, 1.0)); ax[1].grid(axis="x", visible=False)
fig.tight_layout(h_pad=1.2); save(fig, "ame_fig6_stress")
