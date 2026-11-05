"""d11_extra_figures.py -- extra paper figures, made only from results/*.csv.
Outputs to paper/figs/: figG_training_curves, figH_dagger_rounds, figI_token_cutoff.
Run from src/:  python d11_extra_figures.py
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES, OUT = ROOT / "results", ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
COL = {0: "#7f7f7f", 1: "#d62728", 2: "#1f77b4", 3: "#2ca02c"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def parse(name):
    m = re.match(r"d\d_K(\d)(_latched)?_s(\d)", name)
    return int(m.group(1)), bool(m.group(2)), int(m.group(3))


def legend_below(fig, ax, ncol=4):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, -0.08))


# ---- Fig G: imitation training curves (validation RMSE vs epoch) ----
d = pd.read_csv(RES / "d4_train_log.csv")
d[["K", "latched", "seed"]] = d["model"].apply(lambda s: pd.Series(parse(s)))
d = d[~d.latched]
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for K, g in d.groupby("K"):
    piv = g.pivot_table(index="epoch", columns="seed", values="val_rmse_ms")
    m = piv.mean(axis=1)
    ax.plot(m.index, m.values, color=COL[K], label=f"K={K}")
    ax.fill_between(m.index, piv.min(axis=1), piv.max(axis=1),
                    color=COL[K], alpha=0.15)
ax.set_xlabel("epoch")
ax.set_ylabel("validation RMSE (m/s)")
ax.set_yscale("log")
fig.subplots_adjust(bottom=0.25)
legend_below(fig, ax)
fig.savefig(OUT / "figG_training_curves.png")
plt.close(fig)

# ---- Fig H: DAgger rounds (success vs round, per training seed) ----
d = pd.read_csv(RES / "d5_dagger_log.csv")
d[["K", "latched", "seed"]] = d["model"].apply(lambda s: pd.Series(parse(s)))
d = d[~d.latched]
obs_levels = sorted(d.obs_pct.unique())
fig, axs = plt.subplots(1, len(obs_levels), figsize=(5 * len(obs_levels) / 1.4, 3.6),
                        sharey=True, squeeze=False)
for ax, o in zip(axs[0], obs_levels):
    for (K, s), g in d[d.obs_pct == o].groupby(["K", "seed"]):
        g = g.sort_values("round")
        ax.plot(g["round"], g["success_pct"], color=COL[K], alpha=0.6, lw=1,
                label=f"K={K}" if s == 0 else None)
    ax.set_title(f"{int(o)}% observers")
    ax.set_xlabel("DAgger round")
axs[0][0].set_ylabel("success (%) on 20 episodes")
fig.subplots_adjust(bottom=0.25)
legend_below(fig, axs[0][0])
fig.savefig(OUT / "figH_dagger_rounds.png")
plt.close(fig)

# ---- Fig I: token cut-off test (exploratory) ----
d = pd.read_csv(RES / "d9_token_cutoff.csv")
d = d[(~d["latched"]) & (d["obs_pct"] == 25)].copy()
d["success"] = d["success"].astype(float) * 100
print("variants:", d.variant.unique(), " t_cut:", sorted(d.t_cut.unique()))
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, var in zip(axs, ["flagoff", "flagon"]):
    for K, g in d[d.variant == var].groupby("K"):
        m = g.groupby("t_cut").success.mean()
        ax.plot(m.index, m.values, marker="o", color=COL[K], label=f"K={K}")
    ax.set_title("token and flag removed" if var == "flagoff"
                 else "flag kept on, token zeroed")
    ax.set_xlabel("cut-off step")
axs[0].set_ylabel("success (%) at 25% observers")
fig.subplots_adjust(bottom=0.25)
legend_below(fig, axs[0])
fig.savefig(OUT / "figI_token_cutoff.png")
plt.close(fig)

print("saved figG, figH, figI to", OUT)
