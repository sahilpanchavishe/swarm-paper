"""d10_episode_figures.py -- relay-vs-GNN figure (from CSVs) plus formation snapshots and a
trajectory panel (from one re-run closed-loop episode per formation, K=2, 25% observers).
Outputs to paper/figs/: figL_relay_vs_gnn, figM_snapshots, figN_trajectory.
Run from src/:  python d10_episode_figures.py
"""
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import swarm_sim as S
import d4_common as C

ROOT = Path(__file__).resolve().parent.parent
RES, RUNS, OUT = ROOT / "results", ROOT / "runs", ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
COL = {1: "#d62728", 2: "#1f77b4", 3: "#2ca02c"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# ---------- Fig L: default-slot relay vs GNN (success %, 95% CI) ----------
r = pd.read_csv(RES / "d9_relay_default.csv")
g = pd.read_csv(RES / "d6_table_success.csv")
obs_levels = [25, 50]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
w = 0.38
for ax, o in zip(axs, obs_levels):
    xs = np.arange(len(COL))
    for j, (lab, color) in enumerate([("relay (default slot)", "#999999"), ("GNN", "#1f77b4")]):
        vals, lo, hi = [], [], []
        for K in COL:
            if j == 0:
                s = r[(r.policy == f"relayD K{K}") & (r.obs_pct == o)].success
                if len(s) == 0:
                    vals.append(np.nan); lo.append(0); hi.append(0); continue
                a, b = wilson(int(s.sum()), len(s))
                vals.append(100 * s.mean())
            else:
                row = g[(g.policy == f"GNN K{K}") & (g.obs_pct == o)]
                vals.append(float(row.success_pct.iloc[0]))
                a, b = float(row.ci95_lo.iloc[0]), float(row.ci95_hi.iloc[0])
                # table CIs may be fractions or percents; normalise
                if b <= 1.0 and vals[-1] > 1.0:
                    a, b = a * 100, b * 100
            lo.append(max(vals[-1] - a, 0)); hi.append(max(b - vals[-1], 0))
        ax.bar(xs + (j - 0.5) * w, vals, w, yerr=[lo, hi], capsize=3, label=lab, color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"K={K}" for K in COL])
    ax.set_title(f"{o}% observers")
axs[0].set_ylabel("success (%)")
fig.subplots_adjust(bottom=0.25)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.savefig(OUT / "figL_relay_vs_gnn.png")
plt.close(fig)


# ---------- episode rollout (mirrors d4_common.closed_loop_episode, memoryless K=2) ----------
def load_sd(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def rollout(f, seed, K=2, obs_pct=25, tag="d5_K2_s0"):
    sd = load_sd(RUNS / f"{tag}.pt")
    torch.set_num_threads(1)
    env = C.SwarmEnv()
    model = C.GNN(K, False)
    model.load_state_dict(sd)
    model.eval()
    n_obs = {100: 8, 50: 4, 25: 2}[obs_pct]
    rng = np.random.default_rng(seed)
    p, th = S.random_start(rng, half=C.START_HALF)
    tg = S.slots(f)
    obs = np.zeros(C.N, bool)
    obs[np.random.default_rng([seed, 99]).choice(C.N, n_obs, replace=False)] = True
    env.reset(p, th)
    p, th = env.poses()
    g = torch.tensor(C.gesture_onehot(C.FORM_ID[f]))[None]
    P, E, D = [p.copy()], [S.formation_error(p, tg)], []
    with torch.no_grad():
        for _ in range(C.T_STEPS):
            A = np.array(S.comm_adjacency(p, C.R_COMM), float)
            np.fill_diagonal(A, 0)
            D.append(A.sum(1).mean())
            u = model(torch.tensor(p, dtype=torch.float32)[None], g,
                      torch.tensor(obs, dtype=torch.float32)[None])[0].numpy() * C.V_MAX
            v, w_ = S.to_unicycle(u, th)
            p, th = env.step(v, w_)
            P.append(p.copy())
            E.append(S.formation_error(p, tg))
    return np.array(P), np.array(E), np.array(D), tg, obs


SEED = 10000
runs = {f: rollout(f, SEED) for f in S.FORMATIONS}
dt = C.DT_CTRL if hasattr(C, "DT_CTRL") else 0.032

# ---------- Fig M: start / final snapshots for the four formations ----------
fig, axs = plt.subplots(2, 4, figsize=(11, 5.6), sharex=True, sharey=True)
for j, f in enumerate(S.FORMATIONS):
    P, E, D, tg, obs = runs[f]
    for i, (row, t, ttl) in enumerate([(0, 0, "start"), (1, -1, "final")]):
        ax = axs[row, j]
        ax.scatter(*tg.T, marker="x", color="k", s=40, label="target slot")
        ax.scatter(*P[t].T, c=np.where(obs, "#d62728", "#1f77b4"), s=60, zorder=3)
        ax.set_aspect("equal")
        ax.set_title(f"{f}, {ttl}")
fig.subplots_adjust(bottom=0.14)
from matplotlib.lines import Line2D
hs = [Line2D([], [], marker="o", ls="", color="#d62728", label="observer"),
      Line2D([], [], marker="o", ls="", color="#1f77b4", label="non-observer"),
      Line2D([], [], marker="x", ls="", color="k", label="target slot")]
fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False)
fig.savefig(OUT / "figM_snapshots.png")
plt.close(fig)

# ---------- Fig N: trajectories, error and degree for the split formation ----------
P, E, D, tg, obs = runs["split"]
t = np.arange(len(E)) * dt
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
for i in range(P.shape[1]):
    axs[0].plot(P[:, i, 0], P[:, i, 1], lw=1, color="#d62728" if obs[i] else "#1f77b4")
axs[0].scatter(*P[0].T, marker="o", facecolors="none", edgecolors="k")
axs[0].scatter(*tg.T, marker="x", color="k")
axs[0].set_aspect("equal")
axs[0].set_title("trajectories (split, K=2, 25% obs.)")
axs[1].plot(t, E)
axs[1].set_xlabel("time (s)")
axs[1].set_ylabel("formation error (m)")
axs[1].set_title("formation error")
axs[2].plot(t[1:], D)
axs[2].set_xlabel("time (s)")
axs[2].set_ylabel("mean degree")
axs[2].set_title("communication graph")
fig.tight_layout()
fig.savefig(OUT / "figN_trajectory.png")
plt.close(fig)

print("final errors:", {f: round(float(runs[f][1][-1]), 4) for f in S.FORMATIONS})
print("saved figL, figM, figN to", OUT)
