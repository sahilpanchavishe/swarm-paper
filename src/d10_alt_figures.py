"""Alternative figure set: the additional paper figures, built from results/*.csv and from re-simulated example episodes of the committed models.
CSV figures (exact data, no re-simulation):
  figG_training.png        epoch validation RMSE (plain imitation) and DAgger success per round    <- d4_train_log.csv, d5_dagger_log.csv
  figH_token_cutoff.png    success when the token is removed during the episode (flag-off variant) <- d9_token_cutoff.csv
  figI_relay_default.png   GNN vs hold-still relay vs default-slot relay at 25 % observability      <- d6_table_success.csv, d9_relay_default.csv
  figK_connectivity.png / figM_model_validation.png   copies of figures/connectivity.png and figures/model_validation.png
Illustration figures (one re-simulated episode of d5_K2_s0, 25 % observability, seed 10000; NOT statistics, nothing is quoted from them):
  figJ_snapshots.png       start pose and final pose for the four formations
  figL_trajectory.png      one episode: paths, error vs time, token reach and graph degree vs time
  figN_schematic.png       method overview drawing (no data)
The example episodes use a numpy copy of the GNN forward pass (npgnn.py); it reads the same runs/*.pt files and needs no PyTorch.
Run: cd src && python d10_alt_figures.py        (venv active; about 30 s)"""
import shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent; RES = ROOT / "results"; FIG = ROOT / "paper" / "figs" / "alt"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8.5})
CK = {0: "#999999", 1: "#6baed6", 2: "#2171b5", 3: "#08306b"}
SEED = 10000; EXAMPLE = "d5_K2_s0"; EP_FORMATION = "line"


def fig_training():
    t = pd.read_csv(RES / "d4_train_log.csv"); g = pd.read_csv(RES / "d5_dagger_log.csv")
    t["K"] = t.model.str.extract(r"K(\d)")[0].astype(int); t["lat"] = t.model.str.contains("latched"); t["seed"] = t.model.str.extract(r"_s(\d)$")[0].astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for K in (0, 1, 2, 3):
        d = t[(t.K == K) & ~t.lat]
        m = d.groupby("epoch").val_rmse_ms.agg(["mean", "min", "max"])
        ax[0].plot(m.index, m["mean"], color=CK[K], label=f"K={K}"); ax[0].fill_between(m.index, m["min"], m["max"], color=CK[K], alpha=0.2)
    d = t[t.lat].groupby("epoch").val_rmse_ms.mean(); ax[0].plot(d.index, d.values, color="#e6550d", ls="--", label="K=2 latched")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation RMSE of velocity (m/s)"); ax[0].set_title("(a) Plain imitation: validation error"); ax[0].legend(frameon=False, ncol=2)
    g["K"] = g.model.str.extract(r"K(\d)")[0].astype(int); g["lat"] = g.model.str.contains("latched")
    for (K, lat), d in g[g.obs_pct == 25].groupby(["K", "lat"]):
        col = "#e6550d" if lat else CK[K]
        for m, dm in d.groupby("model"): ax[1].plot(dm.sort_values("round")["round"], dm.sort_values("round").success_pct, color=col, alpha=0.35, lw=0.8)
        mm = d.groupby("round").success_pct.mean(); ax[1].plot(mm.index, mm.values, marker="o", ms=3, color=col, ls="--" if lat else "-", label=f"K={K}" + (" latched" if lat else "") + f" ({d.model.nunique()} seed{'s' if d.model.nunique() > 1 else ''})")
    ax[1].set_xlabel("DAgger round"); ax[1].set_ylabel("success at 25 % observability (%)\n(20 training-time episodes per point)"); ax[1].set_title("(b) DAgger: success per round (thin = single seed)"); ax[1].set_xticks(sorted(g["round"].unique())); ax[1].legend(frameon=False, fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.45))
    fig.tight_layout(); fig.savefig(FIG / "figG_training.png", dpi=300); plt.close(fig)


def fig_cutoff():
    c0 = pd.read_csv(RES / "d9_token_cutoff.csv"); c0 = c0[c0.obs_pct == 25]
    c = pd.concat([c0[(c0.variant == "flagoff") & (c0.t_cut < 1100)], c0[(c0.variant == "flagon") & (c0.t_cut == 1100)]])   # t_cut = 1100: token never removed (same input as the normal run)
    ts = [0, 300, 700, 1100]; fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for K, ls in ((1, "-"), (2, "-"), (3, "-")):
        d = c[c.K == K]
        y = [100 * d[d.t_cut == t].success.mean() for t in ts]; ax.plot(range(4), y, marker="o", color=CK[K], label=f"K={K} ({d.model.nunique()} network{'s' if d.model.nunique() > 1 else ''})", ls=ls)
        if K == 2:
            for m, dm in d.groupby("model"): ax.plot(range(4), [100 * dm[dm.t_cut == t].success.mean() for t in ts], color=CK[2], alpha=0.25, lw=0.8)
    ax.axhline(25, color="k", ls=":", lw=0.8); ax.text(1.0, 12, "dotted line = chance (1 in 4)", ha="left", fontsize=6.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(["0\n(never told)", "300", "700", "never\nremoved"]); ax.set_xlabel("step at which the token is removed"); ax.set_ylabel("success (%)")
    ax.set_ylim(0, 105); ax.legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.45)); ax.set_title("Token removed during the episode\n(25 % observability)")
    fig.tight_layout(); fig.savefig(FIG / "figH_token_cutoff.png", dpi=300); plt.close(fig)


def fig_relay_default():
    S = pd.read_csv(RES / "d6_table_success.csv"); R = pd.read_csv(RES / "d9_relay_default.csv"); R["K"] = R.policy.str.extract(r"K(\d)")[0].astype(int)
    Ks = [1, 2, 3]; fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, pct in zip(axs, (25, 50)):
        w = 0.26; x = np.arange(3)
        hold = [S[(S.policy == f"relay K{K}") & (S.obs_pct == pct)].success_pct.iloc[0] for K in Ks]
        dflt = [100 * R[(R.K == K) & (R.obs_pct == pct)].success.mean() for K in Ks]
        gnn = [S[(S.policy == f"GNN K{K}") & (S.obs_pct == pct)].success_pct.iloc[0] for K in Ks]
        ax.bar(x - w, hold, w, color="#bdbdbd", label="relay, hold still"); ax.bar(x, dflt, w, color="#fdae6b", label="relay, default slot (exploratory)"); ax.bar(x + w, gnn, w, color="#2171b5", label="GNN (DAgger)")
        ax.set_xticks(x); ax.set_xticklabels([f"K={K}" for K in Ks]); ax.set_title(f"{pct} % observability"); ax.set_ylim(0, 105)
    axs[0].set_ylabel("success (%)"); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(FIG / "figI_relay_default.png", dpi=300); plt.close(fig)


def copy_existing():
    for a, b in (("connectivity.png", "figK_connectivity.png"), ("model_validation.png", "figM_model_validation.png")):
        shutil.copy(ROOT / "figures" / a, FIG / b)


# ---------------- illustration figures (one re-simulated episode) ----------------
def example_episodes():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import npgnn, rollout_np
    from mj_env import SwarmEnv
    sd = npgnn.load_pt(ROOT / "runs" / f"{EXAMPLE}.pt"); env = SwarmEnv(); out = {}
    for f in ("circle", "line", "cluster", "split"): out[f] = rollout_np.run(sd, 2, False, f, SEED, 25, env, record=True)
    return out


def draw_robots(ax, P, th=None, obs=None, color="#2171b5", alpha=1.0):
    for i, p in enumerate(P):
        ax.add_patch(Circle(p, 0.037, fc=("#e6550d" if (obs is not None and obs[i]) else color), ec="k", lw=0.4, alpha=alpha))
        ax.text(p[0], p[1], str(i), ha="center", va="center", fontsize=5.5, color="w")
    ax.set_aspect("equal"); ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.tick_params(labelsize=6)


def fig_snapshots(E):
    import swarm_sim as S
    fig, axs = plt.subplots(1, 5, figsize=(7.4, 1.95))
    e0 = E["line"]; draw_robots(axs[0], e0["traj"][0], obs=e0["obs"]); axs[0].set_title("(a) start")
    for k, (ax, f) in enumerate(zip(axs[1:], ("circle", "line", "cluster", "split"))):
        for s in S.slots(f): ax.add_patch(Circle(s, 0.037, fc="none", ec="0.5", lw=0.6, ls="--"))
        draw_robots(ax, E[f]["pf"], obs=E[f]["obs"]); ax.set_title(f"({'bcde'[k]}) {f}, end")
    for ax in axs[1:]: ax.set_yticklabels([])
    axs[0].set_ylabel("y (m)"); [a.set_xlabel("x (m)") for a in axs]
    fig.text(0.5, 0.0, "orange = observers (see the gesture); dashed circles = target slots; numbers = robot role id", ha="center", fontsize=6.5, style="italic"); fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG / "figJ_snapshots.png", dpi=300); plt.close(fig)


def fig_trajectory(E):
    import swarm_sim as S
    e = E[EP_FORMATION]; tr = e["traj"]; t = np.arange(len(tr)) * 5 * 0.032
    fig = plt.figure(figsize=(7.4, 3.0)); gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])
    a0 = fig.add_subplot(gs[0]); cm = plt.get_cmap("tab10")
    for i in range(8):
        a0.plot(tr[:, i, 0], tr[:, i, 1], color=cm(i), lw=0.9); a0.plot(*tr[0, i], "o", color=cm(i), ms=3); a0.plot(*tr[-1, i], "s", color=cm(i), ms=4, mec="k", mew=0.4)
    for s in S.slots(EP_FORMATION): a0.add_patch(Circle(s, 0.037, fc="none", ec="0.5", lw=0.6, ls="--"))
    a0.set_aspect("equal"); a0.set_xlim(-1, 1); a0.set_ylim(-1, 1); a0.set_title(f"(a) paths, {EP_FORMATION} formation\n(circle = start, square = end)"); a0.set_xlabel("x (m)"); a0.set_ylabel("y (m)")
    a1 = fig.add_subplot(gs[1]); a1.semilogy(t, e["err"], color="0.7", lw=0.7); a1.semilogy(t, e["err"].mean(1), color="k", lw=1.4, label="mean over robots"); a1.axhline(0.025, color="r", ls="--", lw=0.8, label="success threshold 25 mm")
    a1.set_xlabel("time (s)"); a1.set_ylabel("distance to own slot (m)"); a1.set_title("(b) formation error"); a1.legend(frameon=False, fontsize=6.5)
    a2 = fig.add_subplot(gs[2]); a2.plot(t, e["know"].sum(1), color="#2171b5", label="robots reached by the token (K=2)"); a2.set_ylabel("robots reached (of 8)"); a2.set_ylim(0, 8.5); a2.set_xlabel("time (s)")
    b = a2.twinx(); b.plot(t, e["deg"].mean(1), color="#e6550d", ls="--", label="mean graph degree"); b.set_ylabel("mean neighbours (R = 0.5 m)"); b.spines["right"].set_visible(True)
    h1, l1 = a2.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels(); a2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.25)); a2.set_title("(c) communication graph")
    fig.tight_layout(); fig.savefig(FIG / "figL_trajectory.png", dpi=300); plt.close(fig)


def fig_schematic():
    fig, ax = plt.subplots(figsize=(7.4, 3.0)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    def box(x, y, w, h, txt, fc="#deebf7"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec="k", lw=0.8)); ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.2)
    def arr(x1, y1, x2, y2, txt=None, dy=1.6):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="k"))
        if txt: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, txt, ha="center", fontsize=6.3)
    box(1, 27, 17, 10, "Human gesture\n(1 of 4 formations:\ncircle, line,\ncluster, split)", "#fee6ce")
    box(24, 27, 20, 10, "Observers (2, 4 or 8 of 8)\nsee the gesture:\ntoken + flag = 1", "#fdd0a2")
    box(24, 3, 20, 14, "Other robots:\ntoken = 0, flag = 0\nreceive it only through\nmessages from\nneighbours (R = 0.5 m)")
    box(52, 8, 22, 24, "Per-robot encoder\n+ K rounds of message\npassing (K = 1, 2, 3)\n\ninputs: own position,\nrole id, token, flag;\nmessages: neighbour state\n+ relative position", "#c6dbef")
    box(81, 14, 17, 12, "Velocity command\n(vx, vy) -> unicycle\n(v, w) -> wheels\nin MuJoCo", "#e5f5e0")
    arr(18, 32, 24, 32); arr(44.3, 31, 51.7, 27, "token", 1.8); arr(44.3, 10, 51.7, 13, "no token", -3.4); arr(74, 20, 81, 20)
    ax.annotate("", xy=(34, 17.3), xytext=(34, 26.5), arrowprops=dict(arrowstyle="<->", lw=0.9, color="#e6550d")); ax.text(35.5, 21.7, "hops over\nthe graph", fontsize=6.3, color="#e6550d", va="center")
    ax.text(50, 0.5, "Training: DAgger, expert = potential-field controller that knows the gesture for every robot. Test: 50 seeds x 4 formations.", ha="center", fontsize=6.5, style="italic")
    fig.tight_layout(); fig.savefig(FIG / "figN_schematic.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    fig_training(); fig_cutoff(); fig_relay_default(); copy_existing(); fig_schematic(); print("CSV figures + schematic written")
    try:
        E = example_episodes(); fig_snapshots(E); fig_trajectory(E); print("example-episode figures written")
    except ImportError as e:
        print("skipped example-episode figures:", e)
    print("saved to", FIG)
