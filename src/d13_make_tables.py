"""Day 13: corrected paper tables and the clearance figure, from the committed raw CSVs only. No simulation, no training.

  python d13_make_tables.py

Writes (new names; the Day 8 tables are left untouched so the d8/d9/d10 checkers keep working):
  paper/tables/table1_success_d13.csv / .md   success with TWO-LEVEL bootstrap 95 % intervals (test scenarios and training seeds
                                              resampled), contact, collision-free success, median minimum centre distance
  paper/tables/table2_paired_d13.csv / .md    the 16 pre-specified comparisons from results/d11_paired.csv (exact McNemar per
                                              training seed, max p over seeds, Holm), with the per-seed differences
  paper/figs/figP_clearance.png / .pdf        distribution of the minimum centre distance per episode, 25 % observability
Sources: results/d6_main_eval.csv (DAgger models d5_*, training seeds 0-2, test seeds 10000-10049), results/d3_baselines.csv
(oracle and relays, same episodes), results/d11_paired.csv (Day 11a statistics)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES, TAB, FIG = ROOT / "results", ROOT / "paper" / "tables", ROOT / "paper" / "figs"
CONTACT_M = 0.074          # centre distance below two robot radii = contact (protocol.md)
NBOOT, SEED = 5000, 0
OBS = (100, 50, 25)


def load():
    g = pd.read_csv(RES / "d6_main_eval.csv")
    g = g[g.method == "DAgger"].copy()
    g["policy"] = np.where(g.latched, "GNN K" + g.K.astype(str) + "+latched", "GNN K" + g.K.astype(str))
    b = pd.read_csv(RES / "d3_baselines.csv").copy()
    b["policy"] = np.where(b.policy == "oracle", "oracle", "relay " + b.policy)
    b["train_seed"] = -1
    cols = ["policy", "obs_pct", "formation", "seed", "train_seed", "success", "collision", "min_clearance", "final_err"]
    return pd.concat([g[cols], b[cols]], ignore_index=True)


def boot_ci(x, rng):
    """Two-level bootstrap of the mean: resample (formation, seed) scenarios, and training seeds. Relays have one 'seed' (-1)."""
    W = x.pivot_table(index=["formation", "seed"], columns="train_seed", values="v", aggfunc="mean").values.astype(float)
    n, m = W.shape; out = np.empty(NBOOT)
    for r in range(NBOOT):
        out[r] = W[rng.integers(0, n, n)][:, rng.integers(0, m, m)].mean()
    return np.percentile(out, [2.5, 97.5])


ORDER = ["oracle"] + [f"relay K{k}" for k in (0, 1, 2, 3, 7)] + [f"relay K{k}+mem" for k in (1, 2, 3, 7)] + \
        [f"GNN K{k}" for k in (0, 1, 2, 3)] + ["GNN K2+latched"]


def table1(d):
    rng = np.random.default_rng(SEED); rows = []
    for pol in ORDER:
        for pct in OBS:
            x = d[(d.policy == pol) & (d.obs_pct == pct)]
            if x.empty: continue
            lo, hi = boot_ci(x.assign(v=x.success.astype(float)), rng)
            rows.append(dict(policy=pol, obs_pct=pct, episodes=len(x), training_seeds=x.train_seed[x.train_seed >= 0].nunique(),
                             success_pct=100 * x.success.mean(), ci_lo=100 * lo, ci_hi=100 * hi,
                             contact_pct=100 * x.collision.mean(),
                             collision_free_success_pct=100 * (x.success & ~x.collision.astype(bool)).mean(),
                             median_min_dist_mm=1000 * x.min_clearance.median(), min_min_dist_mm=1000 * x.min_clearance.min()))
    return pd.DataFrame(rows)


def md_table1(T):
    L = ["| Policy | Obs. | Episodes | Success % [95 % CI] | Contact % | Collision-free success % | Median min. distance (mm) |",
         "|---|---|---|---|---|---|---|"]
    for r in T.itertuples():
        L.append(f"| {r.policy} | {r.obs_pct} % | {r.episodes} | {r.success_pct:.1f} [{r.ci_lo:.1f}, {r.ci_hi:.1f}] | "
                 f"{r.contact_pct:.1f} | {r.collision_free_success_pct:.1f} | {r.median_min_dist_mm:.1f} |")
    L += ["", "Intervals: two-level bootstrap (5000 resamples of test scenarios and training seeds; relays and the oracle are deterministic, "
          "so only scenarios are resampled). GNN rows: 3 training seeds x 200 episodes. Contact = centre distance below 74 mm at any step."]
    return "\n".join(L) + "\n"


def pfmt(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def table2():
    P = pd.read_csv(RES / "d11_paired.csv")
    L = ["| Comparison | Δ success, points [95 % CI] | Per-seed Δ (seeds 0, 1, 2) | Holm p (success) | Δ final error (mm) | Holm p (final error) |",
         "|---|---|---|---|---|---|"]
    for r in P.itertuples():
        seeds = ", ".join(f"{float(s):.1f}" for s in str(r.per_seed_dsuccess_pct).split(";"))
        L.append(f"| {r.comparison} | {r.dsuccess_pct:.1f} [{r.ci_lo:.1f}, {r.ci_hi:.1f}] | {seeds} | {pfmt(r.p_success_holm)} | "
                 f"{1000 * r.dfinal_err_m:.1f} | {pfmt(r.p_final_err_holm)} |")
    L += ["", "Exact McNemar test (success) and Wilcoxon signed-rank test (final error), run separately for each training seed; the largest "
          "p over seeds is Holm-corrected over the 16 comparisons. Intervals: two-level bootstrap. Source: results/d11_paired.csv (src/d11_stats.py)."]
    return P, "\n".join(L) + "\n"


def figure(d):
    pols = [("GNN K1", "#2a78d6"), ("GNN K2", "#eb6834"), ("GNN K3", "#1baf7a"), ("GNN K2+latched", "#eda100")]
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for pol, col in pols:
        x = np.sort(1000 * d[(d.policy == pol) & (d.obs_pct == 25)].min_clearance.values)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=col, lw=2, label=pol)
    ax.axvline(1000 * CONTACT_M, color="#555555", lw=1, ls="--")
    ax.text(1000 * CONTACT_M + 0.8, 0.40, "contact\n(74 mm)", color="#333333", fontsize=7)
    ax.set_xlim(70, 130); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Minimum centre distance between any two robots in the episode (mm)")
    ax.set_ylabel("Share of episodes at or below")
    ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(FIG / f"figP_clearance.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    T1 = table1(d); T1.round(2).to_csv(TAB / "table1_success_d13.csv", index=False)
    (TAB / "table1_success_d13.md").write_text(md_table1(T1))
    P, md2 = table2(); P.to_csv(TAB / "table2_paired_d13.csv", index=False); (TAB / "table2_paired_d13.md").write_text(md2)
    figure(d)
    print(T1.round(1).to_string(index=False))
    print("\nwrote paper/tables/table1_success_d13.{csv,md}, paper/tables/table2_paired_d13.{csv,md}, paper/figs/figP_clearance.{png,pdf}")
