"""Day 8: every table and figure of the paper, built ONLY from files in ../results (no number is typed by hand).
Inputs : results/d6_table_success.csv, d6_table_paired.csv   (written by d6_analysis.py; re-run it after new training seeds)
         results/d6_main_eval.csv                            (raw GNN episodes: DAgger and plain behaviour cloning)
         results/d7_token_ablation.csv                       (raw token-ablation episodes)
         results/token_coverage_vs_K.csv, results/model_validation.csv
Outputs: paper/figs/*.png (300 dpi) and paper/tables/*.csv + *.md  (the .md files paste straight into Word/LaTeX helpers)
Run   : cd src && python d8_make_figures.py          (venv active; takes a few seconds)
Order : python d6_main_eval.py -> python d6_analysis.py -> python d8_make_figures.py   (the last step always reflects the newest seeds)"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
COL = {"oracle": "#222222", "relay K1": "#d9d9d9", "relay K2": "#969696", "relay K3": "#636363",
       "GNN K1": "#6baed6", "GNN K2": "#2171b5", "GNN K3": "#08306b", "GNN K2+latched": "#e6550d"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def md_table(df, path, floatfmt=1):
    """Write a CSV and a pipe-table markdown copy of df."""
    df.to_csv(path.with_suffix(".csv"), index=False)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def fmt_p(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def table1(S, out):
    """Table 1: success % with 95% CI, mean final error, contact rate, per policy and observability."""
    rows = []
    order = ["oracle", "relay K1", "relay K2", "relay K3", "relay K7", "relay K1+mem", "relay K2+mem", "relay K3+mem",
             "GNN K0", "GNN K1", "GNN K2", "GNN K3", "GNN K2+latched"]
    for pol in order:
        r = {"Policy": pol}
        for pct in (100, 50, 25):
            x = S[(S.policy == pol) & (S.obs_pct == pct)]
            if len(x):
                x = x.iloc[0]
                r[f"{pct}% success [95% CI]"] = f"{x.success_pct:.1f} [{x.ci95_lo:.1f}, {x.ci95_hi:.1f}]"
                r[f"{pct}% contact"] = f"{x.contact_pct:.1f}"
            else:
                r[f"{pct}% success [95% CI]"] = "–"; r[f"{pct}% contact"] = "–"
        rows.append(r)
    md_table(pd.DataFrame(rows), out / "table1_success", 1)


def table2(P, out):
    """Table 2: paired comparisons with Holm-corrected p-values (difference = A minus B)."""
    t = pd.DataFrame({
        "Comparison": P.comparison, "n pairs": P.n_pairs,
        "Δ success (pts) [95% CI]": [f"{a:.1f} [{b:.1f}, {c:.1f}]" for a, b, c in zip(P.dsuccess_pct, P.dsuccess_ci_lo, P.dsuccess_ci_hi)],
        "Holm p (success)": [fmt_p(p) for p in P.p_success_holm],
        "Δ final error (mm)": [f"{1000 * x:.1f}" for x in P.dfinal_err_m],
        "Holm p (final error)": [fmt_p(p) for p in P.p_final_err_holm]})
    md_table(t, out / "table2_paired", 1)


def table_validation(V, out):
    t = V.copy()
    t.columns = ["Case", "Ideal path (m)", "Measured path (m)", "Path error (%)", "End-position error (m)", "Ideal yaw (rad)", "Measured yaw (rad)", "Yaw error (%)"]
    md_table(t, out / "table3_model_validation", 3)


def fig_success_vs_K(S, figs):
    """Fig A: success vs number of message-passing layers / relay hops, at 25% and 50% observability."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, pct in zip(axs, (25, 50)):
        def line(prefix, ks, label, colr, ls, mk):
            ys = []
            for k in ks:
                x = S[(S.policy == f"{prefix} K{k}") & (S.obs_pct == pct)]
                ys.append(x.iloc[0] if len(x) else None)
            ok = [(k, y) for k, y in zip(ks, ys) if y is not None]
            if not ok: return
            ax.errorbar([k for k, _ in ok], [y.success_pct for _, y in ok],
                        yerr=[[y.success_pct - y.ci95_lo for _, y in ok], [y.ci95_hi - y.success_pct for _, y in ok]],
                        label=label, color=colr, ls=ls, marker=mk, capsize=2, lw=1.2, ms=4)
        line("GNN", [0, 1, 2, 3], "GNN (DAgger)", COL["GNN K2"], "-", "o")
        line("relay", [0, 1, 2, 3], "relay, no memory", COL["relay K3"], "--", "s")
        o = S[(S.policy == "oracle")]
        if len(o): ax.axhline(o.iloc[0].success_pct, color=COL["oracle"], lw=0.8, ls=":", label="oracle")
        x = S[(S.policy == "GNN K2+latched") & (S.obs_pct == pct)]
        if len(x): ax.scatter([2], [x.iloc[0].success_pct], color=COL["GNN K2+latched"], marker="D", zorder=5, label="GNN K=2 + latched token")
        ax.set_title(f"{pct}% observability"); ax.set_xlabel("K (layers or hops)"); ax.set_xticks([0, 1, 2, 3]); ax.set_ylim(-3, 105)
    axs[0].set_ylabel("success (%)"); axs[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(figs / "figA_success_vs_K.png", dpi=300); plt.close(fig)


def fig_main_bars(S, figs):
    pols = [p for p in ["oracle", "relay K1", "relay K2", "relay K3", "GNN K1", "GNN K2", "GNN K3", "GNN K2+latched"] if p in set(S.policy)]
    fig, ax = plt.subplots(figsize=(9, 3.4)); w = 0.8 / len(pols); labelled = set()
    for j, pol in enumerate(pols):
        for i, pct in enumerate((100, 50, 25)):
            r = S[(S.policy == pol) & (S.obs_pct == pct)]
            if len(r):
                r = r.iloc[0]
                ax.bar(i + j * w, r.success_pct, w, color=COL[pol], label=pol if pol not in labelled else None,
                       yerr=[[r.success_pct - r.ci95_lo], [r.ci95_hi - r.success_pct]], capsize=1.5, error_kw=dict(lw=0.7))
                labelled.add(pol)
    ax.set_xticks([i + 0.4 - w / 2 for i in range(3)]); ax.set_xticklabels(["100%", "50%", "25%"])
    ax.set_xlabel("observability"); ax.set_ylabel("success (%)"); ax.set_ylim(0, 108)
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2)); fig.tight_layout(); fig.savefig(figs / "figB_main_success.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_token_coverage(figs):
    """Fig C: share of robots reached by the token within K hops (start box 1.0 m = protocol), 2 and 4 observers."""
    t = pd.read_csv(RES / "token_coverage_vs_K.csv"); t = t[np.isclose(t.start_half, 0.5)]
    if t.empty: print("  (no start_half=0.5 rows in token_coverage_vs_K.csv; Fig C skipped)"); return
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for n_obs, c in ((2, "#e6550d"), (4, "#2171b5")):
        x = t[t.observers == n_obs]
        ax.plot(x.K, x.mean_pct_reached, marker="o", ms=4, color=c, label=f"{n_obs} observers ({int(100 * n_obs / 8)}%): robots reached")
        ax.plot(x.K, x.pct_episodes_all_reached, marker="s", ms=4, ls="--", color=c, label=f"{n_obs} observers: episodes with all reached")
    ax.set_xlabel("K hops"); ax.set_ylabel("%"); ax.set_ylim(0, 102); ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(figs / "figC_token_coverage.png", dpi=300); plt.close(fig)


def fig_ablation(figs, out):
    """Fig D: token ablation. Normal / blank / wrong token; success against the TRUE and against the SHOWN formation."""
    d = pd.read_csv(RES / "d7_token_ablation.csv")
    g = d.groupby(["model", "obs_pct", "mode"]).agg(true=("success_true", "mean"), shown=("success_shown", "mean"),
                                                   nonobs=("nonobs_follow_shown", "mean"), n=("success_true", "size")).reset_index()
    tab = g.copy()
    for c in ("true", "shown", "nonobs"): tab[c] = 100 * tab[c]
    tab.columns = ["Model", "Obs. %", "Token mode", "Success vs true (%)", "Success vs shown (%)", "Non-observers at shown slot (%)", "n episodes"]
    md_table(tab, out / "table4_token_ablation", 1)
    x = g[g.obs_pct == 25]; models = list(x.model.unique())
    fig, ax = plt.subplots(figsize=(7, 3.2)); w = 0.25
    for j, (mode, lab, c) in enumerate((("normal", "true token: success vs true", "#2171b5"), ("blank", "blank token: success vs true", "#bdbdbd"),
                                        ("wrong", "wrong token: success vs SHOWN", "#e6550d"))):
        v = [100 * (x[(x.model == m) & (x["mode"] == mode)]["shown" if mode == "wrong" else "true"].iloc[0]) for m in models]
        ax.bar(np.arange(len(models)) + (j - 1) * w, v, w, color=c, label=lab)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([m.replace("d5_", "").replace("_s0", "").replace("_", " ") for m in models])
    ax.set_ylabel("success (%)"); ax.set_ylim(0, 108); ax.set_title("Gesture-token ablation, 25% observability", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3); fig.tight_layout(); fig.savefig(figs / "figD_token_ablation.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_contact(S, figs):
    """Fig E: share of episodes in which two robots touched (centre distance < 0.074 m)."""
    pols = [p for p in ["oracle", "relay K2", "GNN K1", "GNN K2", "GNN K3", "GNN K2+latched"] if p in set(S.policy)]
    fig, ax = plt.subplots(figsize=(7, 3.2)); w = 0.8 / 3
    for i, (pct, c) in enumerate(((100, "#9ecae1"), (50, "#4292c6"), (25, "#08519c"))):
        v = [S[(S.policy == p) & (S.obs_pct == pct)].contact_pct.iloc[0] if len(S[(S.policy == p) & (S.obs_pct == pct)]) else np.nan for p in pols]
        ax.bar(np.arange(len(pols)) + (i - 1) * w, v, w, color=c, label=f"{pct}% observability")
    ax.set_xticks(range(len(pols))); ax.set_xticklabels(pols, rotation=15); ax.set_ylabel("episodes with contact (%)"); ax.set_ylim(0, 100)
    ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(figs / "figE_contact.png", dpi=300); plt.close(fig)


def fig_dagger_vs_bc(figs, out):
    """Fig F: plain behaviour cloning vs DAgger (same networks' architecture, same test episodes)."""
    d = pd.read_csv(RES / "d6_main_eval.csv")
    if "method" not in d or d.method.nunique() < 2: print("  (no BC rows in d6_main_eval.csv; Fig F skipped. Run: python d6_main_eval.py --prefix d4)"); return
    d["name"] = d.apply(lambda r: f"K{r.K}" + ("+latched" if r.latched else ""), axis=1)
    g = d.groupby(["name", "method", "obs_pct"]).agg(success=("success", "mean"), contact=("collision", "mean")).reset_index()
    tab = g.copy(); tab["success"] *= 100; tab["contact"] *= 100
    tab.columns = ["GNN", "Training", "Obs. %", "Success (%)", "Contact episodes (%)"]
    md_table(tab, out / "table5_bc_vs_dagger", 1)
    x = g[g.obs_pct == 25]; names = sorted(x.name.unique())
    fig, ax = plt.subplots(figsize=(6, 3.2)); w = 0.38
    for j, (m, c) in enumerate((("BC", "#bdbdbd"), ("DAgger", "#2171b5"))):
        v = [100 * x[(x.name == n) & (x.method == m)].success.iloc[0] if len(x[(x.name == n) & (x.method == m)]) else np.nan for n in names]
        ax.bar(np.arange(len(names)) + (j - 0.5) * w, v, w, color=c, label="behaviour cloning" if m == "BC" else "BC + DAgger")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names); ax.set_ylabel("success (%)"); ax.set_ylim(0, 105); ax.set_title("25% observability", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(figs / "figF_bc_vs_dagger.png", dpi=300); plt.close(fig)


def sanity_checks(S, P):
    """Fail loudly if a table no longer matches the story the paper tells (catches stale CSVs after new seeds)."""
    prim = P[P.comparison.str.startswith("PRIMARY")].iloc[0]
    print("\nPRIMARY (GNN K=2 vs K=1, 25% obs): Δ success = {:.1f} pts, 95% CI [{:.1f}, {:.1f}], Holm p = {}".format(
        prim.dsuccess_pct, prim.dsuccess_ci_lo, prim.dsuccess_ci_hi, fmt_p(prim.p_success_holm)))
    seeds = S[S.policy.str.startswith("GNN")].training_seeds.dropna().unique()
    print("training seeds per GNN policy in the tables:", sorted(int(s) for s in seeds))
    if len(seeds) and max(seeds) < 3:
        print("  NOTE: fewer than 3 training seeds are in the tables. Write Results only after the extra seeds are evaluated.")
    cap = S[(S.obs_pct == 100) & S.policy.isin(["GNN K1", "GNN K2", "GNN K3"])][["policy", "success_pct"]]
    print("capacity check at 100% observability:", "; ".join(f"{r.policy} {r.success_pct:.1f}%" for r in cap.itertuples()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "paper")); a = ap.parse_args()
    out = Path(a.out); figs = out / "figs"; tabs = out / "tables"; figs.mkdir(parents=True, exist_ok=True); tabs.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(RES / "d6_table_success.csv"); P = pd.read_csv(RES / "d6_table_paired.csv"); V = pd.read_csv(RES / "model_validation.csv")
    table1(S, tabs); table2(P, tabs); table_validation(V, tabs)
    fig_main_bars(S, figs); fig_success_vs_K(S, figs); fig_token_coverage(figs); fig_ablation(figs, tabs); fig_contact(S, figs); fig_dagger_vs_bc(figs, tabs)
    sanity_checks(S, P)
    print(f"\nsaved figures to {figs}\nsaved tables  to {tabs}")


if __name__ == "__main__":
    main()
