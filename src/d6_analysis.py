"""Day 6b: tables, paired statistics and the main figure from RAW episode files. No numbers are typed by hand.
Inputs : ../results/d3_baselines.csv (no-learning relays + oracle, Day 3), ../results/d6_main_eval.csv (GNN models, Day 6a)
Outputs: ../results/d6_table_success.csv, d6_table_paired.csv, ../figures/d6_main_success.png, printed summary.
Statistics fixed in protocol.md BEFORE the experiment: paired by episode (formation, seed, observability); Wilcoxon signed-rank (two-sided);
Holm correction across the whole family below; effect = mean paired difference with 95% bootstrap CI (resampling test episodes).
When a GNN has several training seeds, each test episode's value is first averaged over training seeds (the episode stays the unit of analysis).
Contact = centre distance < 0.074 m (frozen definition). Overlap = centre distance < 0.069 m (>= 5 mm overlap), reported separately.
Run: python d6_analysis.py [--eval-csv path]"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
KEY = ["formation", "seed", "obs_pct"]

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def load(eval_csv):
    b = pd.read_csv(ROOT / "results" / "d3_baselines.csv")
    b["policy"] = b.policy.map(lambda s: "oracle" if s == "oracle" else ("relay K" + s[1:] if "+mem" not in s else "relay K" + s[1:].replace("+mem", "") + "+mem"))
    b["train_seed"] = np.nan; b["method"] = "no-learning"
    g = pd.read_csv(eval_csv); g = g[g.method == "DAgger"].copy()
    g["policy"] = g.apply(lambda r: f"GNN K{r.K}" + ("+latched" if r.latched else ""), axis=1)
    cols = KEY + ["policy", "final_err", "success", "success_05", "collision", "min_clearance", "train_seed", "method"]
    d = pd.concat([b[cols], g[cols]], ignore_index=True)
    d["overlap5"] = d.min_clearance < 0.069
    return d

def table_success(d):
    rows = []
    for (pol, pct), x in d.groupby(["policy", "obs_pct"]):
        seeds = x.train_seed.dropna().unique(); k, n = int(x.success.sum()), len(x)
        per_seed = x.groupby("train_seed").success.mean() if len(seeds) else pd.Series(dtype=float)
        lo, hi = wilson(k, n)
        rows.append(dict(policy=pol, obs_pct=pct, training_seeds=len(seeds) or np.nan, episodes=n, success_pct=100 * k / n, ci95_lo=100 * lo, ci95_hi=100 * hi,
                         sd_over_seeds_pct=100 * per_seed.std() if len(per_seed) > 1 else np.nan, success05_pct=100 * x.success_05.mean(),
                         mean_final_err_m=x.final_err.mean(), contact_pct=100 * x.collision.mean(), overlap5mm_pct=100 * x.overlap5.mean()))
    return pd.DataFrame(rows)

def per_episode(d, pol, pct, col):
    x = d[(d.policy == pol) & (d.obs_pct == pct)]
    return x.groupby(["formation", "seed"])[col].mean() if len(x) else None

def compare(d, a, b, pct, label):
    out = []
    for col in ("success", "final_err"):
        x, y = per_episode(d, a, pct, col), per_episode(d, b, pct, col)
        if x is None or y is None: return None
        j = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna(); diff = (j.a - j.b).astype(float).values
        rng = np.random.default_rng(0); boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
        try: p = stats.wilcoxon(diff, zero_method="wilcox").pvalue if np.any(diff != 0) else 1.0
        except ValueError: p = 1.0
        out.append(dict(mean_diff=diff.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5), p=p, n=len(diff)))
    s, e = out
    return dict(comparison=label, A=a, B=b, obs_pct=pct, n_pairs=s["n"], dsuccess_pct=100 * s["mean_diff"], dsuccess_ci_lo=100 * s["ci_lo"], dsuccess_ci_hi=100 * s["ci_hi"], p_success=s["p"],
                dfinal_err_m=e["mean_diff"], dfinal_err_ci_lo=e["ci_lo"], dfinal_err_ci_hi=e["ci_hi"], p_final_err=e["p"])

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj

def family():
    f = [("PRIMARY: K=2 vs K=1 (25% obs)", "GNN K2", "GNN K1", 25), ("K=3 vs K=2 (25% obs)", "GNN K3", "GNN K2", 25), ("K=2 vs K=0 (25% obs)", "GNN K2", "GNN K0", 25),
         ("K=2 vs K=1 (50% obs)", "GNN K2", "GNN K1", 50), ("K=3 vs K=2 (50% obs)", "GNN K3", "GNN K2", 50),
         ("capacity control: K=3 vs K=1 (100% obs)", "GNN K3", "GNN K1", 100), ("capacity control: K=2 vs K=1 (100% obs)", "GNN K2", "GNN K1", 100),
         ("latched K=2 vs memoryless K=2 (25% obs)", "GNN K2+latched", "GNN K2", 25)]
    for K in (1, 2, 3):
        for pct in (25, 50): f.append((f"GNN vs memoryless relay, K={K} ({pct}% obs)", f"GNN K{K}", f"relay K{K}", pct))
    for pct in (25, 50): f.append((f"latched GNN vs relay with memory, K=2 ({pct}% obs)", "GNN K2+latched", "relay K2+mem", pct))
    return f

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--eval-csv", default=str(ROOT / "results" / "d6_main_eval.csv")); a = ap.parse_args()
    d = load(a.eval_csv); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    T = table_success(d); T.to_csv(ROOT / "results" / "d6_table_success.csv", index=False)
    order = ["oracle", "relay K1", "relay K2", "relay K3", "relay K7", "relay K1+mem", "relay K2+mem", "relay K3+mem", "relay K7+mem", "relay K0",
             "GNN K0", "GNN K1", "GNN K2", "GNN K3", "GNN K2+latched"]
    T["o"] = T.policy.map({p: i for i, p in enumerate(order)}); T = T.sort_values(["obs_pct", "o"], ascending=[False, True]).drop(columns="o")
    print("\n=== SUCCESS (final mean error < 0.025 m) with 95% Wilson CI; contact < 0.074 m, overlap >= 5 mm (< 0.069 m) ===")
    print(T.round(2).to_string(index=False))
    rows = [r for r in (compare(d, A, B, pct, lab) for lab, A, B, pct in family()) if r]
    if rows:
        P = pd.DataFrame(rows); P["p_success_holm"] = holm(P.p_success); P["p_final_err_holm"] = holm(P.p_final_err); P.to_csv(ROOT / "results" / "d6_table_paired.csv", index=False)
        print(f"\n=== PAIRED COMPARISONS ({len(P)} in the pre-registered family; Holm-adjusted p; difference = A - B) ===")
        print(P[["comparison", "n_pairs", "dsuccess_pct", "dsuccess_ci_lo", "dsuccess_ci_hi", "p_success", "p_success_holm", "dfinal_err_m", "p_final_err_holm"]].round(4).to_string(index=False))
        print("\nRule: write 'significant' only where the HOLM p is below 0.05. Anything else is 'no evidence of a difference' (and say how wide the CI is).")
    g = T[(T.policy.str.startswith("GNN")) & (T.obs_pct == 100) & (~T.policy.str.contains("latched"))]
    if len(g): print("\nCAPACITY CHECK at 100% observability (every robot sees the gesture, so K should not matter):", "; ".join(f"{r.policy} {r.success_pct:.1f}%" for r in g.itertuples()))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pols = [p for p in ["oracle", "relay K1", "relay K2", "relay K3", "GNN K0", "GNN K1", "GNN K2", "GNN K3", "GNN K2+latched"] if p in set(T.policy)]
    col = {"oracle": "#222222", "relay K1": "#bdbdbd", "relay K2": "#969696", "relay K3": "#636363", "GNN K0": "#c6dbef", "GNN K1": "#6baed6", "GNN K2": "#2171b5", "GNN K3": "#08306b", "GNN K2+latched": "#e6550d"}
    fig, ax = plt.subplots(figsize=(10, 4)); w = 0.8 / len(pols)
    for j, pol in enumerate(pols):
        for i, pct in enumerate((100, 50, 25)):
            r = T[(T.policy == pol) & (T.obs_pct == pct)]
            if len(r): r = r.iloc[0]; ax.bar(i + j * w, r.success_pct, w, color=col[pol], label=pol if i == 0 else None, yerr=[[r.success_pct - r.ci95_lo], [r.ci95_hi - r.success_pct]], capsize=2, error_kw=dict(lw=0.8))
    ax.set_xticks([i + 0.4 - w / 2 for i in range(3)]); ax.set_xticklabels(["100%", "50%", "25%"]); ax.set_xlabel("observability"); ax.set_ylabel("success (%)"); ax.set_ylim(0, 105)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="lower left"); fig.tight_layout(); fig.savefig(ROOT / "figures" / "d6_main_success.png", dpi=300)
    print("\nsaved results/d6_table_success.csv, results/d6_table_paired.csv, figures/d6_main_success.png")

if __name__ == "__main__":
    main()
