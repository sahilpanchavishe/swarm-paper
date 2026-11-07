"""Day 12 (extends Day 11a): confirmatory statistics. Reads the same raw CSVs as d6_analysis.py (nothing typed by hand).
Fixes three problems found in d6_analysis.py:
  1. Success is binary, so the paired test is EXACT McNemar (as the protocol says), not Wilcoxon. Wilcoxon is kept only for final error.
  2. Training seeds are NOT averaged away. Every test is run separately for each training seed (A's seed s against B's seed s, or against
     the single relay run). A comparison is called significant only if it is significant in EVERY seed (intersection-union rule:
     p = max over seeds), then Holm across the whole family.
  3. The confidence interval resamples BOTH levels: test scenarios (formation, seed) and training seeds (two-level bootstrap).
     With only 3 training seeds the seed level is coarse, so per-seed differences are printed next to it; do not hide them.
Run: python d12_stats.py        Writes ../results/d12_paired.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6_analysis as A6

ROOT = A6.ROOT
NBOOT = 5000

def wide(d, pol, pct, col):
    """(scenario x training-seed) matrix; relay/oracle rows have a single column."""
    x = d[(d.policy == pol) & (d.obs_pct == pct)].copy()
    if x.empty: return None
    x["ts"] = x.train_seed.fillna(-1)
    return x.pivot_table(index=["formation", "seed"], columns="ts", values=col, aggfunc="mean")

def mcnemar_exact(a, b):
    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
    return 1.0 if n01 + n10 == 0 else stats.binomtest(n01, n01 + n10, 0.5).pvalue

def wilcoxon_p(a, b):
    diff = a - b
    if not np.any(diff != 0): return 1.0
    try: return stats.wilcoxon(diff, zero_method="wilcox").pvalue
    except ValueError: return 1.0

def pair_columns(Wa, Wb):
    """Pairs of columns to compare: seed s with seed s; a single-column side (relay) is paired with every seed of the other side."""
    ca, cb = list(Wa.columns), list(Wb.columns)
    if len(ca) == 1 and len(cb) == 1: return [(ca[0], cb[0])]
    if len(cb) == 1: return [(c, cb[0]) for c in ca]
    if len(ca) == 1: return [(ca[0], c) for c in cb]
    common = sorted(set(ca) & set(cb)); return [(c, c) for c in common]

def two_level_boot(Sa, Sb, rng, joint=False):
    """Two-level bootstrap of the mean difference. Level 1: test scenarios (shared by both models).
    Level 2: training seeds. joint=True -> both models use the SAME resampled seed indices (columns must be the same seeds,
    same order). joint=False -> each side resamples its own seeds (a single-column side, e.g. a relay, is unaffected)."""
    n = len(Sa); out = np.empty(NBOOT)
    if joint: assert Sa.shape[1] == Sb.shape[1]
    for r in range(NBOOT):
        i = rng.integers(0, n, n)
        ja = rng.integers(0, Sa.shape[1], Sa.shape[1])
        jb = ja if joint else rng.integers(0, Sb.shape[1], Sb.shape[1])
        out[r] = Sa[i][:, ja].mean() - Sb[i][:, jb].mean()
    return np.percentile(out, [2.5, 97.5])

def compare(d, label, a, b, pct):
    Sa, Sb = wide(d, a, pct, "success"), wide(d, b, pct, "success")
    Ea, Eb = wide(d, a, pct, "final_err"), wide(d, b, pct, "final_err")
    if Sa is None or Sb is None: return None
    idx = Sa.index.intersection(Sb.index); Sa, Sb, Ea, Eb = Sa.loc[idx], Sb.loc[idx], Ea.loc[idx], Eb.loc[idx]
    if Sa.shape[1] > 1 and Sb.shape[1] > 1:      # Day 14 fix: pooled difference and CI use ONLY the training seeds both sides have
        common = sorted(set(Sa.columns) & set(Sb.columns)); Sa, Sb, Ea, Eb = Sa[common], Sb[common], Ea[common], Eb[common]
    prs = pair_columns(Sa, Sb)
    p_s = [mcnemar_exact(Sa[ca].values, Sb[cb].values) for ca, cb in prs]
    p_e = [wilcoxon_p(Ea[ca].values, Eb[cb].values) for ca, cb in prs]
    per_seed = [100 * (Sa[ca].mean() - Sb[cb].mean()) for ca, cb in prs]
    joint = Sa.shape[1] > 1 and Sb.shape[1] > 1
    if joint: assert list(Sa.columns) == list(Sb.columns)
    A_, B_ = Sa.values.astype(float), Sb.values.astype(float)
    lo, hi = two_level_boot(A_, B_, np.random.default_rng(0), joint=joint)
    lo_i, hi_i = two_level_boot(A_, B_, np.random.default_rng(0), joint=False)   # sensitivity: independent seed resampling
    return dict(comparison=label, A=a, B=b, obs_pct=pct, n_scenarios=len(idx), n_seed_pairs=len(prs),
                dsuccess_pct=100 * (Sa.values.mean() - Sb.values.mean()), ci_lo=100 * lo, ci_hi=100 * hi, ci_lo_indep=100 * lo_i, ci_hi_indep=100 * hi_i,
                per_seed_dsuccess_pct=";".join(f"{x:.1f}" for x in per_seed), seeds_a_better=int(sum(x > 0 for x in per_seed)),
                p_success_max=max(p_s), p_final_err_max=max(p_e), dfinal_err_m=float(np.mean(Ea.values) - np.mean(Eb.values)))

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", nargs="+", help="d11 result CSV(s) (e.g. results/d12_confirm.csv results/d12_capacity.csv); default = the Day 3 + Day 6 files")
    ap.add_argument("--extra", action="store_true", help="also compare the capacity controls (needs d12_capacity.csv) against the main models")
    a = ap.parse_args(); out_name = "d12_paired.csv"
    if a.csv:
        d = pd.concat([pd.read_csv(c) for c in a.csv], ignore_index=True)
        for c, v in dict(loss=0, delay=0, noise=0, safety=False, change=False, start_half=0.5).items():
            if c in d: d = d[d[c] == v]                       # unperturbed episodes only
        out_name = "d12_paired_" + "_".join(Path(c).stem.replace("d12_", "") for c in a.csv) + ".csv"
    else:
        d = A6.load(str(ROOT / "results" / "d6_main_eval.csv"))
    fam = list(A6.family())
    if a.extra:
        for pct in (100, 25):
            for lab, A_, B_ in (("capacity: K=1 wide vs K=1", "GNN K1-wide", "GNN K1"), ("capacity: K=2 vs K=1 wide", "GNN K2", "GNN K1-wide"),
                                ("capacity: K=2 shared vs K=1", "GNN K2-shared", "GNN K1"), ("capacity: K=2 vs K=2 shared", "GNN K2", "GNN K2-shared"),
                                ("capacity: K=0 wide vs K=0", "GNN K0-wide", "GNN K0"), ("capacity: K=1 vs K=0 wide", "GNN K1", "GNN K0-wide")):
                fam.append((f"{lab} ({pct}% obs)", A_, B_, pct))
    rows = [r for r in (compare(d, lab, a_, b_, pct) for lab, a_, b_, pct in fam) if r]
    P = pd.DataFrame(rows)
    P["p_success_holm"] = A6.holm(P.p_success_max); P["p_final_err_holm"] = A6.holm(P.p_final_err_max)
    P.to_csv(ROOT / "results" / out_name, index=False); print(f"(writing results/{out_name})")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(f"=== {len(P)} paired comparisons; McNemar per training seed, significant only if EVERY seed is (max p), Holm over the family ===")
    print(P[["comparison", "dsuccess_pct", "ci_lo", "ci_hi", "per_seed_dsuccess_pct", "seeds_a_better", "p_success_max", "p_success_holm", "p_final_err_holm"]].round(4).to_string(index=False))
    print("\nCI = two-level bootstrap (scenarios and training seeds). Wording rule: 'significant' only if the Holm p is below 0.05 AND the per-seed differences all point the same way.")

if __name__ == "__main__":
    main()
