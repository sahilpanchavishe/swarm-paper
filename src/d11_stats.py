"""Day 11a: corrected confirmatory statistics. Reads the same raw CSVs as d6_analysis.py (nothing typed by hand).
Fixes three problems found in d6_analysis.py:
  1. Success is binary, so the paired test is EXACT McNemar (as the protocol says), not Wilcoxon. Wilcoxon is kept only for final error.
  2. Training seeds are NOT averaged away. Every test is run separately for each training seed (A's seed s against B's seed s, or against
     the single relay run). A comparison is called significant only if it is significant in EVERY seed (intersection-union rule:
     p = max over seeds), then Holm across the whole family.
  3. The confidence interval resamples BOTH levels: test scenarios (formation, seed) and training seeds (two-level bootstrap).
     With only 3 training seeds the seed level is coarse, so per-seed differences are printed next to it; do not hide them.
Run: python d11_stats.py        Writes ../results/d11_paired.csv"""
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

def two_level_boot(Sa, Sb, rng):
    n = len(Sa); out = np.empty(NBOOT)
    for r in range(NBOOT):
        i = rng.integers(0, n, n)
        ja = rng.integers(0, Sa.shape[1], Sa.shape[1]); jb = rng.integers(0, Sb.shape[1], Sb.shape[1])
        out[r] = Sa[i][:, ja].mean() - Sb[i][:, jb].mean()
    return np.percentile(out, [2.5, 97.5])

def compare(d, label, a, b, pct):
    Sa, Sb = wide(d, a, pct, "success"), wide(d, b, pct, "success")
    Ea, Eb = wide(d, a, pct, "final_err"), wide(d, b, pct, "final_err")
    if Sa is None or Sb is None: return None
    idx = Sa.index.intersection(Sb.index); Sa, Sb, Ea, Eb = Sa.loc[idx], Sb.loc[idx], Ea.loc[idx], Eb.loc[idx]
    prs = pair_columns(Sa, Sb)
    p_s = [mcnemar_exact(Sa[ca].values, Sb[cb].values) for ca, cb in prs]
    p_e = [wilcoxon_p(Ea[ca].values, Eb[cb].values) for ca, cb in prs]
    per_seed = [100 * (Sa[ca].mean() - Sb[cb].mean()) for ca, cb in prs]
    lo, hi = two_level_boot(Sa.values.astype(float), Sb.values.astype(float), np.random.default_rng(0))
    return dict(comparison=label, A=a, B=b, obs_pct=pct, n_scenarios=len(idx), n_seed_pairs=len(prs),
                dsuccess_pct=100 * (Sa.values.mean() - Sb.values.mean()), ci_lo=100 * lo, ci_hi=100 * hi,
                per_seed_dsuccess_pct=";".join(f"{x:.1f}" for x in per_seed), seeds_a_better=int(sum(x > 0 for x in per_seed)),
                p_success_max=max(p_s), p_final_err_max=max(p_e), dfinal_err_m=float(np.mean(Ea.values) - np.mean(Eb.values)))

def main():
    d = A6.load(str(ROOT / "results" / "d6_main_eval.csv"))
    rows = [r for r in (compare(d, lab, a, b, pct) for lab, a, b, pct in A6.family()) if r]
    P = pd.DataFrame(rows)
    P["p_success_holm"] = A6.holm(P.p_success_max); P["p_final_err_holm"] = A6.holm(P.p_final_err_max)
    P.to_csv(ROOT / "results" / "d11_paired.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(f"=== {len(P)} paired comparisons; McNemar per training seed, significant only if EVERY seed is (max p), Holm over the family ===")
    print(P[["comparison", "dsuccess_pct", "ci_lo", "ci_hi", "per_seed_dsuccess_pct", "seeds_a_better", "p_success_max", "p_success_holm", "p_final_err_holm"]].round(4).to_string(index=False))
    print("\nCI = two-level bootstrap (scenarios and training seeds). Wording rule: 'significant' only if the Holm p is below 0.05 AND the per-seed differences all point the same way.")

if __name__ == "__main__":
    main()
