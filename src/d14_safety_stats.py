"""Day 14: paired statistics for the safety-filter results, on COLLISION-FREE success (success and no contact), seeds 20000-20049.
d12_stats.py drops every row with safety=True, so it cannot read d12_safety_final.csv; this script keeps them and reuses its tests unchanged
(exact McNemar per training seed, significant only if every seed is, Holm over the family, two-level bootstrap CI).
Run: python d14_safety_stats.py        Writes ../results/d14_safety_paired.csv"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6_analysis as A6
import d12_stats as S12

ROOT = S12.ROOT

def load(name, suffix=""):
    d = pd.read_csv(ROOT / "results" / name)
    d = d[(d.loss == 0) & (d.delay == 0) & (d.noise == 0) & (d.change == False) & (d.start_half == 0.5)].copy()
    d["success"] = d.success.astype(bool) & ~d.collision.astype(bool)      # collision-free success
    d["policy"] = d.policy + suffix
    return d

F = load("d12_safety_final.csv", " +filter")
C = load("d12_confirm.csv"); C = C[C.safety == False]
d = pd.concat([F, C], ignore_index=True)

fam = []
for pct in (25, 50):
    for p in ("GNN K1", "GNN K2", "GNN K3", "GNN K2+latched", "relay K2+mem"):
        fam.append((f"filter effect: {p}, with vs without ({pct}% obs)", p + " +filter", p, pct))
    fam += [(f"with filter: K=2 vs K=1 ({pct}% obs)", "GNN K2 +filter", "GNN K1 +filter", pct),
            (f"with filter: K=3 vs K=2 ({pct}% obs)", "GNN K3 +filter", "GNN K2 +filter", pct),
            (f"with filter: latched K=2 vs memoryless K=2 ({pct}% obs)", "GNN K2+latched +filter", "GNN K2 +filter", pct),
            (f"with filter: GNN K=2 vs relay K=2+mem ({pct}% obs)", "GNN K2 +filter", "relay K2+mem +filter", pct),
            (f"with filter: GNN K=3 vs relay K=2+mem ({pct}% obs)", "GNN K3 +filter", "relay K2+mem +filter", pct),
            (f"with filter: latched K=2 vs relay K=2+mem ({pct}% obs)", "GNN K2+latched +filter", "relay K2+mem +filter", pct)]

rows = [r for r in (S12.compare(d, lab, a, b, pct) for lab, a, b, pct in fam) if r]
P = pd.DataFrame(rows)
P["p_success_holm"] = A6.holm(P.p_success_max); P["p_final_err_holm"] = A6.holm(P.p_final_err_max)
P.to_csv(ROOT / "results" / "d14_safety_paired.csv", index=False); print("(writing results/d14_safety_paired.csv)")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(f"=== {len(P)} paired comparisons on collision-free success; McNemar per training seed, significant only if EVERY seed is, Holm over the family ===")
print(P[["comparison", "dsuccess_pct", "ci_lo", "ci_hi", "per_seed_dsuccess_pct", "seeds_a_better", "p_success_max", "p_success_holm"]].round(4).to_string(index=False))
print("\nWording rule: 'significant' only if the Holm p is below 0.05 AND the per-seed differences all point the same way.")
