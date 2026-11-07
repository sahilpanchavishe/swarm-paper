"""Day 14 (EXPLORATORY, seeds 30000-30024): wide start box with the pre-selected safety filter (dsafe 0.10, gamma 20).
Same tests as d12_stats.py. Contact is 0 everywhere, so success == collision-free success.
Run: python d14_widebox_safe_stats.py     Writes ../results/d14_widebox_safe_paired.csv"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6_analysis as A6
import d12_stats as S12

ROOT = S12.ROOT
def load(n):
    d = pd.read_csv(ROOT / "results" / n)
    return d[(d.loss == 0) & (d.delay == 0) & (d.noise == 0) & (d.change == False) & (d.start_half == 0.9)].copy()
F = load("d12_widebox_safe.csv"); F = F[F.safety == True].copy(); F["policy"] = F.policy + " +filter"
U = load("d12_widebox.csv"); U = U[U.safety == False]
d = pd.concat([F, U], ignore_index=True)
for df in (d,): df["success"] = df.success.astype(bool) & ~df.collision.astype(bool)   # collision-free success

fam = []
for pct in (25, 50):
    for p in ("GNN K2", "GNN K3", "GNN K2+latched"):
        fam.append((f"filter effect: {p}, with vs without ({pct}% obs)", p + " +filter", p, pct))
    fam += [(f"filtered GNN K=2 vs relayD K=2+mem ({pct}% obs)", "GNN K2 +filter", "relayD K2+mem +filter", pct),
            (f"filtered GNN K=3 vs relayD K=2+mem ({pct}% obs)", "GNN K3 +filter", "relayD K2+mem +filter", pct),
            (f"filtered latched K=2 vs relayD K=2+mem ({pct}% obs)", "GNN K2+latched +filter", "relayD K2+mem +filter", pct),
            (f"filtered latched K=2 vs memoryless K=2 ({pct}% obs)", "GNN K2+latched +filter", "GNN K2 +filter", pct)]
P = pd.DataFrame([r for r in (S12.compare(d, lab, a, b, pct) for lab, a, b, pct in fam) if r])
P["p_success_holm"] = A6.holm(P.p_success_max)
P.to_csv(ROOT / "results" / "d14_widebox_safe_paired.csv", index=False); print("(writing results/d14_widebox_safe_paired.csv)")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(P[["comparison", "dsuccess_pct", "ci_lo", "ci_hi", "per_seed_dsuccess_pct", "seeds_a_better", "p_success_max", "p_success_holm"]].round(4).to_string(index=False))
