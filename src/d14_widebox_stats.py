"""Day 14 (EXPLORATORY, seeds 30000-30024): wide start box (1.8 m), GNN vs the strong hand-designed relay (relayD = head to the default slot + latest-wins memory).
Same tests as d12_stats.py (exact McNemar per training seed, significant only if every seed is, Holm per metric, two-level bootstrap CI).
Two metrics: success (final error < 25 mm) and collision-free success (success and no contact).
Run: python d14_widebox_stats.py        Writes ../results/d14_widebox_paired.csv"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6_analysis as A6
import d12_stats as S12

ROOT = S12.ROOT
d0 = pd.concat([pd.read_csv(ROOT / "results" / n) for n in ("d12_widebox.csv", "d12_widebox_relayD.csv")], ignore_index=True)
d0 = d0[(d0.loss == 0) & (d0.delay == 0) & (d0.noise == 0) & (d0.change == False) & (d0.start_half == 0.9) & (d0.safety == False)].copy()

fam = []
for pct in (25, 50):
    fam += [(f"GNN K=2 vs relayD K=2, same hops, no memory ({pct}% obs)", "GNN K2", "relayD K2", pct),
            (f"GNN K=2 vs relayD K=2+mem ({pct}% obs)", "GNN K2", "relayD K2+mem", pct),
            (f"GNN K=3 vs relayD K=3+mem ({pct}% obs)", "GNN K3", "relayD K3+mem", pct),
            (f"latched GNN K=2 vs relayD K=2+mem ({pct}% obs)", "GNN K2+latched", "relayD K2+mem", pct),
            (f"latched K=2 vs memoryless K=2 GNN ({pct}% obs)", "GNN K2+latched", "GNN K2", pct),
            (f"relayD K=2+mem vs hold-still relay K=2+mem ({pct}% obs)", "relayD K2+mem", "relay K2+mem", pct)]

out = []
for metric in ("success", "collision-free success"):
    d = d0.copy()
    if metric != "success": d["success"] = d.success.astype(bool) & ~d.collision.astype(bool)
    P = pd.DataFrame([r for r in (S12.compare(d, lab, a, b, pct) for lab, a, b, pct in fam) if r])
    P["p_success_holm"] = A6.holm(P.p_success_max); P.insert(0, "metric", metric); out.append(P)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(f"\n=== WIDE BOX (exploratory), {metric}: {len(P)} paired comparisons, Holm within this metric ===")
    print(P[["comparison", "dsuccess_pct", "ci_lo", "ci_hi", "per_seed_dsuccess_pct", "seeds_a_better", "p_success_max", "p_success_holm"]].round(4).to_string(index=False))
pd.concat(out).to_csv(ROOT / "results" / "d14_widebox_paired.csv", index=False); print("\n(writing results/d14_widebox_paired.csv)")
