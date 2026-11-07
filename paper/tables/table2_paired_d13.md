> **SUPERSEDED on 7 Nov 2026** by paper/tables/t14_*.csv, which are built from the Day 12 / Day 14 results (training seeds 0-4, confirmatory seeds 20000-20049). This table uses the Day 6 data. Do not cite it in the paper.

| Comparison | Δ success, points [95 % CI] | Per-seed Δ (seeds 0, 1, 2) | Holm p (success) | Δ final error (mm) | Holm p (final error) |
|---|---|---|---|---|---|
| PRIMARY: K=2 vs K=1 (25% obs) | 25.2 [18.2, 32.7] | 22.0, 31.0, 22.5 | <0.0001 | -22.0 | <0.0001 |
| K=3 vs K=2 (25% obs) | 1.5 [-0.2, 3.5] | 1.0, 2.0, 1.5 | 1.0000 | -1.5 | 1.0000 |
| K=2 vs K=0 (25% obs) | 97.5 [95.7, 99.0] | 97.5, 97.5, 97.5 | <0.0001 | -333.3 | <0.0001 |
| K=2 vs K=1 (50% obs) | 19.2 [13.0, 26.3] | 15.0, 26.0, 16.5 | <0.0001 | -15.4 | 0.0080 |
| K=3 vs K=2 (50% obs) | 1.7 [0.0, 4.2] | 3.5, 0.5, 1.0 | 1.0000 | -0.8 | 1.0000 |
| capacity control: K=3 vs K=1 (100% obs) | 16.3 [10.7, 23.0] | 13.5, 21.5, 14.0 | <0.0001 | -13.6 | 0.0034 |
| capacity control: K=2 vs K=1 (100% obs) | 16.0 [10.3, 22.7] | 13.5, 21.0, 13.5 | <0.0001 | -13.7 | 0.0386 |
| latched K=2 vs memoryless K=2 (25% obs) | 2.5 [1.0, 4.3] | 2.5, 2.5, 2.5 | 0.5000 | -2.4 | 1.0000 |
| GNN vs memoryless relay, K=1 (25% obs) | 33.3 [23.5, 42.5] | 36.5, 27.5, 36.0 | <0.0001 | -61.8 | <0.0001 |
| GNN vs memoryless relay, K=1 (50% obs) | -6.5 [-14.5, 1.5] | -4.0, -12.0, -3.5 | 1.0000 | 1.5 | 1.0000 |
| GNN vs memoryless relay, K=2 (25% obs) | 12.0 [7.0, 17.3] | 12.0, 12.0, 12.0 | 0.0004 | -17.7 | 0.0017 |
| GNN vs memoryless relay, K=2 (50% obs) | 1.2 [-2.0, 4.3] | -0.5, 2.5, 1.5 | 1.0000 | -1.4 | 1.0000 |
| GNN vs memoryless relay, K=3 (25% obs) | 10.0 [5.8, 14.8] | 9.5, 10.5, 10.0 | 0.0014 | -13.5 | 1.0000 |
| GNN vs memoryless relay, K=3 (50% obs) | 1.3 [-0.2, 3.3] | 1.5, 1.5, 1.0 | 1.0000 | -0.8 | 1.0000 |
| latched GNN vs relay with memory, K=2 (25% obs) | 0.0 [0.0, 0.0] | 0.0, 0.0, 0.0 | 1.0000 | 0.0 | 1.0000 |
| latched GNN vs relay with memory, K=2 (50% obs) | 0.5 [0.0, 1.5] | 0.5, 0.5, 0.5 | 1.0000 | -0.4 | 1.0000 |

Exact McNemar test (success) and Wilcoxon signed-rank test (final error), run separately for each training seed; the largest p over seeds is Holm-corrected over the 16 comparisons. Intervals: two-level bootstrap. Source: results/d11_paired.csv (src/d11_stats.py).
