**Table 7. Paired comparisons (exact McNemar per training seed, max p over seeds, Holm over the family). Call a difference significant only if Holm p < 0.05 AND every training seed points the same way (seeds_a_better counts seeds where the first model wins).**

| family | comparison | dsuccess_pct | ci_lo | ci_hi | per_seed_dsuccess_pct | seeds_a_better | p_success_max | p_success_holm |
|---|---|---|---|---|---|---|---|---|
| confirmatory, success | PRIMARY: K=2 vs K=1 (25% obs) | 31.6 | 24.7 | 39.1 | 27.5;36.0;21.0;42.0;31.5 | 5 | 0.0 | 0.0 |
| confirmatory, success | K=3 vs K=2 (25% obs) | 0.0 | -2.7 | 2.3 | 1.0;1.5;2.5;0.5;-5.5 | 4 | 1.0 | 1.0 |
| confirmatory, success | K=2 vs K=0 (25% obs) | 97.5 | 95.5 | 99.17 | 97.5;98.5;96.5 | 3 | 0.0 | 0.0 |
| confirmatory, success | K=2 vs K=1 (50% obs) | 27.0 | 21.1 | 33.1 | 19.5;31.0;21.0;33.0;30.5 | 5 | 0.0 | 0.0 |
| confirmatory, success | K=3 vs K=2 (50% obs) | 0.5 | -1.9 | 2.4 | 3.0;0.5;2.0;1.0;-4.0 | 4 | 1.0 | 1.0 |
| confirmatory, success | capacity control: K=3 vs K=1 (100% obs) | 19.0 | 12.0 | 26.5 | 11.5;25.0;11.0;28.0;19.5 | 5 | 0.0 | 0.0 |
| confirmatory, success | capacity control: K=2 vs K=1 (100% obs) | 18.7 | 11.6 | 26.2 | 10.5;24.5;8.5;27.5;22.5 | 5 | 0.0 | 0.02 |
| confirmatory, success | latched K=2 vs memoryless K=2 (25% obs) | 2.5 | 0.83 | 4.5 | 2.5;1.5;3.5 | 3 | 0.25 | 1.0 |
| confirmatory, success | GNN vs memoryless relay, K=1 (25% obs) | 31.3 | 21.1 | 40.9 | 34.5;27.0;40.0;22.0;33.0 | 5 | 0.0 | 0.0 |
| confirmatory, success | GNN vs memoryless relay, K=1 (50% obs) | -18.4 | -25.7 | -10.9 | -12.5;-21.5;-13.0;-24.0;-21.0 | 0 | 0.0 | 0.02 |
| confirmatory, success | GNN vs memoryless relay, K=2 (25% obs) | 11.9 | 7.0 | 17.1 | 11.0;12.0;10.0;13.0;13.5 | 5 | 0.0 | 0.01 |
| confirmatory, success | GNN vs memoryless relay, K=2 (50% obs) | 0.6 | -1.7 | 3.0 | -1.0;1.5;0.0;1.0;1.5 | 3 | 1.0 | 1.0 |
| confirmatory, success | GNN vs memoryless relay, K=3 (25% obs) | 8.9 | 4.1 | 13.9 | 9.0;10.5;9.5;10.5;5.0 | 5 | 0.11 | 1.0 |
| confirmatory, success | GNN vs memoryless relay, K=3 (50% obs) | 0.6 | -2.0 | 3.0 | 1.5;1.5;1.5;1.5;-3.0 | 4 | 0.25 | 1.0 |
| confirmatory, success | latched GNN vs relay with memory, K=2 (25% obs) | 0.0 | 0.0 | 0.0 | 0.0;0.0;0.0 | 0 | 1.0 | 1.0 |
| confirmatory, success | latched GNN vs relay with memory, K=2 (50% obs) | 0.5 | 0.0 | 1.5 | 0.5;0.5;0.5 | 3 | 1.0 | 1.0 |
| confirmatory, success | capacity: K=1 wide vs K=1 (100% obs) | 3.83 | -5.5 | 14.67 | -2.5;10.0;4.0 | 2 | 0.51 | 1.0 |
| confirmatory, success | capacity: K=2 vs K=1 wide (100% obs) | 10.67 | 5.17 | 16.33 | 13.0;14.5;4.5 | 3 | 0.06 | 0.95 |
| confirmatory, success | capacity: K=2 shared vs K=1 (100% obs) | 15.33 | 8.33 | 24.83 | 11.5;24.5;10.0 | 3 | 0.0 | 0.0 |
| confirmatory, success | capacity: K=2 vs K=2 shared (100% obs) | -0.83 | -2.83 | 1.0 | -1.0;0.0;-1.5 | 0 | 1.0 | 1.0 |
| confirmatory, success | capacity: K=0 wide vs K=0 (100% obs) | -6.33 | -11.67 | -1.17 | -3.5;-11.0;-4.5 | 0 | 0.26 | 1.0 |
| confirmatory, success | capacity: K=1 vs K=0 wide (100% obs) | 77.17 | 67.67 | 85.0 | 81.5;69.5;80.5 | 3 | 0.0 | 0.0 |
| confirmatory, success | capacity: K=1 wide vs K=1 (25% obs) | 3.67 | -5.17 | 12.5 | 2.0;10.5;-1.5 | 2 | 0.8 | 1.0 |
| confirmatory, success | capacity: K=2 vs K=1 wide (25% obs) | 24.5 | 19.33 | 29.83 | 25.5;25.5;22.5 | 3 | 0.0 | 0.0 |
| confirmatory, success | capacity: K=2 shared vs K=1 (25% obs) | 28.17 | 20.0 | 36.67 | 29.5;32.5;22.5 | 3 | 0.0 | 0.0 |
| confirmatory, success | capacity: K=2 vs K=2 shared (25% obs) | 0.0 | -3.0 | 3.5 | -2.0;3.5;-1.5 | 1 | 0.55 | 1.0 |
| confirmatory, success | capacity: K=0 wide vs K=0 (25% obs) | 0.0 | 0.0 | 0.0 | 0.0;0.0;0.0 | 0 | 1.0 | 1.0 |
| confirmatory, success | capacity: K=1 vs K=0 wide (25% obs) | 69.33 | 61.5 | 76.83 | 70.0;62.5;75.5 | 3 | 0.0 | 0.0 |
| confirmatory, collision-free success | filter effect: GNN K1, with vs without (25% obs) | 72.9 | 66.6 | 79.2 | 72.0;69.5;75.0;70.0;78.0 | 5 | 0.0 | 0.0 |
| confirmatory, collision-free success | filter effect: GNN K2, with vs without (25% obs) | 55.9 | 47.2 | 64.4 | 54.5;49.5;62.5;67.0;46.0 | 5 | 0.0 | 0.0 |
| confirmatory, collision-free success | filter effect: GNN K3, with vs without (25% obs) | 29.8 | 12.1 | 58.0 | 16.5;22.0;10.0;17.0;83.5 | 5 | 0.0 | 0.0 |
| confirmatory, collision-free success | filter effect: GNN K2+latched, with vs without (25% obs) | 3.67 | 1.5 | 6.0 | 2.5;4.0;4.5 | 3 | 0.06 | 0.88 |
| confirmatory, collision-free success | filter effect: relay K2+mem, with vs without (25% obs) | 0.5 | 0.0 | 1.5 | 0.5 | 1 | 1.0 | 1.0 |
| confirmatory, collision-free success | with filter: K=2 vs K=1 (25% obs) | 18.7 | 13.6 | 24.3 | 17.5;22.0;12.0;24.5;17.5 | 5 | 0.0 | 0.0 |
| confirmatory, collision-free success | with filter: K=3 vs K=2 (25% obs) | 0.7 | -0.9 | 2.1 | 0.5;1.5;1.5;1.0;-1.0 | 4 | 1.0 | 1.0 |
| confirmatory, collision-free success | with filter: latched K=2 vs memoryless K=2 (25% obs) | 1.67 | 0.5 | 3.33 | 1.5;1.5;2.0 | 3 | 0.25 | 1.0 |
| confirmatory, collision-free success | with filter: GNN K=2 vs relay K=2+mem (25% obs) | -1.5 | -2.6 | -0.6 | -1.5;-1.5;-2.0;-1.0;-1.5 | 0 | 0.5 | 1.0 |
| confirmatory, collision-free success | with filter: GNN K=3 vs relay K=2+mem (25% obs) | -0.8 | -2.2 | 0.0 | -1.0;0.0;-0.5;0.0;-2.5 | 0 | 1.0 | 1.0 |
| confirmatory, collision-free success | with filter: latched K=2 vs relay K=2+mem (25% obs) | 0.0 | 0.0 | 0.0 | 0.0;0.0;0.0 | 0 | 1.0 | 1.0 |
| confirmatory, collision-free success | filter effect: GNN K1, with vs without (50% obs) | 77.9 | 72.0 | 83.3 | 76.5;77.0;80.5;77.5;78.0 | 5 | 0.0 | 0.0 |
| confirmatory, collision-free success | filter effect: GNN K2, with vs without (50% obs) | 58.8 | 49.8 | 68.1 | 56.0;48.0;68.0;70.0;52.0 | 5 | 0.0 | 0.0 |
| confirmatory, collision-free success | filter effect: GNN K3, with vs without (50% obs) | 30.3 | 12.9 | 59.2 | 13.0;20.0;13.5;18.0;87.0 | 5 | 0.0 | 0.0 |
| confirmatory, collision-free success | filter effect: GNN K2+latched, with vs without (50% obs) | 3.17 | 1.33 | 5.33 | 2.5;3.0;4.0 | 3 | 0.06 | 0.88 |
| confirmatory, collision-free success | filter effect: relay K2+mem, with vs without (50% obs) | 0.5 | 0.0 | 1.5 | 0.5 | 1 | 1.0 | 1.0 |
| confirmatory, collision-free success | with filter: K=2 vs K=1 (50% obs) | 13.2 | 8.9 | 18.1 | 10.0;19.5;9.0;15.5;12.0 | 5 | 0.0 | 0.0 |
| confirmatory, collision-free success | with filter: K=3 vs K=2 (50% obs) | 0.4 | -0.3 | 1.4 | 1.5;0.0;0.5;0.0;0.0 | 2 | 1.0 | 1.0 |
| confirmatory, collision-free success | with filter: latched K=2 vs memoryless K=2 (50% obs) | 0.67 | 0.0 | 2.0 | 1.5;0.0;0.5 | 2 | 1.0 | 1.0 |
| confirmatory, collision-free success | with filter: GNN K=2 vs relay K=2+mem (50% obs) | 0.0 | -1.2 | 1.4 | -1.0;0.5;0.0;0.5;0.0 | 2 | 1.0 | 1.0 |
| confirmatory, collision-free success | with filter: GNN K=3 vs relay K=2+mem (50% obs) | 0.4 | -0.4 | 1.5 | 0.5;0.5;0.5;0.5;0.0 | 4 | 1.0 | 1.0 |
| confirmatory, collision-free success | with filter: latched K=2 vs relay K=2+mem (50% obs) | 0.5 | 0.0 | 1.5 | 0.5;0.5;0.5 | 3 | 1.0 | 1.0 |
| EXPLORATORY wide box, unfiltered / success | GNN K=2 vs relayD K=2, same hops, no memory (25% obs) | 17.33 | 7.33 | 27.0 | 12.0;21.0;19.0 | 3 | 0.03 | 0.29 |
| EXPLORATORY wide box, unfiltered / success | GNN K=2 vs relayD K=2+mem (25% obs) | -6.67 | -13.33 | -1.67 | -12.0;-3.0;-5.0 | 0 | 0.25 | 1.0 |
| EXPLORATORY wide box, unfiltered / success | GNN K=3 vs relayD K=3+mem (25% obs) | -4.67 | -8.33 | -1.67 | -4.0;-5.0;-5.0 | 0 | 0.12 | 1.0 |
| EXPLORATORY wide box, unfiltered / success | latched GNN K=2 vs relayD K=2+mem (25% obs) | -0.33 | -2.0 | 0.0 | 0.0;0.0;-1.0 | 0 | 1.0 | 1.0 |
| EXPLORATORY wide box, unfiltered / success | latched K=2 vs memoryless K=2 GNN (25% obs) | 6.33 | 1.0 | 13.33 | 12.0;3.0;4.0 | 3 | 0.25 | 1.0 |
| EXPLORATORY wide box, unfiltered / success | relayD K=2+mem vs hold-still relay K=2+mem (25% obs) | 69.0 | 60.0 | 78.0 | 69.0 | 1 | 0.0 | 0.0 |
| EXPLORATORY wide box, unfiltered / success | GNN K=2 vs relayD K=2, same hops, no memory (50% obs) | -3.33 | -6.67 | -0.67 | -5.0;-2.0;-3.0 | 0 | 0.5 | 1.0 |
| EXPLORATORY wide box, unfiltered / success | GNN K=2 vs relayD K=2+mem (50% obs) | -3.33 | -6.67 | -0.67 | -5.0;-2.0;-3.0 | 0 | 0.5 | 1.0 |
| EXPLORATORY wide box, unfiltered / success | GNN K=3 vs relayD K=3+mem (50% obs) | -0.67 | -3.0 | 0.0 | -1.0;-1.0;0.0 | 0 | 1.0 | 1.0 |
| EXPLORATORY wide box, unfiltered / success | latched GNN K=2 vs relayD K=2+mem (50% obs) | -0.33 | -2.0 | 0.0 | -1.0;0.0;0.0 | 0 | 1.0 | 1.0 |
| EXPLORATORY wide box, unfiltered / success | latched K=2 vs memoryless K=2 GNN (50% obs) | 3.0 | 0.0 | 6.67 | 4.0;2.0;3.0 | 3 | 0.5 | 1.0 |
| EXPLORATORY wide box, unfiltered / success | relayD K=2+mem vs hold-still relay K=2+mem (50% obs) | 48.0 | 38.0 | 58.0 | 48.0 | 1 | 0.0 | 0.0 |
| EXPLORATORY wide box, unfiltered / collision-free success | GNN K=2 vs relayD K=2, same hops, no memory (25% obs) | -21.67 | -35.0 | -8.33 | -24.0;-14.0;-27.0 | 0 | 0.06 | 0.19 |
| EXPLORATORY wide box, unfiltered / collision-free success | GNN K=2 vs relayD K=2+mem (25% obs) | -45.67 | -56.0 | -35.67 | -48.0;-38.0;-51.0 | 0 | 0.0 | 0.0 |
| EXPLORATORY wide box, unfiltered / collision-free success | GNN K=3 vs relayD K=3+mem (25% obs) | -17.33 | -25.67 | -10.33 | -23.0;-16.0;-13.0 | 0 | 0.0 | 0.0 |
| EXPLORATORY wide box, unfiltered / collision-free success | latched GNN K=2 vs relayD K=2+mem (25% obs) | -5.67 | -11.68 | -1.0 | -1.0;-5.0;-11.0 | 0 | 1.0 | 1.0 |
| EXPLORATORY wide box, unfiltered / collision-free success | latched K=2 vs memoryless K=2 GNN (25% obs) | 40.0 | 28.67 | 51.33 | 47.0;33.0;40.0 | 3 | 0.0 | 0.0 |
| EXPLORATORY wide box, unfiltered / collision-free success | relayD K=2+mem vs hold-still relay K=2+mem (25% obs) | 69.0 | 60.0 | 78.0 | 69.0 | 1 | 0.0 | 0.0 |
| EXPLORATORY wide box, unfiltered / collision-free success | GNN K=2 vs relayD K=2, same hops, no memory (50% obs) | -56.33 | -68.0 | -44.0 | -56.0;-47.0;-66.0 | 0 | 0.0 | 0.0 |
| EXPLORATORY wide box, unfiltered / collision-free success | GNN K=2 vs relayD K=2+mem (50% obs) | -56.33 | -68.0 | -44.0 | -56.0;-47.0;-66.0 | 0 | 0.0 | 0.0 |
| EXPLORATORY wide box, unfiltered / collision-free success | GNN K=3 vs relayD K=3+mem (50% obs) | -12.67 | -18.33 | -7.67 | -15.0;-11.0;-12.0 | 0 | 0.0 | 0.0 |
| EXPLORATORY wide box, unfiltered / collision-free success | latched GNN K=2 vs relayD K=2+mem (50% obs) | -3.0 | -6.0 | -0.67 | -2.0;-4.0;-3.0 | 0 | 0.5 | 1.0 |
| EXPLORATORY wide box, unfiltered / collision-free success | latched K=2 vs memoryless K=2 GNN (50% obs) | 53.33 | 41.33 | 65.0 | 54.0;43.0;63.0 | 3 | 0.0 | 0.0 |
| EXPLORATORY wide box, unfiltered / collision-free success | relayD K=2+mem vs hold-still relay K=2+mem (50% obs) | 48.0 | 38.0 | 58.0 | 48.0 | 1 | 0.0 | 0.0 |
| EXPLORATORY wide box, filtered | filter effect: GNN K2, with vs without (25% obs) | 38.67 | 28.33 | 49.33 | 38.0;33.0;45.0 | 3 | 0.0 | 0.0 |
| EXPLORATORY wide box, filtered | filter effect: GNN K3, with vs without (25% obs) | 13.0 | 6.33 | 21.0 | 19.0;11.0;9.0 | 3 | 0.0 | 0.04 |
| EXPLORATORY wide box, filtered | filter effect: GNN K2+latched, with vs without (25% obs) | 5.33 | 0.33 | 11.67 | 1.0;4.0;11.0 | 3 | 1.0 | 1.0 |
| EXPLORATORY wide box, filtered | filtered GNN K=2 vs relayD K=2+mem (25% obs) | -7.0 | -12.33 | -2.67 | -10.0;-5.0;-6.0 | 0 | 0.06 | 0.62 |
| EXPLORATORY wide box, filtered | filtered GNN K=3 vs relayD K=2+mem (25% obs) | -4.33 | -8.0 | -1.33 | -4.0;-5.0;-4.0 | 0 | 0.12 | 1.0 |
| EXPLORATORY wide box, filtered | filtered latched K=2 vs relayD K=2+mem (25% obs) | -0.33 | -2.0 | 0.0 | 0.0;-1.0;0.0 | 0 | 1.0 | 1.0 |
| EXPLORATORY wide box, filtered | filtered latched K=2 vs memoryless K=2 (25% obs) | 6.67 | 2.33 | 12.0 | 10.0;4.0;6.0 | 3 | 0.12 | 1.0 |
| EXPLORATORY wide box, filtered | filter effect: GNN K2, with vs without (50% obs) | 53.67 | 41.0 | 65.67 | 51.0;47.0;63.0 | 3 | 0.0 | 0.0 |
| EXPLORATORY wide box, filtered | filter effect: GNN K3, with vs without (50% obs) | 12.0 | 7.0 | 17.67 | 14.0;10.0;12.0 | 3 | 0.0 | 0.02 |
| EXPLORATORY wide box, filtered | filter effect: GNN K2+latched, with vs without (50% obs) | 3.0 | 0.67 | 6.0 | 2.0;4.0;3.0 | 3 | 0.5 | 1.0 |
| EXPLORATORY wide box, filtered | filtered GNN K=2 vs relayD K=2+mem (50% obs) | -2.67 | -6.33 | 0.0 | -5.0;0.0;-3.0 | 0 | 1.0 | 1.0 |
| EXPLORATORY wide box, filtered | filtered GNN K=3 vs relayD K=2+mem (50% obs) | -0.67 | -3.0 | 0.0 | -1.0;-1.0;0.0 | 0 | 1.0 | 1.0 |
| EXPLORATORY wide box, filtered | filtered latched K=2 vs relayD K=2+mem (50% obs) | 0.0 | 0.0 | 0.0 | 0.0;0.0;0.0 | 0 | 1.0 | 1.0 |
| EXPLORATORY wide box, filtered | filtered latched K=2 vs memoryless K=2 (50% obs) | 2.67 | 0.0 | 6.34 | 5.0;0.0;3.0 | 2 | 1.0 | 1.0 |
