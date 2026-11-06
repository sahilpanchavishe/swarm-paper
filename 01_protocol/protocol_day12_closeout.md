# Day 12 close-out: the confirmatory programme was not run

The Day 12 addendum (`protocol_day12_addendum.md`, commit 9e26a84) planned a confirmatory evaluation on fresh seeds 20000-20049,
two more training seeds, capacity-control models, a safety filter and exploratory robustness tests. **None of these were run**, because
of compute time. No episode on seeds 20000-20049 exists, and no Day 12 model was trained. The only Day 12 result is the reproduction
check (`results/d12_repro.csv`), which shows that the Day 12 engine reproduces the Day 3, 6 and 9 results exactly.

## What the paper reports instead
* The Day 6 main evaluation (test seeds 10000-10049, 3 training seeds per model), analysed with the corrected Day 11a statistics
  (`src/d11_stats.py`: exact McNemar per training seed, max p over seeds, Holm over the 16 comparisons, two-level bootstrap intervals).
* The Day 7 token ablation and the exploratory Day 9 and 10 analyses, labelled exploratory.

## Deviations the paper states
1. `protocol.md` planned 5 training seeds; 3 were used.
2. The test seeds 10000-10049 were reused in the exploratory analyses of Days 9 and 10, so they are not an untouched hold-out set.
3. The capacity confound between K = 1 and K = 2 was not controlled experimentally.
4. No collision-avoidance (safety) filter was evaluated; contact is reported as measured.
5. Packet loss, delay, noise, the wide start box and mid-episode command changes were not tested.
6. The first analysis (Wilcoxon on seed-averaged success) was replaced by the Day 11a analysis after it had been run.
