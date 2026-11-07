# Protocol addendum, Day 12 (commit this file BEFORE running `d12_eval.py confirm`)

`protocol.md` stays as it is. This addendum records what changed after the critical review of the Day 10 repository, and fixes the confirmatory
analysis before any confirmatory episode exists. Anything not listed here follows `protocol.md`.

## 1. Seed blocks
| Block | Seeds | Use |
|---|---|---|
| demonstrations | 0-1199 | training |
| DAgger rollouts | 1300-4299 | training |
| dev | 5000-5009 | choosing the safety-filter parameters only |
| Day 6 test | 10000-10049 | reused in exploratory work (Days 9-10); now EXPLORATORY, kept for the record |
| **confirmatory** | **20000-20049** | **every headline claim; evaluated once per model, nothing tuned on it** |
| exploratory | 30000+ | packet loss, delay, noise, wide start box, command change |

## 2. Models
Main memoryless GNN K=1, 2, 3: training seeds 0-4 (seeds 3-4 added). K=0 and the latched K=2: training seeds 0-2 (stated as a limitation).
Capacity controls, training seeds 0-2, same recipe: `wide1` (K=1, width 78, about the parameter count of K=2), `shared2` (K=2 with one set of
weights in both layers, the parameter count of K=1), `mlpwide` (K=0, width 110, the parameter count of K=2).

## 3. Statistics (replaces the Wilcoxon-on-success analysis of `d6_analysis.py`, which is kept only as an exploratory table)
* Binary success: exact McNemar test. Final error: Wilcoxon signed-rank. Max per-robot error and contact outcomes: descriptive.
* Every comparison is tested separately for each training seed (A's seed s against B's seed s; a no-learning relay is paired with every seed).
  A comparison is significant only if it is significant in EVERY seed (largest p over seeds), then Holm over the whole family below, and the
  per-seed differences all point the same way.
* Intervals: two-level bootstrap resampling test scenarios (formation, seed) and training seeds. Pooled Wilson intervals are not used for claims.
* Family (Holm): the 16 comparisons of `d6_analysis.family()` plus the 12 capacity comparisons of `d12_stats.py --extra`, on the confirmatory block.
* Primary hypothesis unchanged: GNN K=2 vs K=1, success at 25 % observability. Second primary: GNN K=2 vs K=1-wide at 25 % observability.

## 4. Decision rules fixed in advance
* "Communication depth matters" may be written only if BOTH hold on the confirmatory block: K=2 beats K=1-wide at 25 % observability, and
  K=2-shared beats K=1. Otherwise the claim is reduced to "a second layer helps, and capacity cannot be ruled out as the reason".
* If the latched GNN does not beat "relay with memory" in collision-free success, the paper states that a learned controller adds no
  formation-accuracy advantage over a one-bit-memory relay in this task.
* Contact is centre distance < 0.074 m; near-contact is clearance < 0.09 m. Collision-free success = success and no contact.
  Report max per-robot final error, not only the mean.

## 5. Safety filter
Decentralised pairwise velocity filter (barrier h = d^2 - dsafe^2, each robot enforces half of dh/dt >= -gamma*h, neighbours within 0.30 m).
Parameters (dsafe, gamma) are chosen on the DEV seeds by the highest mean collision-free success of GNN K=2 and K=3 at 25 % observability; if two
settings are within 1 point, take the larger dsafe. It is an empirical filter with no formal guarantee. Evaluated once on the confirmatory block.

## 6. Exploratory experiments (no hypothesis tests; descriptive tables only, labelled exploratory)
Link loss 0.1/0.3/0.5 per step, sensing delay 5/10 control steps, position noise 5/10 mm, 1.8 m start box (initial components reported),
mid-episode command change (switch at 17.6 s, horizon 52.8 s; relays and GNNs with explicit per-robot stale/fresh token registers).

## 7. Deviations to state in the paper
Protocol.md promised 5 training seeds for all K; K=0 and latched have 3. Day 6 test seeds were reused in exploratory work. The Day 12 engine
reproduces Day 3/6/9 results (reported by `d12_eval.py repro`). Training data regenerated on a different machine, if that happens, must be stated.

## 5a. Safety-filter selection outcome (written before safety_final)
On the dev seeds at 25 % observability the mean collision-free success of K=2 and K=3 was 97.5 for both (dsafe, gamma) = (0.10, 10) and (0.10, 20); the written rule does not separate them. Tie-break, fixed here before any confirmatory safety run: lower mean near-contact steps on the dev seeds, which selects (0.10, 20).

## 5b. Post-hoc exploratory follow-up (7 Nov, about 13:15, written before the run finished)
Motivated by the contact asymmetry seen in the unfiltered wide-box results: the wide-box suite re-run with the filter fixed on the dev seeds (dsafe 0.10, gamma 20) on seeds 30000-30024. Nothing is tuned. Exploratory.

## 5b. Post-hoc exploratory follow-up (7 Nov, written after the run completed)
Motivated by the contact asymmetry seen in the earlier unfiltered wide-box results. The wide-box suite was re-run with the filter parameters fixed on the dev seeds (dsafe 0.10, gamma 20) on seeds 30000-30024. Nothing was tuned on these results. Exploratory; no confirmatory claim rests on it.
