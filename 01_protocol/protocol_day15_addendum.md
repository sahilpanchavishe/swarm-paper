# Protocol addendum, Day 15 (v3)
Generated 07 Nov 2026 17:15, before any run of the suites below. Day 15 result files present at generation time: d15_stratified_existing.csv.
DRAFT: read, edit and own every line. Items marked TODO(you) are yours to decide.

## 0. Status of earlier analyses
- The stratification of the Day 12/14 runs by initial communication graph (results/d15_stratified_existing.csv) was post hoc and exploratory. It showed that relays with a default drift toward the formation centre succeed in disconnected starts, that hold-still relays do not, and that the memoryless GNN K=2 does about equally well in all strata. It motivated Study A.
- Everything below is confirmatory unless labelled exploratory. Parameters are chosen on dev seeds 5000-5009 only. Each final seed block is used once, and nothing is tuned on it.

## 1. Fixed ingredients
- Safety filter: dsafe 0.10, gamma 20, sense 0.30 (chosen on dev seeds on Day 14). Empirical, no formal guarantee.
- Primary metric: collision-free success (final error below 25 mm and no contact). Secondary: success, contact %, median minimum clearance, maximum per-robot error.
- Statistics as in Methods 2.9: shared training-seed bootstrap, per-seed exact McNemar (significant only if every seed is and Holm p < 0.05, Holm within each study's own family), p below 0.0001 reported as such.

## 2. Study A: disconnected starts (confirmatory, evaluation only, no retraining)
- Start box half-width 0.7 m (chosen because 2- and 3-component starts are common there, 32% and 33% of random draws). Communication radius 0.5 m.
- Scenarios: conn1 (connected, random observers), comp2_same (2 components, all observers in one), comp2_split (2 components, observers in both), comp3_same (3 components, all observers in one). Observability 25% and 50%. Filter on.
- Models: GNN K=1, K=2, K=3, K=2+latched (training seeds 0-2); relay K2+mem, relayD K2+mem, relay K2, relayD K2.
- Predictions: (A1) relayD K2+mem reaches at least 99% collision-free success in every scenario. (A2) hold-still relay K2+mem is at least 20 points lower in comp2_same than in conn1. (A3) memoryless GNN K=2 is within 5 points of its conn1 value in every disconnected scenario. TODO(you): confirm or change these.
- Paired comparisons per scenario and observability level: GNN K2 vs relayD K2+mem; latched vs relayD K2+mem; relayD K2+mem vs hold-still relay K2+mem; GNN K2 vs relay K2; latched vs memoryless GNN K2.

## 3. Study B: filter under link loss, delay and noise (confirmatory)
- Link loss is expected not to affect the filter, because the filter uses sensing, not communication. Delay and noise act on the positions the filter sees.
- Margin rule, no tuned constants: dsafe_eff = dsafe + 2 * V_MAX * DT * delay + 6 * sigma_noise, and sense_eff = sense + the same term. It assumes robots move at most V_MAX per second, with DT the control step.

| condition | margin (mm) | dsafe_eff (mm) |
|---|---|---|
| delay 5 steps | 41.2 | 141.2 |
| delay 10 steps | 82.4 | 182.4 |
| noise 5 mm | 30.0 | 130.0 |
| noise 10 mm | 60.0 | 160.0 |

- Dev gate (suite d15_margin_dev, seeds 5000-5009, one training seed): the margin rule is adopted if, for GNN K2, latched GNN K2 and relayD K2+mem, the margined filter has contact in at most 1% of episodes and collision-free success at least 5 points above or within 5 points below the plain filter. TODO(you): confirm the thresholds. If the gate fails, one revised rule may be defined on dev seeds only and recorded here before the final run. Otherwise the margin results are reported as a limitation.
- Final conditions: link loss 0.3 and 0.5 (plain filter); delay 5 and 10 steps and noise 5 and 10 mm (plain and margined). Models: GNN K2, K3, K2+latched, relayD K2+mem.
- Predictions: (B1) under link loss the plain filter keeps contact at or below 1% and collision-free success within 5 points of the clean filtered value. (B2) under delay and noise the margined filter has less contact than the plain filter and at most 1% contact. (B3) the margin costs at most 5 points of success. TODO(you): confirm.

## 4. Study C: token cut-off redo
To be specified in addendum v3.1 once the implementation is read. It will include cut times, five training seeds and a positions-only classifier probe, on seed block 60000+.

## 5. Study D: command change (confirmatory replication)
- Exploratory run on seeds 30000+ is the reference. This is a fresh replication on seeds 70000-70024, with and without the filter. Configuration unchanged: switch at step 550, horizon 1650.
- Predictions: TODO(you): write two or three sentences from paper/tables/t14_6_change.md before running.

## 6. Suites and seed blocks
| suite | units | episodes | seeds |
|---|---|---|---|
| d15_margin_dev | 12 | 960 | 5000-5009 |
| d15_filter_dist | 100 | 20000 | 40000-40024 |
| d15_change | 40 | 8000 | 70000-70024 |
| d15_A_conn1 | 16 | 3200 | 50000-50024 |
| d15_A_comp2_same | 16 | 3200 | 50000-50024 |
| d15_A_comp2_split | 16 | 3200 | 50000-50024 |
| d15_A_comp3_same | 16 | 3200 | 50000-50024 |

## 7. Deviations
Anything not listed here, or changed after a run, is exploratory and labelled so in the paper.

## 3b. Dev outcome and revision (written after the dev check, before the revised rule was run)
Dev check (d15_margin_dev, seeds 5000-5009, 1 training seed, 80 episodes per row). Success / contact, plain filter vs static margin:
delay 10: GNN K2 91.2/83.8 vs 42.5/0.0; latched 92.5/90.0 vs 43.8/0.0; relayD K2+mem 95.0/78.8 vs 46.2/0.0.
noise 10 mm: GNN K2 98.8/2.5 vs 71.2/0.0; latched 100/2.5 vs 76.2/0.0; relayD K2+mem 100/1.2 vs 66.2/0.0.
The gate in section 3 was worded in terms of collision-free success, which the static margin technically met under delay only because the plain filter's collision-free success was 10-21%. The intended criterion was success (prediction B3), and the static margin fails it by about 50 points. We treat the static margin as failed. It is reported as an exploratory dev result: it removes contact by blocking the formation.
Decisions: (1) noise: plain filter only, no margin. (2) delay: one revised rule, chosen before it is run. Pair margin = delay * DT * (|u_i| + |u_j|), the distance the two robots can close during the delay at their commanded speeds; barrier radius dsafe + margin, so robots settling into formation need almost no extra spacing. No tuned constants. (3) The revised rule is checked on dev seeds (suite d15_vm_dev). It is adopted for the final run if contact is at most 1% and success is at most 5 points below the plain filter. Otherwise delay is reported as an open limitation and the final delay runs use the plain filter only.
