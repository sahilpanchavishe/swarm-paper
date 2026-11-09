# Protocol addendum, Day 15 (v3)
Generated 07 Nov 2026 17:15, before any run of the suites below. Day 15 result files present at generation time: d15_stratified_existing.csv.
Edited and owned by the authors. Notes marked [status] were added on 7 Nov 2026 and say when each study was specified relative to its run.

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
- Predictions: (A1) relayD K2+mem reaches at least 99% collision-free success in every scenario. (A2) hold-still relay K2+mem is at least 20 points lower in comp2_same than in conn1. (A3) memoryless GNN K=2 is within 5 points of its conn1 value in every disconnected scenario. [status] A1-A3 were generated into this file before any Study A run and were not changed afterwards (see git history). Study A is confirmatory.
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

- Dev gate (suite d15_margin_dev, seeds 5000-5009, one training seed): the margin rule is adopted if, for GNN K2, latched GNN K2 and relayD K2+mem, the margined filter has contact in at most 1% of episodes and collision-free success at least 5 points above or within 5 points below the plain filter. [status] Thresholds kept as written. The gate failed, see 3b and 3c. If the gate fails, one revised rule may be defined on dev seeds only and recorded here before the final run. Otherwise the margin results are reported as a limitation.
- Final conditions: link loss 0.3 and 0.5 (plain filter); delay 5 and 10 steps and noise 5 and 10 mm (plain and margined). Models: GNN K2, K3, K2+latched, relayD K2+mem.
- Predictions: (B1) under link loss the plain filter keeps contact at or below 1% and collision-free success within 5 points of the clean filtered value. [status] B2 and B3 are void because both margin rules failed on dev (3c). Study B started at about 23:30 on 7 Nov, before these status notes were committed, and the first delay results were visible in its log. Study B is therefore reported as exploratory, and B1 is an expectation, not a pre-specified test. Expectation for delay: high contact at 5 and 10 steps, as in the dev runs (40-90%).

## 4. Study C: token cut-off redo
To be specified in addendum v3.1 once the implementation is read. It will include cut times, five training seeds and a positions-only classifier probe, on seed block 60000+.

## 5. Study D: command change (confirmatory replication)
- Exploratory run on seeds 30000+ is the reference. This is a fresh replication on seeds 70000-70024, with and without the filter. Configuration unchanged: switch at step 550, horizon 1650.
- Predictions: [status] No predictions were written before this run and its results were seen before this note. Study D is reported as an exploratory replication, not a confirmatory one. Reference: paper/tables/t14_6_change.md.

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

## 3c. Dev outcome of the revised delay rule (written after the dev run, before the final B run)
Dev run d15_vm_dev (seeds 5000-5009, 1 training seed, 80 episodes per row). Success % / contact %:
delay 5, plain: GNN K2 95.0/40.0, latched 100/51.2, relayD K2+mem 96.2/13.8. Velocity-dependent margin: 67.5/0, 65.0/0, 43.8/0.
delay 10, plain (from d15_margin_dev): 91.2/83.8, 92.5/90.0, 95.0/78.8. Velocity-dependent margin: 48.8/0, 53.8/0, 12.5/0.
The adoption rule in 3b (contact at most 1%, success at most 5 points below plain) is not met: success falls by 27 to 83 points. Both margin rules remove contact only by blocking the formation. No further rule is tried on dev seeds.
Decisions: the final B run uses the plain filter only, under link loss 0.3 and 0.5, delay 5 and 10 steps, noise 5 and 10 mm. Predictions B2 and B3 are void. Prediction B1 stays. Delay safety is reported as an open limitation, with the dev results above described as exploratory (one training seed).

## 4b. Study C specification (supersedes section 4; written before any run on the 60000 block)
Exploratory reference: d9_token_cutoff.csv (seeds 10000+, start box 0.5, 25% observability). That run found about 24% success when the flag and token were removed from step 0 (chance for four formations), about 72-73% when removed at step 300 or 700 (error at the cut 33 mm and 4 mm), and about 6% when only the token was blanked with the observer flag left on.
Design: fresh seeds 60000-60019 (4 formations, 80 episodes per model and condition), 25% observability, start box 0.5, no safety filter, horizon 1100 steps.
Primary: GNN K2 (training seeds 0-4) and GNN K2+latched (seeds 0-2); cut times 0, 100, 200, 300, 500, 700, 1100; variants flagoff and flagon.
Secondary: GNN K1 and K3 (seeds 0-2); cut times 0, 300, 700, 1100; both variants.
Position-only readout: a parameter-free nearest-template classifier (the formation whose slot set the swarm is closest to), recorded at the cut and at the end. A learned probe is not part of the confirmatory design; positions at the cut are saved so one can be added later as exploratory.
Reported (descriptive, with the range over training seeds): success after removal, success conditional on the template readout being right or wrong at the cut, and the share of failures that end nearest to the formation the policy builds when never informed.
Predictions (written on 7 Nov 2026 before the Study C run started, committed before its first episode): (C1) flagoff at t_cut=0 gives at most 35% success for every model (chance is 25%). (C2) for GNN K2, flagoff at t_cut 300 to 700 gives success above 50% and at least 10 points below the never-cut control (t_cut=1100). (C3) flagon gives at most 15% success at every t_cut before 1100. The thresholds come from the exploratory reference above.
Interpretation rule: if success after removal tracks how well the template readout identifies the formation, spatial cues explain it, and the paper does not claim implicit communication. If it does not, say so and call it unexplained.

## 8. Exploratory additions (written 8 Nov 2026 before they were run; none is confirmatory)
- Frontier sweep (suite d15_frontier): filtered runs on dev seeds 5000-5009 with start-box half-width 0.5, 0.7, 0.9, 1.1 and 1.3, to see where the engineered relay or the GNN stops reaching the formation. The GNN divides positions by 1.1, so 1.3 is outside its input range and any GNN drop there is partly an input-scaling effect.
- Drift measurement (d15_drift.py, seeds 80000-80019, wide box, 25% observability, no filter): distance of uninformed robots to the formation centre and number of communication components over time for GNN K2. Tests whether the memoryless GNN brings uninformed robots together on its own.
- Failure replay (d15_replay.py): the 12 worst failing filtered wide-box memoryless-GNN episodes, re-simulated; reports speed and filter activity of the worst robot. Reports nothing if the replay does not reproduce the saved errors.

## 9. Study D confirmatory replication (written 8 Nov 2026 before any run on seeds 90000-90024)
Exploratory reference: d15_change (seeds 70000-70024), whose results were seen. This is a fresh replication (suite d15_change_conf), same policies and conditions, with and without the filter, switch at step 550, horizon 1650.
Predictions, thresholds taken from the exploratory reference:
(D1) naive latching fails: GNN K2+latched/naive and relay K2+naive reach at most 5% success in every condition.
(D2) versioned latching works: GNN K2+latched/ver, relay K2+ver and relay K3+ver reach at least 93% success at 25% observability and at least 98% at 50%.
(D3) with the filter on, contact is at most 1% for every policy.
(D4) memoryless GNN K2 reaches at least 85% at 25% and at least 95% at 50% observability.
Primary metric collision-free success; same statistics as Study A. Anything not listed is exploratory.

## 10. Exploratory analyses added 8 Nov 2026 (specified before running)
- Failure classification: re-simulate all failing filtered wide-box memoryless-GNN episodes (d15_replay.py --max 100) and label the worst robot at the end: held_by_filter (filter altered its command in more than 50% of the last 100 steps), still_moving (speed at least 20 mm/s), stopped_isolated (nearest neighbour more than 100 mm), otherwise stopped_in_crowd. Thresholds are arbitrary and fixed here.
- Learned probe (d15_probe.py): logistic regression on rotation-free features (sorted distances to the centroid and sorted pairwise distances) of the swarm positions at the cut, K2 flagoff runs of d15_cutoff_primary.csv, 5-fold cross-validation grouped by start seed, to test whether positions alone identify the formation.

## 11. Study B confirmatory replication (written 8 Nov 2026 before any run on seeds 100000-100024)
Same suite as d15_filter_dist (plain filter, link loss 0.3 and 0.5, delay 5 and 10 steps, noise 5 and 10 mm), fresh seeds. Thresholds from the exploratory run, which was seen.
(B1) Link loss: contact at most 1% for every policy and level. Collision-free success at least 80% for every policy except memoryless GNN K2 at 25% observability and loss 0.5, which is expected between 80% and 90%.
(B2) Noise 5 mm: contact at most 1%. Noise 10 mm: contact at most 6%.
(B3) Delay: contact at least 25% for every policy at 5 steps and at least 80% at 10 steps. The filter is not expected to be safe under delay.
Primary metric collision-free success, statistics as Study A. Anything not listed is exploratory.

## 12. Observer imperfection study (written 8 Nov 2026, before any code or run)
Design (all conditions use the plain filter off, wide-box off, start box 0.5, 25% and 50% observability, fresh seeds 110000-110024, four formations, policies as in Study A: GNN K2, GNN K2+latched, relayD K2+mem, relay K2+mem):
(O1) Missed command: at every control step each observer independently fails to read the command with probability p in {0.3, 0.6}.
(O2) Wrong command: at every control step each observer independently reads a uniformly random wrong formation with probability p in {0.1, 0.3}.
(O3) Contradiction: half of the observers (rounded down, at least one) read a different, fixed wrong formation for the whole episode; at 25% observability this is one of two observers.
Success is measured against the true formation. Old behaviour is unchanged when none of these keys is set.
Predictions (written 8 Nov 2026 before any observer run; no pilot data, thresholds are expectations, not fits). (O1) Policies with memory (GNN K2+latched, relayD K2+mem) are barely affected by missed commands: at least 95% success at both miss rates and both observability levels. Memoryless GNN K2 degrades with the miss rate but stays above 70% at p=0.6 and 25% observability. The hold-still relay without memory degrades most. (O2) Wrong commands hurt the memory policies more than the memoryless GNN, because a wrong token that has been copied by other robots is not corrected. At p=0.3 and 25% observability I expect GNN K2+latched and relayD K2+mem to score at least 10 points below memoryless GNN K2, and I expect every policy to fall below its clean success rate. (O3) With a fixed contradicting observer, every policy scores at most 60% at 25% observability and at least 20 points below its clean value, because the swarm receives two different commands for the whole episode. At 50% observability (two of four observers wrong) I expect the same or worse. Primary metric: success against the true formation. Any result not covered here is exploratory.

## 12b. Observer study implementation notes (written 8 Nov 2026 before any run)
Suite d15_obs: the six conditions are clean (control), miss 0.3 and 0.6, wrong 0.1 and 0.3, contra; filter off; seeds 110000-110024. The imperfections act on policies that read observers through the registers (GNN K2, GNN K2+latched, relayD K2+mem, relay K2+mem). A memoryless relay without memory is not run, so the O1 sentence about it is not tested. Contradiction uses one observer at 25% and two at 50%, drawn in observer order, and its wrong formation is fixed per episode.

## 13. Delay-aware filter, confirmatory (written 9 Nov 2026 before any run on seeds 120000-120024)
Filter: d15_pred.py (predicts current positions from delayed snapshots by finite-difference velocity, pad 0.02 m). Pad chosen on dev seeds 5000-5009 (d15_pred_dev): at delay 5 and pad 0.02 contact was 0.0% and success 85-95%; pad 0.04 lowered success to 60-70%. Dev results were seen. Suite d15_pred_conf: GNN K2, K2+latched, K3, relayD K2+mem, delay 5 and 10, 25% and 50% observability, seeds 120000-120024.
(P1) Contact at most 2% for every policy and cell. (P2) Collision-free success at least 75% at delay 5 and at least 65% at delay 10 for every policy. (P3) Success lower at delay 10 than delay 5 for the GNN policies. The filter has no formal guarantee; the study tests it empirically.
Primary metric collision-free success. Anything not listed is exploratory.

## 13. Delay-aware filter, confirmatory (written 9 Nov 2026 before any run on seeds 120000-120024)
Filter: d15_pred.py (predicts current positions from delayed snapshots by finite-difference velocity, pad 0.02 m). Pad chosen on dev seeds 5000-5009 (d15_pred_dev): at delay 5 and pad 0.02 contact was 0.0% and success 85-95%; pad 0.04 lowered success to 60-70%. Dev results were seen. Suite d15_pred_conf: GNN K2, K2+latched, K3, relayD K2+mem, delay 5 and 10, 25% and 50% observability, seeds 120000-120024.
(P1) Contact at most 2% for every policy and cell. (P2) Collision-free success at least 75% at delay 5 and at least 65% at delay 10 for every policy. (P3) Success lower at delay 10 than delay 5 for the GNN policies. The filter has no formal guarantee; the study tests it empirically.
Primary metric collision-free success. Anything not listed is exploratory.

## 14. Exploratory addition (written 9 Nov 2026 after the observer results were seen): memoryless relays under observer errors
Suite d15_obs_nomem runs relay K2 and relayD K2 (no register memory) on the d15_obs conditions, seeds 110000-110024, to test the O1 sentence left untested in section 12b. Memoryless relays now read the imperfect observer command; behaviour is unchanged when the keys are unset. Exploratory, not confirmatory.

## 15. Observer study with the safety filter, matched filter off/on, randomised persistent errors (written 9 Nov 2026 before any run on seeds 130000-130024)

Reason. Review of the Day 15 results showed that (a) the observer study d15_obs ran with the safety filter off, so contact in it is mostly the raw policy's contact (memoryless GNN K2 reaches only 40.7% collision-free success in the clean condition), (b) its paired statistics used formation success, and a post hoc re-analysis on collision-free success (d15_Ocf_paired.csv, exploratory) changes the ranking, and (c) in the contradiction condition the corrupted observers are the lowest-indexed observers, so robot index can influence the result. miss and wrong are resampled at every step (intermittent errors); contra is fixed for the whole episode (persistent errors).

Design. Suite d15_obs_safe: GNN K2, GNN K2+latched, relayD K2+mem, relay K2+mem; 3 training seeds; 25 scenarios per cell; observability 25% and 50%; seeds 130000-130024 (fresh block); every condition run with the safety filter off and on on identical scenarios (matched). Conditions: clean; miss 0.6; wrong 0.3; contra with half of the observers wrong, chosen at random (contra_rand, own generator, seed [seed,67]); contra_n=1, one persistently wrong observer chosen at random. Observers with errors are chosen per scenario, identically for every policy.

Primary metric: collision-free success (formation success and contact reported alongside). Paired two-level bootstrap with joint training-seed resampling, B=4000, Holm per family, as in the earlier addenda.

Predictions (filter on unless stated; observability 25% and 50% both):
(S1) Contact at most 2% for every policy in the clean, miss 0.6 and wrong 0.3 conditions.
(S2) wrong 0.3: collision-free success of memoryless GNN K2 exceeds that of GNN K2+latched and of relay K2+mem by at least 20 points.
(S3) miss 0.6 at 25%: GNN K2+latched and the relays at least 95% collision-free success; memoryless GNN K2 at most 80%.
(S4) contra_n=1 at 50% (one wrong observer of four): no policy reaches 50% collision-free success.
(S5) contra with half of the observers wrong, randomised: every policy at most 10% collision-free success.

Anything not listed is exploratory. The filter has no formal guarantee; the study tests it empirically. The matched filter-off arm tests whether the filter, not the observer errors, is responsible for the contact seen in d15_obs.

## 16. Matched three-arm delay study (written 10 Nov 2026 before any run of arms A and B)
Purpose: section 13 showed that the predictor filter gives 0% contact under delay, but it had no matched control, so the gain over the earlier filter and over no filter is not isolated. This study runs the missing arms on the same scenarios as section 13.
Design: arm A = no safety filter (suite d15_arm_nofilter); arm B = pairwise filter applied to the delayed positions, no prediction, plain() parameters (suite d15_arm_stale); arm C = predictor filter, pad 0.02 m (already run as d15_pred_conf, seeds 120000-120024, results seen before this section was written). Policies GNN K2, GNN K2+latched, GNN K3, relayD K2+mem; delays 5 and 10 control steps; 25% and 50% observability; seeds 120000-120024, model seeds 0-2. Everything except the filter is identical across arms.
Primary metric: collision-free success (reaches the true formation within the error threshold with no robot contact). Secondary: success, contact rate, minimum clearance, final maximum error. Paired comparisons B vs A, C vs B and C vs A per policy, delay and observability, joint-seed bootstrap, Holm over the whole study.
Disclosure: when this section was written, the delay-free results of d15_obs_safe (section 15) had been seen. There, without a filter, contact was 63-77% for memoryless GNN K2, 4-6% for GNN K2+latched and 0% for both relays. The predictions below are informed by that.
(T1) Without a filter, contact is at least 40% for memoryless GNN K2 and GNN K3 at both delays, below 20% for GNN K2+latched, and at most 5% for relayD K2+mem.
(T2) The delayed-position filter (B) is not a useful fix: at delay 10 its contact is at least 50% for every policy, and its collision-free success for GNN K2 is within 10 points of arm A.
(T3) The predictor filter (C) beats B on collision-free success by at least 50 points at delay 10 for every GNN policy, significant after Holm.
(T4) For the GNN policies, C beats A on collision-free success by at least 30 points. For relayD K2+mem, C and A are within 10 points of each other, because the relay already has high collision-free success without a filter.
(T5) The predictor filter costs at most 10 points of plain success relative to arm A for every policy.
Anything not listed is exploratory. Misses are recorded in paper/misses_d15.md.

## 16a. Clarification of T4 (written 10 Nov 2026, before any run of arms A and B)
T4 says "the GNN policies". It is read as memoryless GNN K2 and GNN K3. GNN K2+latched is not scored in T4. In the delay-free, filter-off results of section 15 that policy already had about 95% collision-free success and 4-6% contact, so a gain of 30 points from a filter was not possible. This was visible when section 16 was written, so the original wording of T4 was a mistake. GNN K2+latched is reported without a threshold. T1, T2, T3 and T5 are unchanged.

## 16a. Clarification of T4 (written 10 Nov 2026, before any run of arms A and B)
T4 says "the GNN policies". It is read as memoryless GNN K2 and GNN K3. GNN K2+latched is not scored in T4. In the delay-free, filter-off results of section 15 that policy already had about 95% collision-free success and 4-6% contact, so a gain of 30 points from a filter was not possible. This was visible when section 16 was written, so the original wording of T4 was a mistake. GNN K2+latched is reported without a threshold. T1, T2, T3 and T5 are unchanged.

## 17. Provenance and confidence study (written 10 Nov 2026 before any code or run; arm results of section 16 not yet seen)
Question: can a non-learned relay that counts distinct observer sources resolve a persistently wrong observer, and does an observer's reported confidence help or hurt?
Design: 50% observability (four observers), filter on (plain() parameters), delay 0, four formations, seeds 140000-140024, model seeds 0-2 for the GNN. Exactly one observer is persistently wrong for the whole episode (random choice per episode, own seeded generator); the other three read the true command. Every observer carries a confidence value that the policies may use.
Conditions: (C0) control, all four correct, equal confidence 0.7. (C1) one wrong observer, all confidences 0.7. (C2, reviewer item B4) the wrong observer has confidence 0.9, the correct ones 0.5. (C3, reviewer item B5) the wrong observer has confidence 0.3, the correct ones 0.9.
Policies: relay K2+mem (existing, keeps the first token it receives); relay K2+maxconf (adopts the token with the highest confidence it has seen); relay K2+prov (keeps, per robot, the set of distinct observer sources that vouched for each formation, forwards source identities with the message, and adopts the formation with the most distinct sources); GNN K2+latched (existing, does not use confidence).
Not run: 25% observability, where one wrong observer out of two is a tie that no policy can resolve without outside information; this is stated as a limit, not tested.
Primary metric: collision-free success. Secondary: success, contact rate, time to correct token at all robots if logged.
(V1) relay K2+mem and GNN K2+latched score at most 20% in C1, C2 and C3, as the existing policies did with one wrong observer in section 15 (5% at 50%).
(V2) relay K2+maxconf scores at most 20% in C2 and at least 70% in C3. In C1 it behaves like the existing relay (at most 20%), because equal confidences give it nothing to separate the sources.
(V3) relay K2+prov scores at least 60% in C1, C2 and C3, because it does not use confidence and three of the four sources are correct. In C0 it scores at least 95%, so it does not lose against the existing relay when nobody is wrong.
Anything not listed is exploratory. Misses are recorded in paper/misses_d15.md.
