# Predictions that were not met (report all of them in Results)
- D4 (Study D confirmatory): filtered memoryless GNN K2 at 25% observability scored 84.3% against a threshold of 85%.
- C3 (Study C): GNN K3 with the flag left on reached about 22% at cuts 300 and 700, above the 15% threshold.
- O2 (observers, filter off, section 12): memoryless GNN K2 did not fall below its clean success rate under wrong commands (98.0% vs 98.0% at p=0.3, 25% observability).
- P3 (delay-aware filter, section 13): success at delay 10 was not consistently lower than at delay 5 for the GNN policies.
- T1 (section 16, no filter): GNN K2+latched had 72-73% contact at delay 5 and 92% at delay 10 (predicted below 20%), and relayD K2+mem had 71-73% and 92% (predicted at most 5%). Both predictions were extrapolated from the delay-free results and do not hold under delay. Memoryless GNN K2 and K3 met their part of T1.
- T4 (section 16, relayD clause): the predictor filter improved relayD K2+mem collision-free success over no filter by 63-64 points at delay 5 and 89 points at delay 10 (predicted within 10 points). The GNN K2 and K3 clause was met; GNN K2+latched was excluded from T4 by the clarification in section 16a.
- T5 (section 16): the predictor filter cost GNN K2+latched 11.0 points of plain success at delay 5 (both observability levels), against a limit of 10. All other cells were within 10 points.

# Untested when written, tested afterwards
- O1 (section 12): the sentence about the hold-still relay without memory could not be tested in the original suite. It was tested later as an exploratory addition (protocol section 14) and held there. It is not a miss, and it is not confirmatory.

# Predictions met, with thin margins (do not describe as strong validation)
- S1-S5 (section 15, safety-on observer study, seeds 130000-130024): all five met. S3 passed with memoryless K2 at 78.7% against a limit of 80%. S2 at 50% observability passed with a gap of 27.3 points against a threshold of 20.

# Section 16 (matched three-arm delay study) and section 17 (provenance and confidence): met
- T2: the filter on delayed positions is no better than no filter (collision-free success within 2.3 points for GNN K2 at delay 10; relayD +3 to +4 points, not significant).
- T3: the predictor filter beats the delayed-position filter by 81-83 points for every GNN policy at delay 10, Holm-significant (the Holm value sits at the bootstrap floor).
- V1, V2, V3 (section 17): all met. Count-by-source scored 100% collision-free in all four conditions, so the thresholds (at least 60%) were not informative at the top end.
- Caveat for section 17: the confidence values are fixed constants chosen for the design, not outputs of a classifier. The B4 and B5 rows show how each fusion rule behaves for a given confidence pattern. They are not evidence about real gesture classifiers.
- Caveat for section 17: count-by-source needs authenticated, distinct observer identities and an honest majority. With one wrong observer out of two (25% observability) or two out of four it is a tie and no rule here resolves it.
- Section 16 and 17 arms are confirmatory. Arm C of section 16 was run and seen before section 16 was written.

# Labels
- Confirmatory: A, C, D (replication), B (replication), observers section 12 (formation success), safety-on observer study section 15, delay filter section 13 (pad tuned on dev seeds 5000-5009).
- Exploratory: original B and D runs, drift, frontier, replay, failure classes, probe, no-memory relay (section 14), and every collision-free analysis of the observer study and Study D (added after the first results were seen).

# Limitations that follow from these records
- In the original observer study (section 12) the contradicting observers were the lowest-numbered ones, so robot ID could influence the result. Section 15 randomises the choice.
- miss and wrong are resampled at every control step, so they model intermittent errors, not a persistently wrong observer. Only the contradiction conditions are persistent.
- Latched GNN versus relayD K2+mem is statistically indistinguishable in every observer cell, so the learned policy shows no advantage over the hand-built relay once memory is added. [Authors: write the sentence for this in your own words.]
