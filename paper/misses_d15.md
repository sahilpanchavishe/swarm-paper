# Predictions that were not met (report all of them in Results)
- D4 (Study D confirmatory): filtered memoryless GNN K2 at 25% observability scored 84.3% against a threshold of 85%.
- C3 (Study C): GNN K3 with the flag left on reached about 22% at cuts 300 and 700, above the 15% threshold.
- O2 (observers, filter off, section 12): memoryless GNN K2 did not fall below its clean success rate under wrong commands (98.0% vs 98.0% at p=0.3, 25% observability).
- P3 (delay-aware filter, section 13): success at delay 10 was not consistently lower than at delay 5 for the GNN policies.

# Untested when written, tested afterwards
- O1 (section 12): the sentence about the hold-still relay without memory could not be tested in the original suite. It was tested later as an exploratory addition (protocol section 14) and held there. It is not a miss, and it is not confirmatory.

# Predictions met, with thin margins (do not describe as strong validation)
- S1-S5 (section 15, safety-on observer study, seeds 130000-130024): all five met. S3 passed with memoryless K2 at 78.7% against a limit of 80%. S2 at 50% observability passed with a gap of 27.3 points against a threshold of 20.

# Pending
- T1-T5 (section 16, matched three-arm delay study): arms A and B not yet run when this file was updated.

# Labels
- Confirmatory: A, C, D (replication), B (replication), observers section 12 (formation success), safety-on observer study section 15, delay filter section 13 (pad tuned on dev seeds 5000-5009).
- Exploratory: original B and D runs, drift, frontier, replay, failure classes, probe, no-memory relay (section 14), and every collision-free analysis of the observer study and Study D (added after the first results were seen).

# Limitations that follow from these records
- In the original observer study (section 12) the contradicting observers were the lowest-numbered ones, so robot ID could influence the result. Section 15 randomises the choice.
- miss and wrong are resampled at every control step, so they model intermittent errors, not a persistently wrong observer. Only the contradiction conditions are persistent.
- Latched GNN versus relayD K2+mem is statistically indistinguishable in every observer cell, so the learned policy shows no advantage over the hand-built relay once memory is added. [Authors: write the sentence for this in your own words.]
