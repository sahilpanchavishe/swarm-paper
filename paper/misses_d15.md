# Predictions that were not met (report all of them in Results)
- D4 (Study D confirmatory): filtered memoryless GNN K2 at 25% observability scored 84.3% against a threshold of 85%.
- C3 (Study C): GNN K3 with the flag left on reached about 22% at cuts 300 and 700, above the 15% threshold.
- O2 (observers): memoryless GNN K2 did not fall below its clean success rate under wrong commands (98.0% vs 98.0% at p=0.3, 25% observability).
- P3 (delay-aware filter): success at delay 10 was not consistently lower than at delay 5 for the GNN policies.
- O1 (observers): the hold-still no-memory relay sentence was untested in section 12; tested later as an exploratory addition (protocol section 14).
Study labels: A, C, D (replication), B (replication), observers, delay filter = confirmatory (delay pad tuned on dev seeds 5000-5009).
Exploratory: original B and D runs, drift, frontier, replay, failure classes, probe, no-memory relay.
