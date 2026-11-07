# Results outline (Day 14), claims in the order the paper should make them

Rule for every sentence: cite a table row. CONFIRMATORY = seeds 20000-20049, protocol fixed before the run. EXPLORATORY = seeds 30000+ or added after the protocol; descriptive only.
"Significant" means Holm p < 0.05 AND every training seed points the same way (t14_7). All numbers below come from paper/tables/t14_*.csv; re-check them there before quoting.

## 3.1 Reference points (t14_1)
- Oracle succeeds always. A memory-free relay with the same depth reaches 87.0% (K=2, 25% obs). With memory: 99.5-100%.
- relayD (uninformed robots head for the default slot) without memory: 83.0% at 25% obs, mean max error 83 mm.

## 3.2 Primary claim: two message-passing rounds beat one (CONFIRMATORY, t14_1, t14_7)
- K=2 vs K=1 at 25% obs: +31.6 points, 95% CI 24.7 to 39.1, 5 of 5 training seeds, Holm p < 0.0001. At 50%: +27.0.
- K=3 vs K=2: 0.0 points (CI -2.7 to 2.3). No benefit from a third round. State this plainly; drop any "K=3 is better" language.

## 3.3 Not a parameter-count effect (CONFIRMATORY, t14_2, t14_7; training seeds 0-2 only)
- 25% obs, matched seeds: K=1 69.3%, K=1 widened to K=2's size 73.0% (+3.7, CI -5.2 to 12.5, not significant), K=2 97.5%, K=2 with one shared weight set (K=1's size) 97.5%, wide no-message network 0%.
- K=2 vs wide K=1: +24.5 (Holm < 0.0001). K=2 vs shared K=2: 0.0.
- Limitation to state: 3 training seeds, so the smallest per-seed p is 0.0625.

## 3.4 Part of the gain is not token passing (CONFIRMATORY, t14_7)
- At 100% obs every robot already knows the command, yet K=2 beats K=1 by 18.7 points. Claim "a second hop of neighbour information helps", not "two hops spread the command".

## 3.5 Latched token and relays with memory (CONFIRMATORY, t14_1, t14_7)
- Latched K=2 vs memoryless K=2: small (+2.5 on matched seeds, not significant). Latched GNN equals a relay with memory (100%). Learning shows no advantage over a good relay in the standard box. Say so.
- vs the memory-free relay: GNN K=2 +11.9 at 25% (significant), K=1 at 50% is -18.4 (the relay wins).

## 3.6 Contact, and what the safety filter does (CONFIRMATORY, t14_3, t14_7, figR)
- Without the filter, contact occurs in 57% (K=2) and 30% (K=3) of episodes at 25% obs; K=3 is 85% in one training seed. Collision-free success: K=2 42.6%, K=3 69.4%.
- With the pre-selected filter (dsafe 0.10 m, gamma 20, chosen on dev seeds, tie-break written before the run): contact 0.0% in 11,200 episodes; collision-free success K=2 98.5%, K=3 99.2%, K=1 79.8%.
- K=2 vs K=1 with the filter: +18.7 (25%) / +13.2 (50%), still significant. Report both gaps.
- Caveats: empirical filter, no formal guarantee, accurate positions, simulation. Contact under sensing delay is 75-95% for everything (t14_5), filter not tested there.

## 3.7 Exploratory: robustness (t14_5, figS)
- Link loss 0.5 at 25% obs: GNN K=2 84.2%, K=3 93.8%, memory-free relay 51.3%, relayD 56.3%; memory relays and latched GNN 100%.
- Delay 10 steps: success stays 98-99% for GNNs but contact is 94-96% for the GNNs and 89-96% for relays.
- Noise 10 mm: success 99-100% for GNNs, contact 65% (K=2), 24% (latched).

## 3.8 Exploratory: wide start box, 1.8 m (t14_4, t14_7, d14_widebox_*)
- Hold-still relays with memory fall to 31-32%; relayD with memory reaches 100%, with or without the filter, with zero contact.
- GNN K=2 93%, K=3 95.7%, latched 99.7% (filtered); latched equals relayD+mem; memory-free K=2 and K=3 are 7 and 4 points lower, same direction in all 3 seeds, not significant.
- Memory-free relayD gets 76%; GNN K=2 is higher (+17 points) but not significant after Holm.
- Honest reading: the learned policy matches a hand-designed relay only when it has memory. It is not better than a well-designed relay. The contribution is that it learns this behaviour from demonstrations.

## 3.9 Exploratory: mid-episode command change (t14_6)
- At 25% obs: memory-free GNN K=2 85%, K=3 84%, K=1 60%; version-aware latched GNN 96% (contact 12.7%); version-aware relay 96% (contact 0). Naive latching: 0% for both. No GNN advantage here.

## 3.10 Max error (all tables)
- Report mean max per-robot error beside every success rate. K=2 at 25% obs: 6.4 mm; K=1: 146 mm. In the wide box (filtered) K=2 is 38 mm with a p95 of 316 mm, so a few episodes leave one robot far off.

## Claims the paper must NOT make
- "Learning beats engineering" (relayD+mem equals the GNN in the wide box; memory relays equal it in the standard box).
- "K=3 is better than K=2".
- "The filter guarantees safety".
- Anything from the Day 6 numbers, figures A-P, or the `*_d13` tables.
- Any significance claim for wide-box comparisons.

## Limitations to write
8 robots only; a one-hot command token, not vision or gestures; simulation, accurate positions; 3 training seeds for K=0, latched, and capacity controls; empirical filter, untested under delay; exploratory experiments have 25 scenarios x 4 formations and no confirmatory status; hand-designed relay baselines are given the formation geometry.
