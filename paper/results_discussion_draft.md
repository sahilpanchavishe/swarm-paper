# Results, Discussion and Limitations (DRAFT 1)

Every number below is taken from the files in `results/` and `paper/tables/` (three training seeds for each model, including the latched model). `src/d8_check_numbers.py` and `src/d9_check_numbers.py` recompute them and fail if any differ. Sections 3.9 and 3.10 are exploratory: they were added after the protocol was frozen. Items marked **[CHECK]** need your decision or a check. Table and figure numbers follow `paper/tables/` and `paper/figs/`:
Table 1 success rates, Table 2 paired comparisons, Table 3 model validation, Table 4 token ablation, Table 5 plain imitation vs DAgger; Fig. A success vs K, Fig. B main bars, Fig. C token coverage, Fig. D token ablation, Fig. E contact rate, Fig. F plain imitation vs DAgger.

---

## 3 Results

### 3.1 Reference points: expert and no-learning relays

The oracle expert, with every robot informed, succeeds in 100 % of the 200 test episodes (95 % CI 98.1–100) with no contact. Its mean convergence time is 8.7 s and the slowest episode takes 14.8 s, well inside the 35.2 s horizon.

The relay baselines show how much the task depends on how far the gesture token travels (Table 1, Fig. A). At 25 % observability, the memoryless relay succeeds in 39.0 % of episodes for K = 1, 85.5 % for K = 2, 89.0 % for K = 3 and 89.5 % for K = 7. At 50 % observability it reaches 85.5 %, 97.0 %, 98.5 % and 98.5 % for the same hop counts. Relays that remember the token once received succeed in 100 % of episodes at 25 % observability and 99.5 % at 50 % for every K ≥ 1. No relay episode had contact. With no communication (K = 0) no relay episode succeeded at either level.

The coverage of the communication graph explains the shape of these curves (Fig. C). Measured on random start configurations, a token from two observers reaches on average 76.3 % of the robots within one hop, 92.8 % within two hops and 96.0 % within three, and 96.7 % at the largest hop count we tested. All eight robots are reached in 12.8 %, 65.0 % and 80.8 % of starts for K = 1, 2 and 3. With four observers the figures are 94.6 %, 98.3 % and 98.5 % of robots, and 64.6 %, 88.6 % and 90.4 % of starts. 

### 3.2 Primary result: K = 2 against K = 1 at 25 % observability

The pre-specified primary comparison favours K = 2 (Table 2, Fig. A, Fig. B). At 25 % observability the K = 2 GNN succeeds in 97.5 % of episodes (95 % CI 95.9–98.5) against 72.3 % (68.6–75.8) for K = 1. The paired difference is 25.2 points (95 % CI 20.8–29.7, Holm-corrected p < 0.0001), and the mean final error falls by 22.0 mm. The direction is the same for every training seed. K = 2 succeeds in 97.5 % of episodes for each of seeds 0, 1 and 2, while K = 1 succeeds in 75.5 %, 66.5 % and 75.0 %. The K = 1 result therefore varies between training runs by about 9 points, while K = 2 does not vary at all.

The gap persists at 50 % observability (K = 2: 98.2 %, K = 1: 79.0 %; difference 19.2 points, CI 15.3–23.2, Holm p < 0.0001). Without any communication (K = 0), the GNN never succeeds at 25 % observability and succeeds in 0.2 % of episodes at 50 %.

### 3.3 Part of the K = 1 deficit is network depth, not gesture passing

At 100 % observability every robot sees the gesture, so the number of message-passing layers should not matter for the gesture itself. K = 1 nevertheless reaches only 83.5 % (80.3–86.3), against 99.5 % for K = 2 and 99.8 % for K = 3 (Table 1). The paired difference between K = 2 and K = 1 is 16.0 points (CI 12.7–19.5, Holm p < 0.0001). A K = 1 network is therefore a weaker controller even with full information, for example because it sees only direct neighbours when avoiding collisions.

This matters for reading the primary result. Of the 25.2-point advantage of K = 2 at 25 % observability, about 16 points also appear when the gesture is not a problem. Roughly 9 points can then be attributed to the extra reach of the token. This subtraction is a rough estimate because the two differences come from different test conditions and we did not isolate the effects experimentally. We therefore report the full difference and the capacity-controlled estimate side by side and do not attribute the whole gap to gesture relaying.

### 3.4 Comparison with the no-learning relay

The GNN beats the memoryless relay only when observability is low (Table 2). At 25 % observability the K = 2 GNN is 12.0 points better (CI 7.2–17.3, Holm p < 0.0001 for success; p = 0.011 for final error) and the K = 3 GNN is 10.0 points better (CI 5.8–14.7, p = 0.0001 for success, but p = 0.20 for final error). At K = 1 the GNN is 33.3 points better. At 50 % observability there is no clear advantage for K = 2 (+1.2 points, CI −1.3 to 4.0) or for K = 3 (+1.3, CI −0.2 to 3.3), and at K = 1 the GNN is slightly worse (−6.5 points, CI −12.8 to 0.0; Holm p = 0.22 for success and 0.027 for final error). Section 3.9 repeats this comparison against a stronger no-learning baseline.

Against relays that remember the token, the latched GNN (three training seeds) shows no difference: both reach 100 % at 25 % observability, and at 50 % the difference is +0.5 points (not significant). The relay with memory also never has a contact, whereas the latched GNN has contact in 4.0 % of episodes. A simple relay with one bit of memory per robot therefore matches the best learned policy in success and does better on contact.

### 3.5 Gesture-token ablation

The ablation (Table 4, Fig. D) uses the seed-0 networks with 80 episodes per cell, and it tests whether the policy uses the token and whether the token travels through the graph.

- **The policy needs the token.** With an all-zero token, success at 25 % observability falls to 1.2 % (K = 1), 2.5 % (K = 2), 8.8 % (K = 3) and 0 % (latched K = 2), against 70.0 %, 98.8 %, 97.5 % and 100 % with the true token.
- **The policy follows the token it receives.** When observers are shown the token of a different formation, success measured against the *shown* formation equals normal success in every model (70.0 %, 98.8 %, 97.5 % and 100 %), and success against the true formation is 0 %.
- **The token reaches robots that are not observers.** In the wrong-token mode, 89.8 % (K = 1), 99.6 % (K = 2), 99.0 % (K = 3) and 100 % (latched) of the non-observing robots end within 0.05 m of their slot in the shown formation. Non-observers have no other source for this token than their neighbours. In blank mode, 21–25 % of non-observers end near a shown-formation slot by chance in the three memoryless models, so this value is the chance level.

At 100 % observability the three tests give the same pattern: no success with a blank token, and success equal to normal success when scored against the shown formation (86.2 % for K = 1, 100 % for the others).

### 3.6 Plain imitation against DAgger

Plain behaviour cloning is clearly worse than DAgger (Table 5, Fig. F). At 25 % observability, success rises from 39.3 % to 72.3 % for K = 1, from 75.0 % to 97.5 % for K = 2 and from 94.2 % to 99.0 % for K = 3. Contact falls from 86.8 % to 51.7 % for K = 2 and from 72.7 % to 13.3 % for K = 3. The latched model succeeds in 100 % of episodes with both methods, and DAgger lowers its contact rate from 12.0 % to 4.0 %. These comparisons are descriptive and are not part of the pre-specified family of tests.

One result goes the other way. For K = 0 at 100 % observability, success is 36.3 % with plain imitation and 17.0 % with DAgger, and it is 0.0 % with both at 25 %. We have no tested explanation. A K = 0 network cannot see its neighbours, but the oracle labels depend on neighbour positions through the repulsion term, so the labels it is asked to reproduce are partly unpredictable from its inputs. 
### 3.7 Contact and final error

Contact is the main weakness of the learned policies (Table 1, Fig. E). A contact is any centre distance below 0.074 m, which is the sum of two robot radii. At 25 % observability, contact occurs in 87.3 % of K = 1 episodes, 51.7 % of K = 2 episodes, 13.3 % of K = 3 episodes and 4.0 % of latched K = 2 episodes. The rates at 50 % and 100 % observability are similar (K = 2: 55.2 % and 52.2 %). The oracle and every relay had no contact.

These contacts are light brushes. No episode of any GNN had an overlap of 5 mm or more (centre distance below 0.069 m), and the smallest centre distance over all GNN episodes was 72.5 mm. Contact for K = 2 varies between formations (37.3 % for split to 66.0 % for cluster at 25 % observability). The mean final error at 25 % observability is 25.6 mm for K = 1, 3.5 mm for K = 2, 2.0 mm for K = 3 and 1.2 mm for the latched model.

### 3.8 Differences between formations

At 25 % observability, K = 2 succeeds in 98.0 % (circle), 95.3 % (cluster), 96.7 % (line) and 100 % (split) of episodes. K = 1 is weakest on the cluster formation (54.0 %), against 73.3 % for circle, 80.7 % for line and 81.3 % for split. The memoryless K = 2 relay does worst on the split formation (60.0 %) and best on the cluster (100 %), whereas the K = 2 GNN succeeds on every split episode.

### 3.9 Exploratory: a stronger no-learning baseline

The relay of Section 3.1 lets robots without the token hold still, whereas the GNN's uninformed robots keep moving. To test whether this difference explains the GNN's advantage, we repeated the relay with a second rule (Section 2.11.1): robots without the token head for the average of the four candidate slots of their role. This analysis was added after the main results were known. Its comparisons use their own Holm family of 12 and are not part of the 16 pre-specified comparisons.

The success of this default-slot relay is 50.0 %, 81.5 %, 90.0 % and 90.0 % for K = 1, 2, 3 and 7 at 25 % observability, and 92.0 %, 99.0 %, 99.0 % and 99.0 % at 50 % observability. With K = 0 it is 0.0 % at both levels, and no episode had contact. Compared with the hold-still relay, the new rule raises success at K = 1 and 25 % observability by 11.0 points (CI 4.5–17.5, Holm p = 0.012). At K = 2 and 25 % observability it lowers success by 4.0 points (CI −7.0 to −1.0, Holm p = 0.069) and raises the final error by 5.9 mm (Holm p = 0.0011).

The GNN still beats this stronger baseline at 25 % observability: by 22.3 points at K = 1 (CI 14.2–30.2, Holm p = 0.0001), by 16.0 points at K = 2 (CI 10.5–21.7, Holm p < 0.0001) and by 9.0 points at K = 3 (CI 5.0–13.5, Holm p = 0.0002, but p = 0.21 for the final error). Of the 33.3-point advantage at K = 1 over the hold-still relay, 11.0 points are therefore explained by what the uninformed robots do and 22.3 points remain. At K = 2 the new rule makes the relay slightly worse, so the GNN's advantage grows from 12.0 to 16.0 points. At 50 % observability the GNN is 13.0 points worse than the default-slot relay at K = 1 (CI −18.7 to −7.3, Holm p = 0.0002) and indistinguishable from it at K = 2 (−0.8 points, CI −2.5 to 1.2) and K = 3 (+0.8 points, CI −0.3 to 2.5).

The advantage over the hold-still relay is not spread evenly over the formations (descriptive, 25 % observability; 150 GNN episodes per formation for K = 2 against 50 relay episodes). For K = 2 it is +40.0 points on split, +10.0 on circle, +2.7 on line and −4.7 on cluster. For K = 1 it is +63.3, +53.3, +38.7 and −22.0 points.

### 3.10 Exploratory: removing the token during the episode

This test asks whether a memoryless GNN uses its own position as a memory of the gesture (Section 2.11.2). The token is correct until step t_cut and absent afterwards. In the flag-off variant, which we use as the main evidence, the token and the flag "I know the gesture" are zero for every robot, so the network sees what an uninformed robot normally sees. Success at 25 % observability was:

| Network (episodes per cell) | Never removed | Removed at step 0 | Removed at step 300 | Removed at step 700 |
|---|---|---|---|---|
| K = 1, seed 0 (80) | 70.0 % | 17.5 % | 55.0 % | 61.3 % |
| K = 2, seeds 0 to 2 (240) | 97.1 % | 24.2 % | 71.7 % | 72.9 % |
| K = 3, seed 0 (80) | 97.5 % | 23.8 % | 73.8 % | 73.8 % |

A policy that is never told (removal at step 0) succeeds in about one episode in four, which is what guessing one of four formations would give. The three K = 2 seeds agree: 25.0 %, 23.8 % and 23.8 % at step 0, and 75.0 %, 70.0 % and 73.8 % at step 700. When the token is removed later, K = 2 still completes 71.7 % (step 300) and 72.9 % (step 700) of the episodes, three times the never-told level but well below the 97.1 % of the control.

The remaining success depends strongly on the formation. For K = 2 and removal at step 700, success is 100 % on circle, 100 % on split, 91.7 % on line and 0.0 % on cluster. The cluster robots are close to complete at the cut (mean error 3.9 mm) but end with a mean error of 431.1 mm, so they leave a formation that was nearly built. The error got worse after the cut in 25.8 % of the K = 2 episodes cut at step 700, about the share of cluster episodes (one quarter).

In the flag-on variant, where observers keep the flag while the token is zero, success falls to 2.1 % when the token is removed at step 0 and to 5.8 % at steps 300 and 700 for K = 2 (K = 1: 1.2 % at step 0 and 7.5 % at step 700; K = 3: 8.8 % and 20.0 %). This input never occurs in training, so we do not interpret it. It also qualifies Section 3.5: the blank mode there is this flag-on input, so its collapse to 0–9 % partly reflects an unfamiliar input. The flag-off test at step 0 (17.5–24.2 %) shows that a policy that is never told still ends near chance level.

---

## 4 Discussion

**What the results support.** The learned policy needs the gesture token, follows it when it is wrong, and passes it to robots that cannot see the gesture. Two message-passing layers are clearly better than one at low observability, the result holds across all three training seeds, and the paired statistics agree. Training with DAgger is essential, because plain imitation fails on states the learner itself visits.

**What they do not support.** First, a large part of the advantage of K = 2 over K = 1 is a capacity effect (Section 3.3). Second, K = 3 is better than K = 2 by only 1.5 points at 25 % observability (Holm p = 0.031 for success, 0.44 for final error), so we do not claim a benefit from a third layer. Third, a hand-designed relay with a one-bit memory reaches 100 % success with no contact, so the task itself does not require a learned policy. What the learned policy offers is a single controller trained to imitate the expert from local information only, with behaviour that the relay does not have, for example success on the split formation where the memoryless relay fails (60.0 % at K = 2).

**Why the GNN beats the memoryless relay at 25 % observability.** Two exploratory analyses (Sections 3.9 and 3.10) point to an explanation that we did not plan for. A robot's own position and role identify its slot only once the formation is known, but near its slot the position itself carries information about which formation is being built. The memoryless GNN appears to use this as a memory of the gesture. When the token is removed from every robot at step 300 or 700, K = 2 still completes 71.7 % and 72.9 % of the episodes, against 24.2 % when nobody is ever told. This memory depends on the formation. After removal at step 700 it works for circle (100 %), split (100 %) and line (91.7 %) and fails for cluster (0.0 %), where the robots leave a formation that was 3.9 mm from complete. The advantage of the GNN over the memoryless relay shows the same pattern (K = 2, 25 % observability: +40.0 points on split, +10.0 on circle, +2.7 on line and −4.7 on cluster). The explanation also fits the split formation, whose two blocks are 1.2 m apart and cannot share a token because R_comm is 0.5 m, but is still solved.

We read this as consistent evidence and not as proof. Both analyses were designed after the main results, the comparison by formation is descriptive, and the cut-off test covers one seed for K = 1 and K = 3. The consequence for the paper's claim is that part of the learned policy's advantage over the memoryless relay reflects the robots' positions acting as a memory and not better communication. The latched model and the relay with memory store the gesture explicitly and do not depend on this effect.

**Contact.** The K = 2 policy touches another robot in about half of its episodes while the expert and the relays never do. These contacts are brushes (no overlap of 5 mm or more), but the policy still depends on contact that the physical model tolerates. A third layer reduces contact to 13.3 %, and the latched model to 4.0 %, which suggests that contact is related to uncertainty about the formation and not only to the controller design. We did not test this.

**Relation to prior work.** **[To write after the reference check: cite only papers you have opened. Place the paper against GNN formation control work and DAgger.]**

---

## 5 Limitations

1. **Simulation only.** All results come from a MuJoCo model of eight e-pucks, validated against ideal kinematics to within 8 % (Table 3). There are no hardware experiments, and the model's start-up lag and friction are our own choices. A lower torque cap produced collisions in the expert controller.
2. **Eight robots and four formations.** The number of robots and the formation set are fixed, and the role assignment is fixed (robot i takes slot i). We did not test transfer to other team sizes or to unseen formations.
3. **Not a setting without global information.** Each robot knows its own world-frame position and its role. Only the choice of formation must be learned from neighbours.
4. **Training seeds.** We used three training seeds for each model, not the five we first planned. The token ablation (Section 3.5) uses the seed-0 networks, and the cut-off test (Section 3.10) uses seed 0 for K = 1 and K = 3 and seeds 0 to 2 for K = 2. The seed-to-seed spread of K = 1 (66.5 % to 75.5 % at 25 % observability) shows that single-seed results can mislead.
5. **Baselines.** The relays are hand-designed and could be improved. The first lets robots without the token hold still. The second, stronger relay (Section 3.9) sends them to the average slot of their role. K = 0 lacks neighbour positions as well as the gesture, so it is not a clean no-communication baseline.
6. **Start box.** Episodes start in a 1.0 m box so that the graph is usually connected. A wider start box was not evaluated, although the protocol keeps it as a robustness test. 
7. **No noise or packet loss.** Positions and messages are exact, and communication is instantaneous within R_comm.
8. **Contact.** The K = 2 policy has contact in about half of its episodes.
9. **Multiple comparisons.** The 16 pre-specified comparisons are corrected with Holm's method, but the contact, formation and ablation results are descriptive.
10. **Exploratory analyses.** The default-slot relay (Section 3.9) and the token cut-off test (Section 3.10) were designed after the main results were known. Their comparisons use a separate Holm family of 12, and their findings should be read as hypotheses to be confirmed.
