# Methods (DRAFT 2, updated 7 Nov 2026 for the Day 12 / Day 14 results; AI-assisted draft, to be reviewed and rewritten by the authors)

The constants in this section are checked against the code by `src/d9_check_numbers.py`. Sections 2.11 and 2.12 describe the capacity controls and the safety filter. Section 2.13 describes the exploratory analyses, which were added after the protocol was frozen.

## 2.1 Problem setting

Eight differential-drive robots must assemble one of four target formations (circle, line, cluster, split). A human gesture selects the formation. Only a subset of the robots, the *observers*, sees the gesture; the other robots can receive it only from neighbours through local communication. We study how the amount of information exchange (the number of message-passing layers K) and the fraction of observers affect formation success.

**Information model.** Every robot knows its own world-frame position and its fixed role index (robot i is assigned slot i of the chosen formation). The slot coordinates are expressed in the world frame, so a robot needs its own position and role to know where its slot is. The setting is therefore *not* free of global information, and we do not claim that it is. What a non-observing robot does not know is *which* formation was selected. That one piece of information reaches it only through the communication graph.

**Communication graph.** Two robots are connected when their centre distance is at most R_comm = 0.50 m. The graph is undirected and is recomputed from the current positions at every control step. Messages carry the sender's hidden state and its relative position.

## 2.2 Robot and simulation model

All experiments run in MuJoCo (headless, version 3.17.0) on an arena of 2.2 m × 2.2 m with boundary walls. No official e-puck model exists for MuJoCo, so we built one from the published e-puck dimensions: wheel radius 0.0205 m, axle length 0.052 m, robot radius 0.037 m, and a wheel speed limit of 6.28 rad/s, which gives a maximum forward speed v_max = 0.1287 m/s. Each wheel is driven by a velocity actuator (gain 0.05) with a torque cap of 0.004 N·m. The chassis has mass 0.15 kg and friction coefficient 0.3 against other chassis; the wheels have friction coefficient 1.5 and the floor 1.0. Physics runs at 0.004 s per step and the controller at 0.032 s (8 physics steps per control step).

Policies output a desired planar velocity u. It is converted to a forward speed v and a turn rate ω by a heading controller: ω = k_ω · e_θ, where e_θ is the angle between the heading and the desired direction, and v = min(‖u‖, v_max) · clip(cos e_θ, 0, 1). The pair is then scaled so that both wheel speeds stay within the 6.28 rad/s limit. The heading gain is k_ω = 4.0 s⁻¹.

**Model validation.** We compared the simulated robot with ideal unicycle kinematics in three tests (Table 3): driving straight at 0.10 m/s for 3 s, turning in place at 2 rad/s for 2 s, and an arc at 0.08 m/s with ω = 0.5 rad/s for 4 s. The simulated path lengths and yaw angles differ from the ideal ones by −3.5 % (straight path), −3.2 % (spin yaw), −2.3 % (arc path) and −8.0 % (arc yaw). The shortfall comes mainly from motor start-up lag. Torque and motor-gain settings were fixed after this check; a lower torque cap produced collisions in the expert controller and is reported as a limitation.

## 2.3 Formations

Slot i belongs to robot i (fixed role assignment). All coordinates are in metres, centred on the arena origin.

- circle: radius 0.45, eight slots equally spaced
- line: eight slots on the x axis with spacing 0.20
- cluster: 2 × 4 grid with spacing 0.15
- split: two 2 × 2 blocks with spacing 0.15, centred at x = −0.6 and x = +0.6

## 2.4 Episodes, observers and seeds

An episode starts with robot positions drawn uniformly from [−0.5, 0.5]² (a 1.0 m start box) with a minimum separation of 0.15 m and random headings. We use this box because with the wider 1.8 m box only about 1 % of starts form a connected graph at R_comm = 0.5 m (connectivity analysis: results/connectivity_vs_range.csv). The wide box is used only as an exploratory stress test (Section 2.13). The horizon is T = 1100 control steps (35.2 s); the expert controller needs up to 16.8 s, so an earlier horizon of 17.6 s was too short.

The observer set contains round(ρ · 8) robots chosen uniformly at random from the episode seed, for observability ρ ∈ {100 %, 50 %, 25 %}, that is 8, 4 or 2 observers. The same seed produces the same start poses and the same observers for every policy, so all comparisons are paired.

Seeds are split by purpose: 0–1199 for demonstrations, 1300–4299 for DAgger roll-outs, 5000–5009 for the development episodes used only to choose the safety-filter parameters, 10000–10049 for the first test block (reused in exploratory work, and therefore treated as exploratory), 20000–20049 for the confirmatory block, and 30000 and above for the exploratory stress tests. Every headline claim rests on the confirmatory block, on which each model is evaluated once and nothing is tuned.

## 2.5 Metrics

- **Formation error**: the mean over robots of ‖p_i − slot_i‖ at the final step (m).
- **Success**: formation error below 0.025 m at the final step (we also report a looser threshold of 0.05 m).
- **Contact**: any pair of robots with centre distance below 0.074 m (two robot radii) during the episode. We also report **overlap**, centre distance below 0.069 m (5 mm or more of interpenetration).
- **Collision-free success**: success with no contact during the episode.
- **Maximum per-robot error**: the largest distance from any robot to its slot at the final step, reported next to the mean because a mean can hide one badly misplaced robot.

## 2.6 Expert controller (oracle)

The expert is a potential-field controller with every robot informed of the formation. For robot i with position p_i and slot s_i the desired velocity is

u_i = k_att (s_i − p_i) + Σ_j f_rep(d_ij) d̂_ij + k_tan Σ_j f_rep(d_ij) d̂_ij^⊥

where d_ij = ‖p_i − p_j‖, d̂_ij = (p_i − p_j)/d_ij, d̂^⊥ is d̂ rotated by 90°, and f_rep(d) = k_rep (1/d − 1/r_inf) / d² for d < r_inf and 0 otherwise. We use k_att = 2.8 s⁻¹, k_rep = 5·10⁻³, r_inf = 0.12 m and k_tan = 0.5. The potential-field "force" is treated directly as a velocity command because the model is kinematic at the policy level, so k_att has units of s⁻¹ and no force units are used. The expert reaches the 0.025 m success threshold in 100 % of the test episodes with no contact (Table 1, row "oracle").

## 2.7 No-learning baselines (relays)

To show what a policy must add beyond simple token passing, we evaluate relay controllers that use the same expert. The token is relayed K hops through the communication graph at every step; robots that have it run the expert, and robots that do not have it hold still but still avoid collisions. We report K ∈ {0, 1, 2, 3, 7} (K = 7 is the largest possible hop count for eight robots). The *memoryless* relay forgets the token as soon as no observer is within K hops. The *+mem* variant keeps the token once received. K = 0 means only observers move. These baselines are deliberately simple, and results depend on that design. Two further relays are used in the later experiments: *relayD*, in which a robot without the token heads for the default slot of its role (Section 2.13), and a *version-aware* relay for the command-change test, in which every robot keeps a register of the latest command it has received and the register is overwritten only by a newer command. Both relays are given the formation geometry, which the GNN has to learn from demonstrations.

## 2.8 GNN policy

**Inputs per robot.** Position divided by 1.1, a one-hot role vector (8 entries), the gesture token (a 4-entry one-hot formation vector, multiplied by a "knows the gesture" flag) and the flag itself. For observers the flag is 1; for all others it is 0, except in the latched variant below.

**Architecture.** A two-layer encoder (hidden size 64, ReLU) gives each robot a hidden state. Then K message-passing layers follow. In each layer, a message from robot j to robot i is computed by an MLP from j's hidden state and the relative position (p_j − p_i)/R_comm, only along graph edges. Messages are summed, divided by 4, and passed with the robot's own hidden state through an update MLP; the update is added residually. A head MLP takes the final hidden state, the robot's position and its role vector and outputs a normalised velocity in [−1, 1]², multiplied by v_max and converted by the heading controller of Section 2.2. With K = 0 there are no message layers, so the network sees only its own inputs.

**Latched variant (ablation).** A robot additionally receives a 1-bit latch that is set when the gesture has reached it through K-hop flooding and stays set afterwards. This gives the policy memory of the gesture and tests how much of the gap to the oracle is due to forgetting.

**Training.** The training data are 1200 expert demonstrations (300 per formation). Demonstrations are driven by a mix of behaviours (episode index mod 4: oracle, and memoryless K-hop relay with K = 1, 2 and 3) so the dataset also covers states reached by robots that do not yet know the gesture. The *labels* are always the oracle expert's velocity with every robot informed, so the network learns to imitate "everyone informed" behaviour. States are stored every fourth control step. We split by episode (90 % train, 10 % validation). Behaviour cloning uses the smooth-L1 loss (β = 0.1), Adam, learning rate 2·10⁻³ with a one-cycle schedule, batch size 256 and 30 epochs.

Behaviour cloning alone drifts into states the expert never visited. We therefore add **DAgger**: for three rounds, the current policy is rolled out on 240 training episodes, every visited state is relabelled by the oracle expert and added to the dataset, and the network is fine-tuned for 10 epochs at a learning rate of 5·10⁻⁴. The memoryless models K = 1, 2 and 3 are trained with seeds 0 to 4; K = 0, the latched variant and the capacity controls (Section 2.11) with seeds 0 to 2. Seeds 3 and 4 use the same recipe as seeds 0 to 2.

## 2.9 Evaluation and statistics

Every policy is evaluated on the confirmatory seeds 20000–20049 × 4 formations × 3 observability levels, that is 200 episodes per observability level for each trained model. The models K = 1, 2 and 3 have five training seeds (1000 episodes per level); K = 0, the latched variant and the capacity controls have three (600). Relays and the oracle are deterministic and have one run of 200 episodes per level. Each model is evaluated once on this block, and nothing is tuned on it.

Success rates are reported with 95 % two-level bootstrap confidence intervals (5000 resamples in which both the test scenarios (formation, seed) and the training seeds are resampled). Comparisons are paired by (formation, seed, observability). Because success is a binary outcome, it is tested with the exact McNemar test, and the final error with a two-sided Wilcoxon signed-rank test. Each test is run separately for each training seed: a GNN trained with seed s is compared with the other GNN trained with seed s, and a comparison with a relay uses every training seed against the single relay run. A comparison is called significant only if it is significant in every training seed, so its p-value is the largest of the per-seed values. For each comparison we report the mean paired difference in success with a 95 % two-level bootstrap interval and the per-seed differences, because a small number of training seeds makes the seed level of the interval coarse. When both models have several training seeds, the pooled difference and its interval use only the training seeds that the two models share. A result is called significant only if its Holm-corrected p is below 0.05 and the per-seed differences all point the same way; otherwise we say there is no evidence of a difference and report the width of the interval.

Holm correction is applied within each family of comparisons, never across families. The confirmatory success family has 28 comparisons: the 16 pre-specified comparisons of the protocol and the 12 capacity-control comparisons that the Day 12 addendum specified before the confirmatory run. The collision-free-success family under the safety filter has 22 comparisons. The filter, its parameters and its selection rule were fixed before the final run, but this set of comparisons was specified after the confirmatory success results had been seen. The exploratory comparisons (Section 2.13) are corrected in their own families and are descriptive. The primary hypothesis, fixed before any experiment, is that the K = 2 GNN has a higher success rate than the K = 1 GNN at 25 % observability. The per-seed McNemar procedure replaced a first analysis that tested seed-averaged results with a Wilcoxon test; we made this change after the first analysis, because success is binary and averaging over training seeds hides their variation.

## 2.10 Token ablation

To test whether the policy actually uses the gesture and whether it travels through the graph, we re-run the same episodes in three modes for the K = 1, 2, 3 and latched K = 2 models: the true token goes to the observers (control), an all-zero token goes to the observers (blank), or the observers receive the token of a different formation (wrong). For the wrong-token mode we score success against both the true and the shown formation, and we measure the share of *non-observers* that end within 0.05 m of their slot in the shown formation. Non-observers only receive the token by message passing, so a high share means the token reached them through the graph and they acted on it. We use 20 test seeds × 4 formations per cell n = 80 at 100 % and 25 % observability. This analysis was run before the confirmatory block and is not part of it.

## 2.11 Capacity controls

The K = 2 network has more parameters than the K = 1 network (64,898 against 44,034). To separate communication depth from parameter count we trained three controls with the same recipe and the same training seeds (0 to 2). The first, *wide1*, is a K = 1 network of hidden width 78 with 64,586 parameters, close to the count of K = 2. The second, *shared2*, is a K = 2 network that uses one set of weights in both message-passing layers and has 44,034 parameters, the same as K = 1. The third, *mlpwide*, is a K = 0 network of width 110 with 65,122 parameters and no message passing. If depth rather than size matters, wide1 should stay well below K = 2 and shared2 should match it. The comparisons are the 12 capacity comparisons of the confirmatory family (Section 2.9).

## 2.12 Safety filter

Contact is not penalised in the imitation loss, so we add a decentralised pairwise velocity filter after the policy output. For a pair of robots within 0.30 m, the barrier is h = d² − dsafe², where d is the centre distance, and each robot enforces half of the condition dh/dt ≥ −γh by projecting its commanded velocity. The filter is empirical and carries no formal guarantee. We chose (dsafe, γ) on the development seeds 5000–5009 as the setting with the highest mean collision-free success of the K = 2 and K = 3 GNNs at 25 % observability, taking the larger dsafe if two settings were within one point. Two settings, (0.10 m, 10 s⁻¹) and (0.10 m, 20 s⁻¹), tied at 97.5 %. Before the final run we fixed a tie-break in the protocol addendum, the lower mean number of near-contact steps on the development seeds, which selected dsafe = 0.10 m and γ = 20 s⁻¹. The filter was then evaluated once on the confirmatory block, with the same episodes as the unfiltered runs.

## 2.13 Exploratory analyses (added after the protocol was frozen)

The analyses in this section were designed after the first results were known, or after the protocol was frozen. They are not part of the confirmatory block, their statistics form separate correction families, and the paper reports them as exploratory.

**2.13.1 Default-slot relay.** The relay of Section 2.7 keeps robots without the token still. In a second relay, such a robot heads for the average of the four candidate slots of its role (the mean over the four formations of the slot belonging to that role), while robots with the token go to their true slot as before. The relay is memoryless, uses K ∈ {0, 1, 2, 3, 7} hops, and is evaluated at 50 % and 25 % observability on the same test seeds, start poses and observers as every other policy (50 seeds × 4 formations = 200 episodes per cell). Comparisons with the GNN and with the hold-still relay are paired as in Section 2.9 (Wilcoxon signed-rank test on the final error, bootstrap confidence intervals) with their own Holm correction over the 12 comparisons of this analysis.

**2.13.2 Token cut-off test.** To test whether a memoryless GNN uses its own position as a memory of the gesture, the token is correct until control step t_cut and removed afterwards, at 25 % observability. The cut-off steps are t_cut = 0 (the policy is never told), 300 (9.6 s), 700 (22.4 s) and 1100 (never removed, the control). Two variants of the removal were run. In the *flag-off* variant, the token and the flag "I know the gesture" are set to zero for every robot, which is the input that an uninformed robot normally sees. In the *flag-on* variant, observers keep the flag while the token is zero, which is the same input as the blank mode of Section 2.10. That input does not occur in training, so we use the flag-off variant as the main evidence. The networks are K = 1 and K = 3 (seed 0) and K = 2 (seeds 0, 1 and 2), with 20 test seeds × 4 formations = 80 episodes per network and cell. We report success against the true formation, the formation error at the cut and at the end, and whether the error increased after the cut.

**2.13.3 Link loss, delay and noise.** On seeds 30000 and above, at 25 % and 50 % observability and without the filter, we perturb one factor at a time: each communication link is removed independently with probability 0.1, 0.3 or 0.5 per control step; the controller sees positions (its own and its neighbours') that are 5 or 10 control steps old, while the radio graph uses the true positions; or zero-mean Gaussian noise with a standard deviation of 5 or 10 mm is added to every position the controller sees. The error and contact metrics always use the true positions.

**2.13.4 Wide start box.** The start box is widened to 1.8 m, where the initial communication graph is usually disconnected (initial components are recorded). We compare the GNNs with hold-still relays with memory and with relayD, which sends a robot without the token to the default slot of its role (Section 2.13.1) and, in the +mem variant, keeps the latest token it has received. The unfiltered runs were followed by a filtered run with the parameters of Section 2.12 (seeds 30000–30024). The filtered run is a post-hoc follow-up, motivated by the contact gap seen in the unfiltered results and recorded in addendum 5b after the run had completed; nothing was tuned on it.

**2.13.5 Mid-episode command change.** The commanded formation changes at 17.6 s and the horizon is 52.8 s. Success means reaching the new formation. Policies and relays hold an explicit per-robot register of the latest command. In the *naive* variant a stale command is never overwritten; in the *version-aware* variant it is overwritten by a newer one.
