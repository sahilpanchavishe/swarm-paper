# Methods (DRAFT 1, written from protocol.md and the code in src/)

Items marked **[CHECK]** are things I could not confirm from the repo. Look at the code or decide, then delete the tag.
Numbers in curly braces {like this} come from the result tables and must be copied from paper/tables/ after the extra seeds are evaluated. Do not type them from memory.

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

An episode starts with robot positions drawn uniformly from [−0.5, 0.5]² (a 1.0 m start box) with a minimum separation of 0.15 m and random headings. We use this box because with the wider 1.8 m box only about 1 % of starts form a connected graph at R_comm = 0.5 m (connectivity analysis: results/connectivity_vs_range.csv). The horizon is T = 1100 control steps (35.2 s); the expert controller needs up to 16.8 s, so an earlier horizon of 17.6 s was too short.

The observer set contains round(ρ · 8) robots chosen uniformly at random from the episode seed, for observability ρ ∈ {100 %, 50 %, 25 %}, that is 8, 4 or 2 observers. The same seed produces the same start poses and the same observers for every policy, so all comparisons are paired.

Seeds are split by purpose: 0–1199 for demonstrations, 1300–4299 for DAgger roll-outs, 5000 and above for development checks during training, and 10000–10049 for the final test. The test seeds were used for evaluation only.

## 2.5 Metrics

- **Formation error**: the mean over robots of ‖p_i − slot_i‖ at the final step (m).
- **Success**: formation error below 0.025 m at the final step (we also report a looser threshold of 0.05 m).
- **Contact**: any pair of robots with centre distance below 0.074 m (two robot radii) during the episode. We also report **overlap**, centre distance below 0.069 m (5 mm or more of interpenetration).

## 2.6 Expert controller (oracle)

The expert is a potential-field controller with every robot informed of the formation. For robot i with position p_i and slot s_i the desired velocity is

u_i = k_att (s_i − p_i) + Σ_j f_rep(d_ij) d̂_ij + k_tan Σ_j f_rep(d_ij) d̂_ij^⊥

where d_ij = ‖p_i − p_j‖, d̂_ij = (p_i − p_j)/d_ij, d̂^⊥ is d̂ rotated by 90°, and f_rep(d) = k_rep (1/d − 1/r_inf) / d² for d < r_inf and 0 otherwise. We use k_att = 2.8 s⁻¹, k_rep = 5·10⁻³, r_inf = 0.12 m and k_tan = 0.5. The potential-field "force" is treated directly as a velocity command because the model is kinematic at the policy level, so k_att has units of s⁻¹ and no force units are used. The expert reaches the 0.025 m success threshold in {100 %} of the test episodes with no contact (Table 1, row "oracle").

## 2.7 No-learning baselines (relays)

To show what a policy must add beyond simple token passing, we evaluate relay controllers that use the same expert. The token is relayed K hops through the communication graph at every step; robots that have it run the expert, and robots that do not have it hold still but still avoid collisions. We report K ∈ {0, 1, 2, 3, 7} (K = 7 is the largest possible hop count for eight robots). The *memoryless* relay forgets the token as soon as no observer is within K hops. The *+mem* variant keeps the token once received. K = 0 means only observers move. These baselines are deliberately simple, and results depend on that design.

## 2.8 GNN policy

**Inputs per robot.** Position divided by 1.1, a one-hot role vector (8 entries), the gesture token (a 4-entry one-hot formation vector, multiplied by a "knows the gesture" flag) and the flag itself. For observers the flag is 1; for all others it is 0, except in the latched variant below.

**Architecture.** A two-layer encoder (hidden size 64, ReLU) gives each robot a hidden state. Then K message-passing layers follow. In each layer, a message from robot j to robot i is computed by an MLP from j's hidden state and the relative position (p_j − p_i)/R_comm, only along graph edges. Messages are summed, divided by 4, and passed with the robot's own hidden state through an update MLP; the update is added residually. A head MLP takes the final hidden state, the robot's position and its role vector and outputs a normalised velocity in [−1, 1]², multiplied by v_max and converted by the heading controller of Section 2.2. With K = 0 there are no message layers, so the network sees only its own inputs.

**Latched variant (ablation).** A robot additionally receives a 1-bit latch that is set when the gesture has reached it through K-hop flooding and stays set afterwards. This gives the policy memory of the gesture and tests how much of the gap to the oracle is due to forgetting.

**Training.** The training data are 1200 expert demonstrations (300 per formation). Demonstrations are driven by a mix of behaviours (episode index mod 4: oracle, and memoryless K-hop relay with K = 1, 2 and 3) so the dataset also covers states reached by robots that do not yet know the gesture. The *labels* are always the oracle expert's velocity with every robot informed, so the network learns to imitate "everyone informed" behaviour. States are stored every fourth control step. We split by episode (90 % train, 10 % validation). Behaviour cloning uses the smooth-L1 loss (β = 0.1), Adam, learning rate 2·10⁻³ with a one-cycle schedule, batch size 256 and 30 epochs.

Behaviour cloning alone drifts into states the expert never visited. We therefore add **DAgger**: for three rounds, the current policy is rolled out on 240 training episodes, every visited state is relabelled by the oracle expert and added to the dataset, and the network is fine-tuned for 10 epochs at a learning rate of 5·10⁻⁴. Training seeds 0, 1 and 2 are used for each K = 0 to 3. The latched variant was trained with seed 0 only.

## 2.9 Evaluation and statistics

Every policy is tested on seeds 10000–10049 × 4 formations × 3 observability levels, that is 600 episodes per trained model and 200 episodes per observability level. Where several training seeds exist, each test episode's outcome is first averaged over training seeds, so the test episode remains the unit of analysis, and the spread across training seeds is reported separately.

Success rates are reported with 95 % Wilson confidence intervals. Comparisons are paired by (formation, seed, observability). For each comparison we report the mean paired difference in success with a 95 % bootstrap confidence interval (5000 resamples of test episodes) and test the final error with a two-sided Wilcoxon signed-rank test. All {16} comparisons in the pre-specified family are corrected with the Holm method. The primary hypothesis, fixed before any experiment, is that the K = 2 GNN has a higher success rate than the K = 1 GNN at 25 % observability. A result is called significant only if its Holm-corrected p is below 0.05; otherwise we say there is no evidence of a difference and report the width of the interval.

## 2.10 Token ablation

To test whether the policy actually uses the gesture and whether it travels through the graph, we re-run the same episodes in three modes for the K = 1, 2, 3 and latched K = 2 models: the true token goes to the observers (control), an all-zero token goes to the observers (blank), or the observers receive the token of a different formation (wrong). For the wrong-token mode we score success against both the true and the shown formation, and we measure the share of *non-observers* that end within 0.05 m of their slot in the shown formation. Non-observers only receive the token by message passing, so a high share means the token reached them through the graph and they acted on it. We use 20 test seeds × 4 formations per cell {n = 80} at 100 % and 25 % observability.
