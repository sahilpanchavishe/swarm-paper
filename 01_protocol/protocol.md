# Experiment protocol (frozen on Day 1; changes after Day 6 require re-running everything)

## Task
8 differential-drive e-puck robots in a 2.2 m x 2.2 m arena (MuJoCo, own e-puck model) assemble one of four formations.
A human gesture selects the formation. Only observers see the gesture; others must learn it from neighbours.

## Fixed constants (from swarm_sim.py / mj_env.py)
- N = 8, wheel radius 0.0205 m, axle 0.052 m, wheel speed limit 6.28 rad/s (v_max = 0.1287 m/s)
- Control step 0.032 s, physics step 0.004 s, wheel torque cap 0.004 N*m, R_comm = 0.50 m
- Robot radius 0.037 m. Collision = centre distance < 0.074 m.

## Formations (slot i belongs to robot i)
- circle: radius 0.45 m, 8 slots equally spaced
- line: 8 slots on the x axis, spacing 0.20 m
- cluster: 2 x 4 grid, spacing 0.15 m, centred at origin
- split: two 2 x 2 blocks, spacing 0.15 m, centred at x = -0.6 m and x = +0.6 m

## Episode
- Start: random positions in [-0.5, 0.5]^2 (1.0 m start box; the 1.8 m box is kept as a robustness test), minimum separation 0.15 m, random headings
- Horizon T = 1100 steps (35.2 s). The expert needs up to 16.8 s, so the old 17.6 s horizon was too tight.
- Episode seed s is shared by every policy (paired design). Training seeds 0-9999, test seeds 10000+.
- Observers: round(obs x 8) robots, uniformly at random from the seed. obs in {100%, 50%, 25%} = 8, 4, 2 observers.

## Metrics
- Formation error = mean over robots of ||p_i - slot_i|| (m), at the final step
- Success = formation error < 0.025 m at the final step (also report < 0.05 m)
- Time to converge = first step after which the error stays below 0.025 m
- Collisions = number of episodes with any centre distance < 0.074 m

## Policies compared
Oracle (global token) | flooding-K for K in {0,1,2,3,inf} (token relayed K hops per step, then the expert) | GNN K in {0,1,2,3}, 5 training seeds each

## Statistics (decided before any experiment)
- Primary hypothesis: GNN K=2 has higher success than GNN K=1 at 25% observability.
- Unit of analysis: the test episode (paired across policies). Training-seed variation is reported as a spread.
- Paired Wilcoxon signed-rank on final error; McNemar on success; Holm correction across all comparisons.
- Effect sizes with 95% bootstrap confidence intervals. Two-sided tests only.
- No claim of "consistent" or "significant" without the corrected result.

## Day 6 additions (frozen)
- Main test: seeds 10000-10049 x 4 formations x observability 100/50/25 % (600 episodes per model), same seeds as Day 3 baselines. Day 5 final-test seeds 10000-10009 are inside this range; they were evaluated once and never used for tuning.
- Contact = centre distance < 0.074 m. Overlap = centre distance < 0.069 m (5 mm or more). Both reported.
- Paired by (formation, seed, observability). Wilcoxon signed-rank, Holm correction over the 16-comparison family in d6_analysis.py, bootstrap 95% CI. Primary: GNN K=2 vs K=1 at 25% observability.
- Training seeds: 3 per K (seeds 0-2), NOT 5, because of compute time. State this as a limitation.
