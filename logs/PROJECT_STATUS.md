# Swarm GNN paper - project status (updated 3 Nov 2026)

## Who / where
- Repo: https://github.com/sahilpanchavishe/swarm-paper (Days 1-3 built and pushed from Sahil's Mac)
- Resuming: Yashvardhan, own Mac, from the repo. Last pushed commit: 55699aa "Day 3: baselines"
- Target journals: Archive of Mechanical Engineering (first choice) or World Journal of Engineering. No old code, data or rejection letter exist; everything is rebuilt from zero.

## Decisions made (do not reopen without a reason)
1. Platform = MuJoCo on Mac, headless, own e-puck model (src/mj_env.py). Webots dropped.
2. Horizon 1100 steps (35.2 s). Old 17.6 s was too short even for the expert.
3. Start box = [-0.5, 0.5]^2 (1.0 m). The old 1.8 m box connects only ~1% of starts at R_comm = 0.5 m. The wide box stays as a robustness test.
4. Fixed role assignment (robot i -> slot i). Success = mean final error < 0.025 m (also report < 0.05 m).
5. Statistics fixed in advance: paired tests, Holm correction, bootstrap CIs, no "significant" without the corrected result.
6. Rule: a changed script gets a NEW day-prefixed name (d4_*.py). Never overwrite swarm_sim.py or mj_env.py.
7. Rule: every table and figure comes from a script reading raw CSVs. The old paper's numbers are not reused.

## Progress
- Day 1 Setup + expert: DONE
- Day 2 Validation, connectivity, sensitivity: DONE (baseline gate PASS; torque 0.002 gave 15 collisions, report as limitation)
- Day 3 No-learning baselines: DONE (results/d3_baselines.csv, 3,800 episodes)
- Day 4 Data generation + GNN training: NOT STARTED
- Day 5 Gate G2: GNN converges at 100% observability (if not, paper becomes a benchmark / negative-result study)
- Day 6 Main experiment, protocol frozen
- Day 7 Statistics, ablations, robustness
- Day 8 Write paper from results
- Day 9 Verify every reference by hand
- Day 10 Final review, journal checklist

## Day 3 results (200 episodes per cell)
Oracle (100% obs): 100% success, 0 collisions. K0: 0% at 25% and 50% observability.
Memoryless relaying at 25% obs: K1 39.0%, K2 85.5%, K3 89.0%, K7 89.5%. At 50% obs: K1 85.5%, K2 97.0%, K3 98.5%, K7 98.5%.
With memory (+mem), every K >= 1 reaches 99.5-100%. No collisions anywhere.
Meaning: a no-learning relay already solves the task, so the paper must justify the GNN.

## Open issues
- Day 4 decision: main GNN is memoryless; add one "latched token" variant as an ablation (pending OK).
- Reviewer risks: absolute positions and one-hot ID in the node vector; no relative neighbour positions; need 5 training seeds per K; split formation disconnected at R_comm = 0.5 m; references unverified.
- Repo is public and contains the old manuscript docx. Consider making it private.
