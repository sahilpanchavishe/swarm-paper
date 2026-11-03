# Day 1 (Mac, MuJoCo) - updated

Files in src/: swarm_sim.py, mj_env.py (UPDATED, replaces your old copy), test_mj_env.py, mj_expert_test.py, expert_check.py (NEW)

1. cd ~/Developer/swarm-paper && source .venv/bin/activate
2. Copy the files above into src/ and cd src
3. python test_mj_env.py        # model vs ideal unicycle
4. python expert_check.py 30    # expert in MuJoCo: success, collisions, time to converge (about 2-4 minutes)
Expected: 100% success, 0 collisions, final error about 0.001 m, convergence 11-17 s.
Send me both outputs in full.

What changed in mj_env.py: chassis lowered to 1.5 mm above the floor (robots were climbing onto each other),
wheels collide only with the floor, chassis only with chassis and walls, wheel torque capped at 0.004 N*m
(was effectively ~0.3 N*m), contacts stiffer.
