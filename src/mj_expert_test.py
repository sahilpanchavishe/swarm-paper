"""Run the potential-field expert inside MuJoCo physics. SMOKE TEST (not a paper result)."""
import numpy as np, sys
import swarm_sim as S
from mj_env import SwarmEnv, N, ROBOT_R
env = SwarmEnv()
def run(formation, seed, T=550):
    rng = np.random.default_rng(seed)
    p, th = S.random_start(rng); tg = S.slots(formation)
    env.reset(p, th); p, th = env.poses(); minc = 9
    for _ in range(T):
        v, w = S.to_unicycle(S.expert_velocity_vector(p, tg), th)
        p, th = env.step(v, w)
        minc = min(minc, S.min_clearance(p))
    return S.formation_error(p, tg), minc
for f in S.FORMATIONS:
    r = np.array([run(f, s) for s in range(10)])
    print(f"{f:8s} final_err mean={r[:,0].mean():.3f} max={r[:,0].max():.3f} | worst clearance={r[:,1].min():.3f} (contact at {2*ROBOT_R:.3f})")
