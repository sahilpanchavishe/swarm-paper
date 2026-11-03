"""Expert reliability check inside MuJoCo. Run: python expert_check.py [n_seeds]
Success = mean slot error < 0.025 m at end. Collision = centre distance < 2*ROBOT_R (0.074 m).
Convergence time = first step after which mean error stays < 0.025 m."""
import sys, numpy as np, swarm_sim as S
from mj_env import SwarmEnv, ROBOT_R
n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
env = SwarmEnv()
def run(f, seed, T=550):
    rng = np.random.default_rng(seed); p, th = S.random_start(rng); tg = S.slots(f)
    env.reset(p, th); p, th = env.poses(); errs = []; minc = 9
    for _ in range(T):
        v, w = S.to_unicycle(S.expert_velocity_vector(p, tg), th); p, th = env.step(v, w)
        errs.append(S.formation_error(p, tg)); minc = min(minc, S.min_clearance(p))
    errs = np.array(errs); bad = np.where(errs >= 0.025)[0]
    t_conv = (bad[-1] + 1 if len(bad) else 0) * S.DT
    return errs[-1], minc, t_conv
for f in S.FORMATIONS:
    r = np.array([run(f, s) for s in range(n)])
    ok = r[:, 0] < 0.025
    print(f"{f:8s} success {ok.mean()*100:5.1f}% | final_err mean {r[:,0].mean():.4f} max {r[:,0].max():.4f} | "
          f"collisions (clear<0.074) {(r[:,1]<2*ROBOT_R-1e-9).sum()}/{n} | t_conv median {np.median(r[ok,2]) if ok.any() else float('nan'):.1f}s max {r[ok,2].max() if ok.any() else float('nan'):.1f}s")
