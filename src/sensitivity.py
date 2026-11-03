"""Day 2b: does the expert still work if our unknown physics parameters are different?
Varies wheel torque cap (0.002, 0.004, 0.008 N*m) and chassis friction (0.1, 0.3, 0.6), one at a time,
over all 4 formations at the protocol horizon (1100 steps = 35.2 s).
Run: python sensitivity.py [n_seeds_per_formation] [workers] [start_half]   (defaults: 30 seeds, cores-1, 0.5 m)
Writes ../results/sensitivity.csv (one row per episode) and prints a summary."""
import sys, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
import swarm_sim as S
from mj_env import SwarmEnv, ROBOT_R

ROOT = Path(__file__).resolve().parent.parent
T_STEPS = 1100
START_HALF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5   # start region half-width (m); protocol v2 = 0.5
CONFIGS = [("baseline", 0.004, 0.3), ("tau_low", 0.002, 0.3), ("tau_high", 0.008, 0.3),
           ("mu_low", 0.004, 0.1), ("mu_high", 0.004, 0.6)]
_env = {}

def get_env(tau, mu):
    key = (tau, mu)
    if key not in _env: _env[key] = SwarmEnv(tau_max=tau, chassis_mu=mu)
    return _env[key]

def one(args):
    name, tau, mu, f, seed = args
    env = get_env(tau, mu)
    rng = np.random.default_rng(seed); p, th = S.random_start(rng, half=START_HALF); tg = S.slots(f)
    env.reset(p, th); p, th = env.poses(); errs = []; minc = 9.0
    for _ in range(T_STEPS):
        v, w = S.to_unicycle(S.expert_velocity_vector(p, tg), th); p, th = env.step(v, w)
        errs.append(S.formation_error(p, tg)); minc = min(minc, S.min_clearance(p))
    errs = np.array(errs); bad = np.where(errs >= 0.025)[0]
    return dict(config=name, tau_max=tau, chassis_mu=mu, formation=f, seed=seed, final_err=errs[-1],
                success=bool(errs[-1] < 0.025), min_clearance=minc, collision=bool(minc < 2 * ROBOT_R - 1e-9),
                t_converge_s=(bad[-1] + 1 if len(bad) else 0) * S.DT if errs[-1] < 0.025 else np.nan)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 1)
    jobs = [(nm, tau, mu, f, s) for nm, tau, mu in CONFIGS for f in S.FORMATIONS for s in range(n)]
    print(f"{len(jobs)} episodes on {workers} workers, start half-width {START_HALF} m ...", flush=True)
    with Pool(workers) as pool:
        rows = pool.map(one, jobs, chunksize=4)
    df = pd.DataFrame(rows); (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "sensitivity.csv", index=False)
    g = df.groupby("config", sort=False).agg(tau_max=("tau_max", "first"), chassis_mu=("chassis_mu", "first"),
                                              success_pct=("success", lambda x: 100 * x.mean()),
                                              collisions=("collision", "sum"), mean_final_err=("final_err", "mean"),
                                              median_t_conv=("t_converge_s", "median"), max_t_conv=("t_converge_s", "max"))
    pd.set_option("display.width", 200); print(g.round(4).to_string())
    base = g.loc["baseline"]
    print("\nGATE (baseline): " + ("PASS" if base.success_pct >= 95 and base.collisions == 0 else "FAIL") + " - baseline success >= 95% and 0 collisions")
    print("SENSITIVITY: " + ("all configs >= 95% success" if (g.success_pct >= 95).all() else "some config below 95% success") +
          f"; configs with collisions: {list(g.index[g.collisions > 0]) or 'none'} (report these in the paper, do not hide them)")
