"""Day 3: no-learning baselines. Does the problem need a GNN at all?
Policies (protocol v2: 1.0 m start box, T = 1100 steps, test seeds 10000+):
  oracle      every robot knows the gesture (centralized knowledge, the ceiling)
  K0          only observers know the gesture; everyone else holds still (no communication)
  K<k>        token flooding: every control step the token is relayed k hops over the current comm graph (R_comm = 0.5 m);
              robots that do not know it hold still but still avoid collisions. Memoryless: a robot only knows the token while it is
              within k hops of an observer (this is what a feed-forward GNN with k layers can see).
  K<k>+mem    same, but a robot that has learned the token keeps it (upper bound for what a memoryless GNN could do)
  K7 = unlimited hops (graph diameter <= 7)
Informed robots run the potential-field expert toward their own slot. Same seed => same start and observers for every policy.
Run: python d3_baselines.py [episodes_per_formation] [workers]      (defaults: 50, cores-1)
Writes ../results/d3_baselines.csv (one row per episode) and ../figures/d3_baselines.png; prints a summary."""
import sys, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
import swarm_sim as S
from mj_env import SwarmEnv, ROBOT_R

ROOT = Path(__file__).resolve().parent.parent
T_STEPS, START_HALF, SEED0 = 1100, 0.5, 10000
OBS_LEVELS = {50: 4, 25: 2}                       # percent -> number of observers (of 8)
_env = []

def reach(A, know, K):
    R = know.copy(); cur = know.copy(); Ai = A.astype(np.int32)
    for _ in range(K):
        nxt = (Ai @ cur.astype(np.int32)) > 0
        if not (nxt & ~R).any(): break
        R |= nxt; cur = nxt
    return R

def variants():
    v = [dict(policy="oracle", obs_pct=100, n_obs=8, K=None, mem=False)]
    for pct, n in OBS_LEVELS.items():
        v.append(dict(policy="K0", obs_pct=pct, n_obs=n, K=0, mem=False))
        for K in (1, 2, 3, 7):
            for mem in (False, True):
                v.append(dict(policy=f"K{K}" + ("+mem" if mem else ""), obs_pct=pct, n_obs=n, K=K, mem=mem))
    return v

def episode(args):
    var, f, seed = args
    if not _env: _env.append(SwarmEnv())
    env = _env[0]
    rng = np.random.default_rng(seed); p, th = S.random_start(rng, half=START_HALF); tg = S.slots(f)
    obs = np.zeros(S.N, bool); obs[np.random.default_rng([seed, 99]).choice(S.N, var["n_obs"], replace=False)] = True
    env.reset(p, th); p, th = env.poses(); know = obs.copy(); errs = []; minc = 9.0
    for _ in range(T_STEPS):
        if var["policy"] == "oracle": know = np.ones(S.N, bool)
        elif var["K"] == 0: know = obs
        else:
            A = S.comm_adjacency(p, S.R_COMM)
            know = reach(A, know if var["mem"] else obs, var["K"])
        targets = np.where(know[:, None], tg, p)
        v, w = S.to_unicycle(S.expert_velocity_vector(p, targets), th); p, th = env.step(v, w)
        errs.append(S.formation_error(p, tg)); minc = min(minc, S.min_clearance(p))
    errs = np.array(errs); bad = np.where(errs >= 0.025)[0]
    return dict(policy=var["policy"], obs_pct=var["obs_pct"], formation=f, seed=seed, final_err=errs[-1],
                success=bool(errs[-1] < 0.025), success_05=bool(errs[-1] < 0.05), collision=bool(minc < 2 * ROBOT_R - 1e-9),
                min_clearance=minc, informed_final=float(know.mean()),
                t_converge_s=(bad[-1] + 1) * S.DT if errs[-1] < 0.025 else np.nan)

def summarize(df):
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    orc = df[df.policy == "oracle"]
    print(f"\nORACLE (all robots know the gesture): success {100*orc.success.mean():.1f}%, collisions {orc.collision.sum()}/{len(orc)}, "
          f"mean final error {orc.final_err.mean():.4f} m, median time {orc.t_converge_s.median():.1f} s")
    d = df[df.policy != "oracle"]
    order = ["K0", "K1", "K1+mem", "K2", "K2+mem", "K3", "K3+mem", "K7", "K7+mem"]
    for pct in (50, 25):
        x = d[d.obs_pct == pct]
        g = x.groupby("policy").agg(success_pct=("success", lambda s: 100 * s.mean()), success05_pct=("success_05", lambda s: 100 * s.mean()),
                                    mean_final_err=("final_err", "mean"), robots_informed=("informed_final", "mean"),
                                    collisions=("collision", "sum"), median_t=("t_converge_s", "median")).reindex(order)
        print(f"\n--- {pct}% observability ({OBS_LEVELS[pct]} observers), {len(x)//len(order)} episodes per policy ---")
        print(g.round(3).to_string())
    x = d[d.obs_pct == 25].pivot_table(index="policy", columns="formation", values="success", aggfunc=lambda s: 100 * s.mean()).reindex(order)
    print("\n--- success % by formation at 25% observability ---"); print(x.round(0).to_string())
    best = d[d.obs_pct == 25].groupby("policy").success.mean().max()
    print(f"\nHEADROOM at 25% observability: oracle {100*orc.success.mean():.0f}% vs best no-learning policy {100*best:.0f}%.")

def plot(df):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = df[df.policy != "oracle"]; order = ["K0", "K1", "K1+mem", "K2", "K2+mem", "K3", "K3+mem", "K7", "K7+mem"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for a, pct in zip(ax, (50, 25)):
        s = d[d.obs_pct == pct].groupby("policy").success.mean().reindex(order) * 100
        a.bar(order, s.values, color=["C7"] + ["C0", "C9"] * 4); a.axhline(100 * df[df.policy == "oracle"].success.mean(), color="k", ls="--", label="oracle")
        a.set_title(f"{pct}% observability"); a.set_xlabel("no-learning policy"); a.tick_params(axis="x", rotation=45)
    ax[0].set_ylabel("success (%)"); ax[0].legend(); fig.tight_layout(); fig.savefig(ROOT / "figures" / "d3_baselines.png", dpi=300)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 1)
    jobs = [(v, f, SEED0 + s) for v in variants() for f in S.FORMATIONS for s in range(n)]
    print(f"{len(jobs)} episodes on {workers} workers ...", flush=True)
    with Pool(workers) as pool: rows = pool.map(episode, jobs, chunksize=8)
    df = pd.DataFrame(rows); (ROOT / "results").mkdir(exist_ok=True); (ROOT / "figures").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "d3_baselines.csv", index=False); summarize(df); plot(df)
    print("\nsaved results/d3_baselines.csv and figures/d3_baselines.png")
