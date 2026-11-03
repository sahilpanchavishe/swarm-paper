"""Day 4a: generate demonstration data in MuJoCo (training seeds 0..9999, never the test seeds 10000+).
Every sampled step stores positions, the oracle-expert label (all robots informed), observers, and the "latch" masks.
Trajectories are driven by a mix of behaviour policies so the dataset also covers states visited by robots that do not yet
know the gesture (they hold still): episode mode = i % 4 -> 0 oracle (all move), 1/2/3 = memoryless K-hop token flooding.
Labels are ALWAYS the oracle expert, so the learner is asked to imitate "everyone informed" behaviour.
Run: python d4_gen_data.py [episodes_per_formation] [workers]      default 300, cores-1
Writes ../data/raw/d4_demos.npz (git-ignored)."""
import sys, os, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import swarm_sim as S
from mj_env import SwarmEnv
from d4_common import reach, oracle_label, FORM_ID, T_STEPS, START_HALF, N, R_COMM

ROOT = Path(__file__).resolve().parent.parent
SUB = 4                                                   # store every 4th control step
_env = []

def gen(args):
    f, seed, mode, n_obs = args
    if not _env: _env.append(SwarmEnv())
    env = _env[0]
    rng = np.random.default_rng(seed); p, th = S.random_start(rng, half=START_HALF); tg = S.slots(f)
    obs = np.zeros(N, bool); obs[np.random.default_rng([seed, 99]).choice(N, n_obs, replace=False)] = True
    env.reset(p, th); p, th = env.poses(); mem = {k: obs.copy() for k in (1, 2, 3)}
    P, Y, L = [], [], []
    for t in range(T_STEPS):
        A = S.comm_adjacency(p, R_COMM)
        for k in mem: mem[k] = reach(A, mem[k], k)
        if t % SUB == 0:
            P.append(p.copy()); Y.append(oracle_label(p, f)); L.append(np.stack([mem[1], mem[2], mem[3]], -1))
        know = np.ones(N, bool) if mode == 0 else reach(A, obs, mode)
        v, w = S.to_unicycle(S.expert_velocity_vector(p, np.where(know[:, None], tg, p)), th); p, th = env.step(v, w)
    return dict(P=np.array(P, np.float32), Y=np.array(Y, np.float32), L=np.array(L, bool), obs=obs, f=FORM_ID[f], seed=seed, mode=mode)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 1)
    # distinct seeds per episode; behaviour mode (0 oracle, 1-3 K-hop flooding) and observer count cycle independently of the formation
    jobs = [(f, fi * n + i, i % 4, [8, 4, 2][(i // 4) % 3]) for fi, f in enumerate(S.FORMATIONS) for i in range(n)]
    assert max(j[1] for j in jobs) < 10000, "training seeds must stay below 10000"
    print(f"{len(jobs)} demonstration episodes on {workers} workers ...", flush=True); t0 = time.time()
    with Pool(workers) as pool: rows = pool.map(gen, jobs, chunksize=4)
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "data" / "raw" / "d4_demos.npz", P=np.stack([r["P"] for r in rows]), Y=np.stack([r["Y"] for r in rows]),
                        L=np.stack([r["L"] for r in rows]), obs=np.stack([r["obs"] for r in rows]), f=np.array([r["f"] for r in rows]),
                        seed=np.array([r["seed"] for r in rows]), mode=np.array([r["mode"] for r in rows]))
    print(f"saved data/raw/d4_demos.npz  ({len(rows)} episodes, {rows[0]['P'].shape[0]} samples each, {time.time()-t0:.0f} s)")
