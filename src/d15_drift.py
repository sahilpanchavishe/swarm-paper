"""Day 15 (EXPLORATORY, no filter): do the uninformed robots of the memoryless GNN K2 move together before the formation forms?
Wide start box, 25% observability, true token throughout. Per episode: mean distance of uninformed robots / observers to the formation centre,
and the number of communication components, at several control steps. rad_slots is the mean slot distance to the centre (the target radius).
Run (from src/): python d15_drift.py [--n 20] [--seed0 80000] [--half 0.9] [--out ../results/d15_drift.csv] [--workers 7]"""
import argparse, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, torch
import swarm_sim as S
from d4_common import GNN, gesture_onehot, FORM_ID, T_STEPS, R_COMM, V_MAX, N
from mj_env import SwarmEnv
from d15_scenarios import labels
ROOT = Path(__file__).resolve().parent.parent
TS = [0, 100, 200, 300, 500, 700, 1099]
_env = []

def episode(a):
    sd, K, f, seed, half, name = a
    torch.set_num_threads(1)
    if not _env: _env.append(SwarmEnv())
    env = _env[0]; model = GNN(K, False); model.load_state_dict(sd); model.eval()
    p, th = S.random_start(np.random.default_rng(seed), half=half); tg = S.slots(f); c0 = np.array(tg).mean(0)
    obs = np.zeros(N, bool); obs[np.random.default_rng([seed, 99]).choice(N, 2, replace=False)] = True
    env.reset(p, th); p, th = env.poses(); g = torch.tensor(gesture_onehot(FORM_ID[f]))[None]; un = ~obs
    row = dict(model=name, formation=f, seed=seed, rad_slots=float(np.linalg.norm(np.array(tg) - c0, axis=1).mean()))
    with torch.no_grad():
        for t in range(T_STEPS):
            if t in TS:
                row[f"rad_un_{t}"] = float(np.linalg.norm(p[un] - c0, axis=1).mean()); row[f"rad_ob_{t}"] = float(np.linalg.norm(p[obs] - c0, axis=1).mean())
                row[f"comp_{t}"] = int(labels(S.comm_adjacency(p, R_COMM))[1])
            u = model(torch.tensor(p, dtype=torch.float32)[None], g, torch.tensor(obs, dtype=torch.float32)[None])[0].numpy() * V_MAX
            v, w = S.to_unicycle(u, th); p, th = env.step(v, w)
    row["final_err"] = float(S.formation_error(p, tg)); row["success"] = bool(row["final_err"] < 0.025); row["init_comp"] = row["comp_0"]
    return row

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed0", type=int, default=80000); ap.add_argument("--half", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--out", default=str(ROOT / "results" / "d15_drift.csv"))
    a = ap.parse_args(); jobs = []
    for s in range(3):
        name = f"d5_K2_s{s}"; sd = torch.load(ROOT / "runs" / f"{name}.pt", map_location="cpu")
        jobs += [(sd, 2, f, a.seed0 + i, a.half, name) for f in S.FORMATIONS for i in range(a.n)]
    with Pool(a.workers) as pool: rows = pool.map(episode, jobs, chunksize=2)
    d = pd.DataFrame(rows); d.to_csv(a.out, index=False)
    pd.set_option("display.width", 220)
    print(f"{len(d)} episodes, success {100 * d.success.mean():.1f}%, mean initial components {d.init_comp.mean():.2f}, mean slot radius {1000 * d.rad_slots.mean():.0f} mm")
    print(pd.DataFrame({"t": TS, "uninformed_radius_mm": [1000 * d[f'rad_un_{t}'].mean() for t in TS], "observer_radius_mm": [1000 * d[f'rad_ob_{t}'].mean() for t in TS],
                        "mean_components": [d[f'comp_{t}'].mean() for t in TS], "pct_connected": [100 * (d[f'comp_{t}'] == 1).mean() for t in TS]}).round(1).to_string(index=False))
    print(f"saved {a.out}")
