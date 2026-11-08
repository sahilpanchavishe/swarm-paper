"""Day 15 (EXPLORATORY): replay the worst failing filtered wide-box memoryless-GNN episodes and look at the worst robot at the end.
Self-check: the replayed final error must match the saved one (warns otherwise).  Per episode: robots outside 25 mm of their slot, worst robot's
speed, share of the last 100 steps in which the filter changed its command, and its nearest-neighbour distance.
Reading: speed ~0 and filter active -> held back by the filter; speed ~0 and filter idle -> stuck by itself; speed high -> still moving/drifting.
Run (from src/): python d15_replay.py [--max 12] [--out ../results/d15_replay.csv]"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.optimize import linear_sum_assignment
import swarm_sim as S
from d4_common import GNN, gesture_onehot, FORM_ID, T_STEPS, V_MAX, N
from d12_common import safety_filter
from mj_env import SwarmEnv
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=12); ap.add_argument("--out", default=str(ROOT / "results" / "d15_replay.csv")); a = ap.parse_args()
d = pd.read_csv(ROOT / "results" / "d12_widebox_safe.csv"); d = d[(d.safety == True) & ~d.success.astype(bool) & d.policy.str.startswith("GNN") & ~d.policy.str.contains("latched")]
d = d.sort_values("final_err", ascending=False).head(a.max); env = SwarmEnv(); rows = []
for _, r in d.iterrows():
    K = int(r.policy.split("K")[1][0]); name = f"d5_K{K}_s{int(r.train_seed)}"
    model = GNN(K, False); model.load_state_dict(torch.load(ROOT / "runs" / f"{name}.pt", map_location="cpu")); model.eval()
    seed, f, pct = int(r.seed), r.formation, int(r.obs_pct)
    p, th = S.random_start(np.random.default_rng(seed), half=0.9); tg = np.array(S.slots(f))
    obs = np.zeros(N, bool); obs[np.random.default_rng([seed, 99]).choice(N, {100: 8, 50: 4, 25: 2}[pct], replace=False)] = True
    env.reset(p, th); p, th = env.poses(); g = torch.tensor(gesture_onehot(FORM_ID[f]))[None]; spd = []; act = []
    with torch.no_grad():
        for t in range(T_STEPS):
            u = model(torch.tensor(p, dtype=torch.float32)[None], g, torch.tensor(obs, dtype=torch.float32)[None])[0].numpy() * V_MAX
            uf = safety_filter(u, p, float(r.get("dsafe", 0.10)), float(r.get("gamma", 20.0)), 0.30)
            if t >= T_STEPS - 100: spd.append(np.linalg.norm(uf, axis=1)); act.append(np.linalg.norm(uf - u, axis=1) > 1e-6)
            v, w = S.to_unicycle(uf, th); p, th = env.step(v, w)
    fin = float(S.formation_error(p, tg)); ri, ci = linear_sum_assignment(np.linalg.norm(p[:, None] - tg[None], axis=2)); e = np.linalg.norm(p[ri] - tg[ci], axis=1)
    k = int(ri[np.argmax(e)]); D = np.linalg.norm(p - p[k], axis=1); D[k] = 9
    rows.append(dict(policy=r.policy, train_seed=int(r.train_seed), seed=seed, formation=f, obs_pct=pct, init_comp=int(r.init_comp), saved_err_mm=1000 * r.final_err, replay_err_mm=1000 * fin,
                     match=bool(abs(fin - r.final_err) < 1e-3), robots_off_25mm=int((e > 0.025).sum()), worst_robot_err_mm=1000 * e.max(), worst_speed_mm_s=1000 * float(np.mean(np.array(spd)[:, k])),
                     worst_filter_active_pct=100 * float(np.mean(np.array(act)[:, k])), worst_nn_dist_mm=1000 * float(D.min()), worst_is_observer=bool(obs[k])))
R = pd.DataFrame(rows); R.to_csv(a.out, index=False); pd.set_option("display.width", 250); print(R.round(1).to_string(index=False))
if not R.match.all(): print("\nWARNING: replay does not reproduce the saved episodes for", int((~R.match).sum()), "rows. Do not use this table; paste the run_episode body from src/d12_common.py.")
print(f"saved {a.out}")
