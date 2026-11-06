import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np, swarm_sim as S, npgnn
from mj_env import SwarmEnv
N, V_MAX, R_COMM, T_STEPS, START_HALF = S.N, S.V_MAX, S.R_COMM, 1100, 0.5
FORM_ID = {f:i for i,f in enumerate(S.FORMATIONS)}
def reach(A, know, K):
    R = know.copy(); cur = know.copy(); Ai = A.astype(np.int32)
    for _ in range(K):
        nxt = (Ai @ cur.astype(np.int32)) > 0
        if not (nxt & ~R).any(): break
        R |= nxt; cur = nxt
    return R
def run(sd, K, latched, f, seed, pct, env, record=False, cut=None):
    rng = np.random.default_rng(seed); p, th = S.random_start(rng, half=START_HALF); tg = S.slots(f)
    obs = np.zeros(N, bool); obs[np.random.default_rng([seed, 99]).choice(N, {100:8,50:4,25:2}[pct], replace=False)] = True
    env.reset(p, th); p, th = env.poses(); latch = obs.copy()
    g = np.zeros(4, np.float32); g[FORM_ID[f]] = 1
    traj=[]; degs=[]; errs=[]; knows=[]; minc = 9.0
    for t in range(T_STEPS):
        A = S.comm_adjacency(p, R_COMM)
        if latched: latch = reach(A, latch, max(K,1)); know = latch
        else: know = obs
        u = npgnn.forward(sd, K, p.astype(np.float32), g, know.astype(np.float32))*V_MAX
        v, w = S.to_unicycle(u, th); p, th = env.step(v, w)
        if record and t % 5 == 0:
            traj.append(p.copy()); degs.append(A.sum(1)); errs.append(np.linalg.norm(p-tg,axis=1)); knows.append(reach(A, obs, K).copy())
        d = np.linalg.norm(p[:,None]-p[None],axis=-1)+np.eye(N)*9; minc = min(minc, d.min())
    e = np.linalg.norm(p-tg,axis=1)
    return dict(final_err=float(e.mean()), success=bool(e.mean()<0.025), p0=None, pf=p.copy(), obs=obs, traj=np.array(traj), deg=np.array(degs), err=np.array(errs), know=np.array(knows), minc=minc)
