"""Day 4 shared code: GNN policy, feature construction, closed-loop evaluation in MuJoCo.

Information model (state this exactly in the paper's Methods):
  * Every robot knows its own world-frame position and its fixed role id (robot i -> slot i of the chosen formation).
    The formation geometry is in the world frame, so a robot needs these two to know where its slot is. This is NOT a
    "no global information" setting; do not claim that.
  * Only observers see the gesture. The gesture reaches other robots only through message passing over the
    communication graph (R_comm = 0.5 m). Messages carry the sender's hidden state and its relative position.
  * Memoryless GNN: K message-passing layers, no recurrent state. A robot can only use the gesture if an observer is within K hops *now*.
  * Latched variant (ablation): a robot additionally gets a 1-bit latch "I have received the gesture earlier" (flooding K hops,
    kept once received), so it carries the gesture forward after its informer moves away.
Output: desired world-frame velocity (m/s) divided by V_MAX, then converted to (v, w) by swarm_sim.to_unicycle.
Label: potential-field expert with EVERY robot informed (oracle gesture knowledge), clipped to |u| <= V_MAX.
"""
import numpy as np, torch, torch.nn as nn
import swarm_sim as S
from mj_env import SwarmEnv, ROBOT_R

N, V_MAX, R_COMM = S.N, S.V_MAX, S.R_COMM
T_STEPS, START_HALF = 1100, 0.5
FORM_ID = {f: i for i, f in enumerate(S.FORMATIONS)}


def reach(A, know, K):
    """Boolean K-hop reach of the 'know' set over adjacency A (same as d3_baselines)."""
    R = know.copy(); cur = know.copy(); Ai = A.astype(np.int32)
    for _ in range(K):
        nxt = (Ai @ cur.astype(np.int32)) > 0
        if not (nxt & ~R).any(): break
        R |= nxt; cur = nxt
    return R


def oracle_label(p, f):
    """Expert velocity with all robots informed, clipped to V_MAX and normalised to [-1, 1]."""
    u = S.expert_velocity_vector(p, S.slots(f))
    n = np.linalg.norm(u, axis=1, keepdims=True)
    return u * np.minimum(1.0, V_MAX / np.maximum(n, 1e-9)) / V_MAX


class GNN(nn.Module):
    def __init__(self, K, latched=False, hid=64):
        super().__init__()
        self.K, self.latched, self.hid = K, latched, hid
        fin = 2 + N + 4 + 1                        # pos, role one-hot, gesture token (4), know flag
        self.enc = nn.Sequential(nn.Linear(fin, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(hid + 2, hid), nn.ReLU(), nn.Linear(hid, hid)) for _ in range(K)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, hid)) for _ in range(K)])
        self.head = nn.Sequential(nn.Linear(hid + 2 + N, 2 * hid), nn.ReLU(), nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, 2))
        self.register_buffer("eye", torch.eye(N))

    def forward(self, P, gest, know):
        """P (B,N,2) metres; gest (B,4) one-hot formation; know (B,N) float 0/1 = observer (or observer OR latched)."""
        B = P.shape[0]
        D = (P[:, :, None] - P[:, None]).norm(dim=-1)
        A = ((D <= R_COMM) & (D > 0)).float()                              # (B,N,N) comm graph
        ids = self.eye.expand(B, N, N)
        tok = gest[:, None, :] * know[:, :, None]
        h = self.enc(torch.cat([P / 1.1, ids, tok, know[..., None]], -1))
        rel = (P[:, None] - P[:, :, None]) / R_COMM                         # rel[b,i,j] = p_j - p_i
        for k in range(self.K):
            hj = h[:, None].expand(B, N, N, self.hid)
            m = self.msg[k](torch.cat([hj, rel], -1)) * A[..., None]
            agg = m.sum(2) / 4.0
            h = h + self.upd[k](torch.cat([h, agg], -1))
        return self.head(torch.cat([h, P / 1.1, ids], -1))                  # (B,N,2) normalised velocity


def gesture_onehot(f_idx):
    g = np.zeros(4, np.float32); g[f_idx] = 1; return g


_env = []

def closed_loop_episode(args):
    """args: (state_dict, K, latched, formation, seed, obs_pct). Returns one result row. Same seed => same start/observers as d3_baselines."""
    sd, K, latched, f, seed, obs_pct = args
    torch.set_num_threads(1)
    if not _env: _env.append(SwarmEnv())
    env = _env[0]
    model = GNN(K, latched); model.load_state_dict(sd); model.eval()
    n_obs = {100: 8, 50: 4, 25: 2}[obs_pct]
    rng = np.random.default_rng(seed); p, th = S.random_start(rng, half=START_HALF); tg = S.slots(f)
    obs = np.zeros(N, bool); obs[np.random.default_rng([seed, 99]).choice(N, n_obs, replace=False)] = True
    env.reset(p, th); p, th = env.poses(); latch = obs.copy(); g = torch.tensor(gesture_onehot(FORM_ID[f]))[None]
    errs = []; minc = 9.0
    with torch.no_grad():
        for _ in range(T_STEPS):
            if latched:
                latch = reach(S.comm_adjacency(p, R_COMM), latch, max(K, 1))
                know = latch
            else:
                know = obs
            u = model(torch.tensor(p, dtype=torch.float32)[None], g, torch.tensor(know, dtype=torch.float32)[None])[0].numpy() * V_MAX
            v, w = S.to_unicycle(u, th); p, th = env.step(v, w)
            errs.append(S.formation_error(p, tg)); minc = min(minc, S.min_clearance(p))
    errs = np.array(errs)
    return dict(K=K, latched=latched, formation=f, seed=seed, obs_pct=obs_pct, final_err=errs[-1], success=bool(errs[-1] < 0.025),
                success_05=bool(errs[-1] < 0.05), collision=bool(minc < 2 * ROBOT_R - 1e-9), min_clearance=minc)
