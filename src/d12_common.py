"""Day 12: closed-loop engine v2 (MuJoCo). NEW file: d4_common.py, d3_baselines.py and swarm_sim.py are untouched.

With every option at its default this engine must reproduce the Day 3 baselines and the Day 6 GNN evaluation (run `python d12_eval.py repro`).
On top of that it adds
  * more metrics: max per-robot final error, contact steps, near-contact steps (< 0.09 m), time to converge, initial graph components;
  * perturbations applied to ALL policies alike (paired): packet loss, sensing delay, position noise, wide start box;
  * an optional decentralised pairwise safety filter on the commanded velocity;
  * a mid-episode command change, with explicit per-robot token registers (so a stale token can really be stale);
  * the capacity-control architectures of d12_models.py.

Perturbation definitions (state them exactly in Methods):
  loss   every control step each communication link is removed independently with probability `loss` (symmetric, memoryless)
  delay  the controller sees positions that are `delay` control steps old (own and neighbours'); the radio graph uses true positions
  noise  zero-mean Gaussian noise (std in metres) added to every position the controller sees; the error metrics use true positions
Token registers (command-change mode and the latched policies):
  each robot holds (token, stamp). Observers read the current command directly. Every step the register is flooded `hops` hops.
  'fresh' (latest-wins): a robot adopts the highest stamp it can reach.   'naive' (latch-first): a robot only adopts a token if it holds none.
  With one fixed command both rules reduce to the Day 3 'relay + memory' / Day 4 latched behaviour (checked by `repro`).
"""
import numpy as np, torch
import swarm_sim as S
from mj_env import SwarmEnv, ROBOT_R
from d4_common import reach, FORM_ID, N, V_MAX, R_COMM, T_STEPS, START_HALF
from d12_models import make_model

FORMS = S.FORMATIONS
SL = np.stack([S.slots(f) for f in FORMS])             # (4, N, 2): slot of robot i in formation c is SL[c, i]
DEFAULT = SL.mean(0)                                   # mean slot of each role over the four formations (relayD default)
CONTACT = 2 * ROBOT_R - 1e-9                           # same contact definition as Day 6
NEAR = 0.09                                            # near-contact margin: 16 mm above contact (0.074 m)
BASE_CFG = dict(start_half=START_HALF, T=T_STEPS, loss=0.0, delay=0, noise=0.0, safety=False, dsafe=0.09, gamma=10.0, sense=0.30,
                change=False, t_switch=550, T_change=1650)
COND_COLS = ["start_half", "loss", "delay", "noise", "safety", "dsafe", "gamma", "change"]
_env = []

# ---------------------------------------------------------------- safety filter
def _project(u0, a, b):
    """Closest point to u0 in {u : a_k . u >= b_k for all k}. Exact: u0, projections on each boundary line, and every pair of lines."""
    tol = 1e-10
    if np.all(a @ u0 >= b - tol): return u0
    m = len(b); cand = [np.zeros(2)]
    for k in range(m):
        cand.append(u0 + a[k] * (b[k] - a[k] @ u0) / (a[k] @ a[k]))
        for l in range(k + 1, m):
            M = np.array([a[k], a[l]]); det = np.linalg.det(M)
            if abs(det) > 1e-12: cand.append(np.linalg.solve(M, np.array([b[k], b[l]])))
    cand = np.array(cand); viol = np.maximum(0.0, (b[None, :] - cand @ a.T)).max(1); dist = ((cand - u0) ** 2).sum(1)
    ok = viol <= tol
    if ok.any(): return cand[np.where(ok)[0][dist[ok].argmin()]]
    return cand[viol.argmin()]                          # already inside the safety margin and over-constrained: least violation

def safety_filter(u, p, dsafe=0.09, gamma=10.0, sense=0.30):
    """Decentralised pairwise control-barrier-style filter on the desired velocity u (N,2) in m/s.
    Barrier h_ij = |p_i-p_j|^2 - dsafe^2. Full condition  dh/dt >= -gamma*h  is  2 d_ij.(u_i-u_j) >= -gamma*h_ij.
    Each robot enforces HALF of it on its own velocity,  2 d_ij.u_i >= -gamma*h_ij/2 ; the two halves add up to the full condition without
    either robot knowing the other's velocity. Only neighbours within `sense` metres are used. Empirical filter, NOT a formal guarantee
    (the unicycle only approximately tracks a commanded velocity)."""
    out = u.copy(); D = p[:, None] - p[None]; d2 = (D ** 2).sum(-1)
    for i in range(len(p)):
        js = [j for j in range(len(p)) if j != i and d2[i, j] < sense ** 2]
        if js: out[i] = _project(u[i], 2 * D[i, js], -0.5 * gamma * (d2[i, js] - dsafe ** 2))
    return out

# ---------------------------------------------------------------- helpers
def n_components(A):
    seen = np.zeros(len(A), bool); c = 0
    for s in range(len(A)):
        if seen[s]: continue
        c += 1; seen[s] = True; stack = [s]
        while stack:
            i = stack.pop()
            for j in np.nonzero(A[i] & ~seen)[0]: seen[j] = True; stack.append(j)
    return c

def update_registers(tok, stamp, obs, cmd, cstamp, A, hops, mode):
    """One control step of token-register updating. tok/stamp (N,) ints, -1 = nothing held. Returns new arrays (inputs are not modified)."""
    tok, stamp = tok.copy(), stamp.copy(); tok[obs] = cmd; stamp[obs] = cstamp     # observers read the command directly
    idx = np.arange(len(tok))
    for _ in range(hops):
        nb = np.where(A, stamp[None, :], -1); j = nb.argmax(1); best = nb[idx, j]
        take = (best > stamp) if mode == "fresh" else ((stamp < 0) & (best >= 0))
        tok = np.where(take, tok[j], tok); stamp = np.where(take, best, stamp)
    return tok, stamp

def gnn_forward(model, P, tok, know, A=None):
    """Same computation as d4_common.GNN.forward, but with a per-robot token tok (B,N,4) and an optional adjacency override A (B,N,N)."""
    B = P.shape[0]
    if A is None:
        D = (P[:, :, None] - P[:, None]).norm(dim=-1); A = ((D <= R_COMM) & (D > 0)).float()
    ids = model.eye.expand(B, N, N)
    h = model.enc(torch.cat([P / 1.1, ids, tok, know[..., None]], -1))
    rel = (P[:, None] - P[:, :, None]) / R_COMM
    for k in range(model.K):
        hj = h[:, None].expand(B, N, N, model.hid)
        m = model.msg[k](torch.cat([hj, rel], -1)) * A[..., None]
        h = h + model.upd[k](torch.cat([h, m.sum(2) / 4.0], -1))
    return model.head(torch.cat([h, P / 1.1, ids], -1))

# ---------------------------------------------------------------- one episode
def run_episode(args):
    """args = (pol, cfg, formation, seed, obs_pct). pol is a plain dict (see d12_eval.parse_token). Returns one result row (dict)."""
    pol, cfg, f, seed, obs_pct = args
    cfg = {**BASE_CFG, **cfg}; torch.set_num_threads(1)
    if not _env: _env.append(SwarmEnv())
    env = _env[0]; kind = pol["kind"]; K = pol.get("K", 0)
    model = None
    if kind == "gnn":
        model = make_model(pol["arch"], K, pol["latched"]); model.load_state_dict(pol["sd"]); model.eval()
    obs_sc = None
    if cfg.get("scenario"):
        from d15_scenarios import make_scenario
        p, th, obs_sc = make_scenario(seed, cfg["scenario"], obs_pct, half=cfg["start_half"])
    else:
        rng = np.random.default_rng(seed); p, th = S.random_start(rng, half=cfg["start_half"])
    fi = FORM_ID[f]; fi1 = fi
    if cfg["change"]: fi1 = (fi + 1 + int(np.random.default_rng([seed, 55]).integers(0, 3))) % 4
    obs = np.zeros(N, bool); obs[np.random.default_rng([seed, 99]).choice(N, {100: 8, 50: 4, 25: 2}[obs_pct], replace=False)] = True
    if obs_sc is not None: obs = obs_sc
    rl, rn = np.random.default_rng([seed, 77]), np.random.default_rng([seed, 78])
    ro = np.random.default_rng([seed, 88]); n_con = max(1, int(obs.sum()) // 2); wrong_f = (fi + 1 + int(np.random.default_rng([seed, 66]).integers(0, 3))) % 4
    T = cfg["T_change"] if cfg["change"] else cfg["T"]
    env.reset(p, th); p, th = env.poses(); init_comp = n_components(S.comm_adjacency(p, R_COMM))
    rtok, rstamp = np.full(N, -1), np.full(N, -1); hist = []; e_old, e_new, minc = [], [], []; know = obs
    with torch.no_grad():
        for t in range(T):
            sw = cfg["change"] and t >= cfg["t_switch"]; ci, cst = (fi1, 1) if sw else (fi, 0)
            obs_e, cn = obs, np.full(N, ci)      # observer imperfections (O1 miss, O2 wrong, O3 contra); no effect when the keys are unset
            if cfg.get("miss") or cfg.get("wrong") or cfg.get("contra"):
                obs_e = obs & (ro.random(N) >= cfg.get("miss", 0.0))
                if cfg.get("wrong"): cn = np.where(ro.random(N) < cfg["wrong"], (ci + 1 + ro.integers(0, 3, N)) % 4, cn)
                if cfg.get("contra"): cn[np.nonzero(obs)[0][:n_con]] = wrong_f
            cmd = cn[obs_e]
            hist.append(p.copy()); pc = hist[max(0, len(hist) - 1 - cfg["delay"])]
            if cfg["noise"] > 0: pc = pc + rn.normal(0.0, cfg["noise"], pc.shape)
            A = S.comm_adjacency(p, R_COMM); M = None
            if cfg["loss"] > 0:
                keep = np.triu(rl.random((N, N)) >= cfg["loss"], 1); M = keep | keep.T; A = A & M
            if kind == "gnn":
                if pol["reg"] == "none": know = obs_e; tok = cn
                else:
                    rtok, rstamp = update_registers(rtok, rstamp, obs_e, cmd, cst, A, max(K, 1), pol["reg"]); know = rstamp >= 0; tok = rtok
                oh = np.zeros((N, 4), np.float32); ix = np.nonzero(know)[0]; oh[ix, tok[ix]] = 1.0
                Pt = torch.tensor(p, dtype=torch.float32)[None]; D = (Pt[:, :, None] - Pt[:, None]).norm(dim=-1)
                At = ((D <= R_COMM) & (D > 0)).float()
                if M is not None: At = At * torch.tensor(M, dtype=torch.float32)[None]
                u = gnn_forward(model, torch.tensor(pc, dtype=torch.float32)[None], torch.tensor(oh)[None],
                                torch.tensor(know, dtype=torch.float32)[None], At)[0].numpy() * V_MAX
            else:
                if kind == "oracle": know = np.ones(N, bool); tok = np.full(N, ci)
                elif pol["mem"] == "none": know = obs if K == 0 else reach(A, obs, K); tok = np.full(N, ci)
                else:
                    rtok, rstamp = update_registers(rtok, rstamp, obs_e, cmd, cst, A, K, pol["mem"]); know = rstamp >= 0; tok = rtok
                tg = SL[np.where(tok >= 0, tok, 0), np.arange(N)]
                u = S.expert_velocity_vector(pc, np.where(know[:, None], tg, pc if pol["default"] == "hold" else DEFAULT))
            if cfg["safety"] and cfg.get("pred"):
                _m = __import__("d15_pred"); hh = [hist[max(0, len(hist) - 1 - cfg["delay"])]] if len(hist) < 2 + cfg["delay"] else [hist[-2 - cfg["delay"]], hist[-1 - cfg["delay"]]]
                pp = _m.predict([h_ + (pc - hist[max(0, len(hist) - 1 - cfg["delay"])]) for h_ in hh], cfg["delay"], S.DT)
                u = _m.safety_filter_pred(u, pp, cfg["dsafe"], cfg["gamma"], cfg["sense"], cfg["pred"])
            elif cfg["safety"]: u = (__import__("d15_filter").safety_filter_vm(u, pc, cfg["dsafe"], cfg["gamma"], cfg["sense"], cfg["delay"] * S.DT * cfg["vmargin"]) if cfg.get("vmargin") else safety_filter(u, pc, cfg["dsafe"], cfg["gamma"], cfg["sense"]))
            v, w = S.to_unicycle(u, th); p, th = env.step(v, w)
            e1 = S.formation_error(p, SL[fi1]); e_new.append(e1); e_old.append(S.formation_error(p, SL[fi]) if cfg["change"] else e1)
            minc.append(S.min_clearance(p))
    errs, minc = np.array(e_new), np.array(minc); bad = np.where(errs >= 0.025)[0]; ok = bool(errs[-1] < 0.025)
    row = dict(policy=pol["name"], train_seed=pol.get("train_seed", np.nan), formation=f, seed=seed, obs_pct=obs_pct,
               final_err=errs[-1], success=ok, success_05=bool(errs[-1] < 0.05), max_err=float(np.linalg.norm(p - SL[fi1], axis=1).max()),
               collision=bool(minc.min() < CONTACT), min_clearance=float(minc.min()), contact_steps=int((minc < CONTACT).sum()),
               near_steps=int((minc < NEAR).sum()), t_converge_s=((bad[-1] + 1 if len(bad) else 0) * S.DT) if ok else np.nan,
               informed_final=float(np.mean(know)), init_comp=init_comp,
               **{c: cfg[c] for c in COND_COLS}, **{k: cfg[k] for k in ("scenario", "vmargin", "pred") if cfg.get(k)}, **{k: cfg[k] for k in ("miss", "wrong", "contra") if k in cfg})
    if cfg["change"]: row.update(err_pre_switch=float(e_old[cfg["t_switch"] - 1]), old_final_err=float(e_old[-1]), new_formation=FORMS[fi1])
    return row
