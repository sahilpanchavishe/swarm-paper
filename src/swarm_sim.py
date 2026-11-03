"""
swarm_sim.py  -- Day 1: fast kinematic simulator for 8 differential-drive (e-puck-like) robots.
Pure NumPy. No GPU needed. Used for the large experiment batches; a subset is later re-checked in Webots.
"""
import numpy as np

# ---- e-puck constants (from the e-puck spec / your manuscript) ----
N = 8
DT = 0.032                 # s per control step (550 steps = 17.6 s)
R_WHEEL = 0.0205           # m
AXLE = 0.052               # m
W_WHEEL_MAX = 6.28         # rad/s, MAX WHEEL speed (not yaw rate!)
V_MAX = W_WHEEL_MAX * R_WHEEL        # ~0.1287 m/s
ARENA = 1.1                # half-width, m
R_COMM = 0.50              # m
ROBOT_RADIUS = 0.037       # m
FORMATIONS = ["circle", "line", "cluster", "split"]

def slots(formation):
    """Return (N,2) slot coordinates for a formation. Slot i is assigned to robot i (fixed roles)."""
    if formation == "circle":
        a = 2*np.pi*np.arange(N)/N
        return 0.45*np.c_[np.cos(a), np.sin(a)]
    if formation == "line":
        x = (np.arange(N)-(N-1)/2)*0.2
        return np.c_[x, np.zeros(N)]
    if formation == "cluster":
        g = [(i, j) for j in range(2) for i in range(4)]
        g = np.array(g, float) - [1.5, 0.5]
        return 0.15*g
    if formation == "split":
        g = np.array([(i, j) for j in range(2) for i in range(2)], float) - [0.5, 0.5]
        return np.vstack([0.15*g + [-0.6, 0], 0.15*g + [0.6, 0]])
    raise ValueError(formation)

def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def random_start(rng, min_sep=0.15, half=0.9):
    """Random positions in the arena with a minimum separation, random headings."""
    while True:
        p = rng.uniform(-half, half, size=(N, 2))
        d = np.linalg.norm(p[:, None]-p[None], axis=-1) + np.eye(N)*9
        if d.min() > min_sep:
            return p, rng.uniform(-np.pi, np.pi, N)

def comm_adjacency(p, r=R_COMM):
    d = np.linalg.norm(p[:, None]-p[None], axis=-1)
    A = (d <= r) & ~np.eye(len(p), dtype=bool)
    return A

def expert_velocity_vector(p, targets, k_att=2.8, k_rep=5e-3, r_inf=0.12, k_tan=0.5):
    """Desired planar velocity (m/s) from attraction + repulsion + tangential deflection.
    NOTE: we treat the potential-field 'force' as a velocity command (kinematic model, no dynamics),
    so k_att has units 1/s -- fix this in the paper's nomenclature."""
    u = k_att*(targets - p)
    diff = p[:, None]-p[None]                          # p_i - p_j
    d = np.linalg.norm(diff, axis=-1) + np.eye(len(p))*1e9
    mag = np.where(d < r_inf, k_rep*(1/d - 1/r_inf)/d**2, 0.0)
    dhat = diff/d[..., None]
    rep = (mag[..., None]*dhat).sum(1)
    perp = np.stack([-dhat[..., 1], dhat[..., 0]], -1)
    tan = (k_tan*mag[..., None]*perp).sum(1)
    return u + rep + tan

def to_unicycle(u, theta, k_w=4.0):
    """Convert desired velocity vector to (v, omega) respecting the WHEEL speed limit."""
    speed = np.linalg.norm(u, axis=1)
    err = wrap(np.arctan2(u[:, 1], u[:, 0]) - theta)
    w = k_w*err
    v = np.minimum(speed, V_MAX)*np.clip(np.cos(err), 0, 1)
    # wheel-limit scaling: keep v and w within |wheel| <= W_WHEEL_MAX
    wl = (v - w*AXLE/2)/R_WHEEL
    wr = (v + w*AXLE/2)/R_WHEEL
    s = np.maximum(1.0, np.maximum(np.abs(wl), np.abs(wr))/W_WHEEL_MAX)
    return v/s, w/s

def step(p, theta, v, w):
    p = p + DT*np.c_[v*np.cos(theta), v*np.sin(theta)]
    theta = wrap(theta + DT*w)
    p = np.clip(p, -ARENA+ROBOT_RADIUS, ARENA-ROBOT_RADIUS)
    return p, theta

def formation_error(p, targets):
    """Mean Euclidean distance between each robot and its assigned slot (m). DEFINE THIS IN THE PAPER."""
    return np.linalg.norm(p-targets, axis=1).mean()

def min_clearance(p):
    d = np.linalg.norm(p[:, None]-p[None], axis=-1) + np.eye(len(p))*9
    return d.min()

def run_expert(formation, seed, T=550):
    rng = np.random.default_rng(seed)
    p, th = random_start(rng)
    tg = slots(formation)
    err = [formation_error(p, tg)]; clr = [min_clearance(p)]
    for _ in range(T):
        u = expert_velocity_vector(p, tg)
        v, w = to_unicycle(u, th)
        p, th = step(p, th, v, w)
        err.append(formation_error(p, tg)); clr.append(min_clearance(p))
    return dict(err=np.array(err), min_clear=min(clr), final_err=err[-1],
                max_slot_err=np.linalg.norm(p-tg, axis=1).max())

if __name__ == "__main__":
    # SMOKE TEST ONLY (checks the code runs; these are not paper results)
    for f in FORMATIONS:
        res = [run_expert(f, s) for s in range(20)]
        fe = np.array([r["final_err"] for r in res]); mc = np.array([r["min_clear"] for r in res])
        print(f"{f:8s} final_err mean={fe.mean():.3f} max={fe.max():.3f} | "
              f"min clearance worst={mc.min():.3f} (2*radius={2*ROBOT_RADIUS:.3f})")
