"""Day 15: delay-aware pairwise safety filter. The filter sees positions that are `delay` steps old (history h, newest last). Each robot's current
position is predicted as p_delayed + v_est * delay_s (v_est from a finite difference over the last two snapshots), then the same pairwise barrier
as d12_common.safety_filter is applied to the predicted positions, with an extra margin `pad` for prediction error and a speed-dependent term.
With delay = 0 and pad = 0 it must equal safety_filter exactly (self-test: python d15_pred.py)."""
import numpy as np

def predict(hist, delay, dt):
    """hist: list of (N,2) delayed-position snapshots, newest last. Returns predicted current positions."""
    pc = hist[-1]
    if delay <= 0 or len(hist) < 2: return pc
    v = (hist[-1] - hist[-2]) / dt
    return pc + v * (delay * dt)

def safety_filter_pred(u, ppred, dsafe, gamma, sense, pad=0.0):
    from d12_common import _project
    D = ppred[:, None] - ppred[None]; d2 = (D ** 2).sum(-1); out = u.copy(); r = dsafe + pad
    for i in range(len(ppred)):
        js = [j for j in range(len(ppred)) if j != i and d2[i, j] < (sense + pad) ** 2]
        if js: out[i] = _project(u[i], 2 * D[i, js], -0.5 * gamma * (d2[i, js] - r ** 2))
    return out

if __name__ == "__main__":
    from d12_common import safety_filter
    rng = np.random.default_rng(0); worst = 0.0
    for _ in range(200):
        p = rng.uniform(-0.3, 0.3, (8, 2)); u = rng.normal(0, 0.1, (8, 2))
        worst = max(worst, np.abs(safety_filter_pred(u, p, 0.09, 10.0, 0.30) - safety_filter(u, p, 0.09, 10.0, 0.30)).max())
    print("self-test max difference:", worst, "PASS" if worst < 1e-12 else "FAIL")
