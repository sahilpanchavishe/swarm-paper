"""Day 15: velocity-dependent delay margin for the pairwise safety filter. Identical to d12_common.safety_filter except that the barrier radius of a
pair is dsafe + delay_s * (|u_i| + |u_j|). With delay_s = 0 it must reproduce safety_filter exactly (self-test)."""
import numpy as np

def safety_filter_vm(u, p, dsafe, gamma, sense, delay_s):
    from d12_common import _project
    D = p[:, None] - p[None]; d2 = (D ** 2).sum(-1); out = u.copy(); sp = np.linalg.norm(u, axis=1)
    for i in range(len(p)):
        js = [j for j in range(len(p)) if j != i and d2[i, j] < sense ** 2]
        if js:
            r = dsafe + delay_s * (sp[i] + sp[js])
            out[i] = _project(u[i], 2 * D[i, js], -0.5 * gamma * (d2[i, js] - r ** 2))
    return out
