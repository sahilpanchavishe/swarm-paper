"""Day 15: start configurations with a chosen number of communication components and a chosen observer placement.
Rejection-samples the existing S.random_start, so the geometry is the same family as the old runs.
kind: conn1 (1 component, observers random) | comp2_same (2 components, all observers in one component)
      comp2_split (2 components, >=1 observer in each of two components) | comp3_same (3 components, all observers in one)
Deterministic in `seed`. Observers: k = 2 (25%) or 4 (50%)."""
import numpy as np
import swarm_sim as S
from d4_common import N, R_COMM

KINDS = {"conn1": (1, "random"), "comp2_same": (2, "same"), "comp2_split": (2, "split"), "comp3_same": (3, "same")}

def labels(A):
    lab = -np.ones(len(A), int); c = 0
    for s in range(len(A)):
        if lab[s] >= 0: continue
        lab[s] = c; stack = [s]
        while stack:
            i = stack.pop()
            for j in np.nonzero(A[i] & (lab < 0))[0]: lab[j] = c; stack.append(j)
        c += 1
    return lab, c

def make_scenario(seed, kind, obs_pct, half=0.9, max_tries=20000):
    n_comp, mode = KINDS[kind]; k = {50: 4, 25: 2}[obs_pct]
    rng = np.random.default_rng([seed, 1234]); orng = np.random.default_rng([seed, 99])
    for _ in range(max_tries):
        p, th = S.random_start(rng, half=half)
        lab, c = labels(S.comm_adjacency(p, R_COMM))
        if c != n_comp: continue
        sizes = np.bincount(lab); obs = np.zeros(N, bool)
        if mode == "random":
            obs[orng.choice(N, k, replace=False)] = True
        elif mode == "same":
            ok = [g for g in range(c) if sizes[g] >= k]
            if not ok: continue
            g = ok[orng.integers(len(ok))]; obs[orng.choice(np.nonzero(lab == g)[0], k, replace=False)] = True
        else:  # split: one observer in each of the first two components, the rest anywhere
            a, b = orng.choice(c, 2, replace=False)
            obs[orng.choice(np.nonzero(lab == a)[0])] = True; obs[orng.choice(np.nonzero(lab == b)[0])] = True
            rest = np.nonzero(~obs)[0]; obs[orng.choice(rest, k - 2, replace=False)] = True
        return p, th, obs
    raise RuntimeError(f"no {kind} start found in {max_tries} tries (half={half})")
