"""Day 12: fast self-test of the new code. No MuJoCo episode is run and nothing is trained (takes a few seconds).
  python d12_selftest.py
Checks: parameter-count formula and the capacity-control sizes; per-robot-token forward == original GNN.forward; token registers == Day 3 'reach'
flooding in the static case, and stale vs fresh behaviour after a command change; the safety filter satisfies its constraints and is a no-op when far."""
import numpy as np, torch
import swarm_sim as S
from d4_common import GNN, reach, N, R_COMM
from d12_common import update_registers, gnn_forward, safety_filter, n_components, SL
from d12_models import count, n_params, wide_hid, make_model, GNNShared

ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

print("1. parameter counts")
for K in (0, 1, 2, 3): check(f"formula == real count, K={K}", count(K, 64) == n_params(GNN(K)), f"({count(K, 64)})")
tgt = n_params(GNN(2))
for K, nm in ((1, "wide1"), (0, "mlpwide")):
    h = wide_hid(K); m = make_model("wide", K); print(f"     {nm}: hidden width {h}, {n_params(m)} parameters (K=2 base has {tgt}, K=1 base has {n_params(GNN(1))})")
    check(f"{nm} within 1 % of K=2", abs(n_params(m) - tgt) / tgt < 0.01)
sh = make_model("shared", 2); check("shared2 has the parameter count of K=1", n_params(sh) == n_params(GNN(1)), f"({n_params(sh)})")

print("2. forward equivalence (per-robot tokens, adjacency override)")
torch.manual_seed(0)
for arch, K in (("base", 0), ("base", 1), ("base", 2), ("base", 3), ("wide", 1), ("shared", 2)):
    m = make_model(arch, K).eval(); B = 5; P = torch.rand(B, N, 2) * 0.8 - 0.4; g = torch.eye(4)[torch.randint(0, 4, (B,))]; kn = (torch.rand(B, N) < 0.4).float()
    ref = m(P, g, kn); tok = g[:, None, :] * kn[:, :, None]
    D = (P[:, :, None] - P[:, None]).norm(dim=-1); A = ((D <= R_COMM) & (D > 0)).float()
    check(f"{arch} K={K}: gnn_forward == GNN.forward", torch.allclose(gnn_forward(m, P, tok, kn), ref, atol=1e-6) and torch.allclose(gnn_forward(m, P, tok, kn, A), ref, atol=1e-6))

print("3. token registers")
rng = np.random.default_rng(1); mism = 0
for trial in range(300):
    p = rng.uniform(-0.6, 0.6, (N, 2)); A = S.comm_adjacency(p, R_COMM); obs = np.zeros(N, bool); obs[rng.choice(N, rng.integers(1, 4), replace=False)] = True
    for K in (1, 2, 3, 7):
        tok, st = np.full(N, -1), np.full(N, -1); know = obs.copy()
        for _ in range(3):
            tok, st = update_registers(tok, st, obs, 2, 0, A, K, "fresh"); know = reach(A, know, K); mism += int(((st >= 0) != know).any())
        tok2, st2 = update_registers(np.full(N, -1), np.full(N, -1), obs, 2, 0, A, K, "naive"); mism += int(((st2 >= 0) != reach(A, obs, K)).any())
check("flooding with memory == Day 3 reach() (fresh and naive, static)", mism == 0, f"(mismatches {mism})")
# command change on a chain of 5 robots, observer at one end, 1 hop per step
A = np.zeros((N, N), bool)
for i in range(4): A[i, i + 1] = A[i + 1, i] = True
obs = np.zeros(N, bool); obs[0] = True
for mode in ("fresh", "naive"):
    tok, st = np.full(N, -1), np.full(N, -1)
    for _ in range(6): tok, st = update_registers(tok, st, obs, 1, 0, A, 1, mode)          # command 1, stamp 0
    for _ in range(6): tok, st = update_registers(tok, st, obs, 3, 1, A, 1, mode)          # command 3, stamp 1
    check(f"after a change the chain holds {'the NEW' if mode == 'fresh' else 'the OLD'} token ({mode})", (tok[:5] == (3 if mode == "fresh" else 1) * np.ones(5)).all() or (mode == "naive" and tok[0] == 3 and (tok[1:5] == 1).all()), f"tokens {tok[:5]}")

print("4. safety filter")
u = np.array([[0.1, 0.0], [-0.1, 0.0]]); p = np.array([[0.0, 0.0], [0.1, 0.0]]); f = safety_filter(u, p)
check("head-on pair at 0.10 m is slowed to the barrier limit (0.0475 m/s each)", np.allclose(f[:, 0], [0.0475, -0.0475], atol=1e-4), f"({f[:, 0]})")
far = rng.uniform(-1, 1, (N, 2)) * 3; uu = rng.normal(0, 0.1, (N, 2)); check("no-op when nobody is within 0.30 m", np.allclose(safety_filter(uu, far), uu))
viol = 0; cnt = 0; moved = 0
for _ in range(400):
    p = rng.uniform(-0.25, 0.25, (N, 2)); uu = rng.normal(0, 0.12, (N, 2))
    if S.min_clearance(p) < 0.09: continue                              # filter's guarantee applies from a state outside the margin
    f = safety_filter(uu, p); D = p[:, None] - p[None]; d2 = (D ** 2).sum(-1); cnt += 1; moved += int(not np.allclose(f, uu))
    for i in range(N):
        for j in range(N):
            if i != j and d2[i, j] < 0.09:
                viol += int(2 * D[i, j] @ f[i] < -0.5 * 10.0 * (d2[i, j] - 0.09 ** 2) - 1e-9)
check("every pair constraint holds after filtering (400 random clusters)", viol == 0, f"(violations {viol}, filter active in {moved}/{cnt})")
check("n_components: two clusters", n_components(S.comm_adjacency(np.array([[0, 0], [0.1, 0], [2, 2], [2.1, 2.0]] + [[5 + i, 5] for i in range(4)], float), 0.5)) == 2 + 4)
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED: send me this output")
