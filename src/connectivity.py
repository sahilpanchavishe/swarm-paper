"""Day 2c: communication-graph analysis (pure NumPy, about a minute).
Answers: (1) how often is the 8-robot graph connected at the start, for different start-box sizes?
(2) what fraction of robots can a token reach in K hops from 2 or 4 observers? (3) is each final formation connected?
Run: python connectivity.py     Writes ../results/connectivity_*.csv and ../figures/connectivity.png"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import swarm_sim as S

ROOT = Path(__file__).resolve().parent.parent
N_EP = 2000; KMAX = 6; HALVES = (0.9, 0.7, 0.5, 0.4)

def components(A):
    n = len(A); seen = -np.ones(n, int); c = 0
    for s in range(n):
        if seen[s] >= 0: continue
        stack = [s]; seen[s] = c
        while stack:
            u = stack.pop()
            for v in np.where(A[u])[0]:
                if seen[v] < 0: seen[v] = c; stack.append(v)
        c += 1
    return c

def reach(A, obs, K):
    R = obs.copy(); cur = obs.copy()
    for _ in range(K): cur = (A.astype(int) @ cur.astype(int)) > 0; R = R | cur
    return R

def hop_diameter(A):
    n = len(A); D = np.full((n, n), 99); np.fill_diagonal(D, 0); M = np.eye(n, dtype=bool)
    for k in range(1, n):
        M = M | ((A.astype(int) @ M.astype(int)) > 0); D[M & (D == 99)] = k
    return D.max()

def main():
    (ROOT / "results").mkdir(exist_ok=True); (ROOT / "figures").mkdir(exist_ok=True)
    radii = np.round(np.arange(0.30, 0.81, 0.05), 2)
    rows_c, rows_t = [], []
    for h in HALVES:
        starts = [S.random_start(np.random.default_rng(10000 + s), half=h) for s in range(N_EP)]
        for r in radii:
            rows_c.append(dict(start_half=h, R_comm=r, pct_connected=100 * np.mean([components(S.comm_adjacency(p, r)) == 1 for p, _ in starts])))
        for n_obs in (2, 4):
            vals = np.zeros((N_EP, KMAX + 1))
            for s, (p, _) in enumerate(starts):
                rng = np.random.default_rng(20000 + s); obs = np.zeros(S.N, bool); obs[rng.choice(S.N, n_obs, replace=False)] = True
                A = S.comm_adjacency(p, S.R_COMM)
                for K in range(KMAX + 1): vals[s, K] = reach(A, obs, K).mean()
            for K in range(KMAX + 1):
                rows_t.append(dict(start_half=h, observers=n_obs, K=K, mean_pct_reached=100 * vals[:, K].mean(),
                                   pct_episodes_all_reached=100 * np.mean(vals[:, K] == 1.0)))
    dfc, dft = pd.DataFrame(rows_c), pd.DataFrame(rows_t)
    dfc.to_csv(ROOT / "results" / "connectivity_vs_range.csv", index=False)
    dft.to_csv(ROOT / "results" / "token_coverage_vs_K.csv", index=False)
    rows = []
    for f in S.FORMATIONS:
        A = S.comm_adjacency(S.slots(f), S.R_COMM); c = components(A)
        rows.append(dict(formation=f, components=c, hop_diameter=hop_diameter(A) if c == 1 else np.nan, mean_degree=A.sum(1).mean()))
    dff = pd.DataFrame(rows); dff.to_csv(ROOT / "results" / "connectivity_formations.csv", index=False)

    pd.set_option("display.width", 220)
    print("(1) % of random starts with a connected graph at R_comm = 0.5 m, by start-box half-width:")
    print(dfc[dfc.R_comm == 0.5][["start_half", "pct_connected"]].round(1).to_string(index=False))
    print("\n(2) % of robots reached by the token after K hops (R_comm = 0.5 m):")
    print(dft[dft.K.isin([0, 1, 2, 3, 6])].pivot_table(index=["start_half", "observers"], columns="K", values="mean_pct_reached").round(1).to_string())
    print("\n(3) final formations at R_comm = 0.5 m:\n", dff.round(2).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for h in (0.9, 0.5):
        d = dfc[dfc.start_half == h]; ax[0].plot(d.R_comm, d.pct_connected, "o-", label=f"start box {2*h:.1f} m x {2*h:.1f} m")
    ax[0].axvline(0.5, color="k", ls="--"); ax[0].set_xlabel("communication range R_comm (m)"); ax[0].set_ylabel("random starts connected (%)")
    ax[0].set_title("Connectivity at the start"); ax[0].legend()
    for h, ls in ((0.9, "--"), (0.5, "-")):
        for n_obs, col in ((2, "C3"), (4, "C2")):
            d = dft[(dft.start_half == h) & (dft.observers == n_obs)]
            ax[1].plot(d.K, d.mean_pct_reached, ls, marker="o", color=col, label=f"{n_obs} observers, box {2*h:.1f} m")
    ax[1].set_xlabel("hops K"); ax[1].set_ylabel("robots reached by the token (%)"); ax[1].set_title("Token coverage after K hops (start)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "connectivity.png", dpi=300)
    print("\nsaved results/connectivity_*.csv, results/token_coverage_vs_K.csv and figures/connectivity.png")

if __name__ == "__main__":
    main()
