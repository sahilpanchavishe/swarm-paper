"""Day 9a: a FAIRER no-learning baseline. EXPLORATORY (added after Day 6; not in the frozen protocol, say so in the paper).

Why: Day 6 showed the K=1 GNN beating the K=1 relay by 33 points at 25 % observability. A memoryless K-layer GNN can only know the gesture within K hops
of an observer, exactly like the memoryless relay of Day 3, so that gap cannot come from better communication. The difference is what uninformed
robots DO: the Day 3 relay makes them hold still, the GNN does not. Here an uninformed robot instead heads to the AVERAGE of the four candidate slots
of its own role (the best guess without the gesture, and what an MSE-trained network drifts towards). Informed robots go to their true slot as before.
Same starts, observers and seeds as Day 3 and Day 6: test seeds 10000..10000+n-1 x 4 formations x observability 50/25 %, T = 1100 steps, MuJoCo.

Policies: "relayD K<k>", k = 0 (only observers know), 1, 2, 3, 7 (7 = unlimited hops). All memoryless.
Outputs: ../results/d9_relay_default.csv (one row per episode), ../results/d9_relay_default_paired.csv, and a printed table.
The paired comparisons use d6_analysis.py's own functions (Wilcoxon, bootstrap CI) with their own Holm family; they are POST HOC and must be reported as exploratory.
Run:  python d9_relay_default.py [seeds_per_formation=50] [workers=cores-1]        (about 10 minutes on 8 cores for n = 50)"""
import sys, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
import swarm_sim as S
from mj_env import SwarmEnv, ROBOT_R
from d3_baselines import reach, T_STEPS, START_HALF, SEED0, OBS_LEVELS

ROOT = Path(__file__).resolve().parent.parent
KS = (0, 1, 2, 3, 7)
DEFAULT = np.mean([S.slots(f) for f in S.FORMATIONS], axis=0)          # (8, 2): mean over the four formations of each role's slot
_env = []

def episode(args):
    K, pct, f, seed = args
    if not _env: _env.append(SwarmEnv())
    env = _env[0]
    rng = np.random.default_rng(seed); p, th = S.random_start(rng, half=START_HALF); tg = S.slots(f)
    obs = np.zeros(S.N, bool); obs[np.random.default_rng([seed, 99]).choice(S.N, OBS_LEVELS[pct], replace=False)] = True
    env.reset(p, th); p, th = env.poses(); errs = []; minc = 9.0
    for _ in range(T_STEPS):
        know = obs if K == 0 else reach(S.comm_adjacency(p, S.R_COMM), obs, K)
        targets = np.where(know[:, None], tg, DEFAULT)
        v, w = S.to_unicycle(S.expert_velocity_vector(p, targets), th); p, th = env.step(v, w)
        errs.append(S.formation_error(p, tg)); minc = min(minc, S.min_clearance(p))
    return dict(policy=f"relayD K{K}", obs_pct=pct, formation=f, seed=seed, final_err=errs[-1], success=bool(errs[-1] < 0.025),
                success_05=bool(errs[-1] < 0.05), collision=bool(minc < 2 * ROBOT_R - 1e-9), min_clearance=minc)

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 1)
    jobs = [(K, pct, f, SEED0 + s) for K in KS for pct in (50, 25) for f in S.FORMATIONS for s in range(n)]
    print(f"{len(jobs)} episodes on {workers} workers ...", flush=True)
    with Pool(workers) as pool: rows = pool.map(episode, jobs, chunksize=8)
    new = pd.DataFrame(rows); (ROOT / "results").mkdir(exist_ok=True); new.to_csv(ROOT / "results" / "d9_relay_default.csv", index=False)

    import d6_analysis as A                                     # same loader, paired test, bootstrap and Holm as the frozen analysis
    d = A.load(str(ROOT / "results" / "d6_main_eval.csv"))
    n2 = new.copy(); n2["train_seed"] = np.nan; n2["method"] = "no-learning"; n2["overlap5"] = n2.min_clearance < 0.069
    d = pd.concat([d, n2[d.columns]], ignore_index=True)
    pd.set_option("display.width", 220)
    print("\n=== SUCCESS % (contact % in brackets): Day 3 relay (uninformed hold still) vs relayD (uninformed head to mean slot) vs GNN (mean over training seeds) ===")
    for pct in (25, 50):
        rows_ = []
        for K in KS:
            r = {"K": K}
            for lab, pol in (("relay (hold)", f"relay K{K}"), ("relayD", f"relayD K{K}"), ("GNN", f"GNN K{K}")):
                x = d[(d.policy == pol) & (d.obs_pct == pct)]
                r[lab] = f"{100 * x.success.mean():5.1f} ({100 * x.collision.mean():4.1f})" if len(x) else "-"
            rows_.append(r)
        print(f"\n--- {pct}% observability ---"); print(pd.DataFrame(rows_).to_string(index=False))

    fam = []
    for K in (1, 2, 3):
        for pct in (25, 50):
            fam.append((f"GNN vs relayD, K={K} ({pct}% obs)", f"GNN K{K}", f"relayD K{K}", pct))
            fam.append((f"relayD vs relay (hold), K={K} ({pct}% obs)", f"relayD K{K}", f"relay K{K}", pct))
    out = [r for r in (A.compare(d, a, b, pct, lab) for lab, a, b, pct in fam) if r]
    P = pd.DataFrame(out); P["p_success_holm"] = A.holm(P.p_success); P["p_final_err_holm"] = A.holm(P.p_final_err)
    P.to_csv(ROOT / "results" / "d9_relay_default_paired.csv", index=False)
    print(f"\n=== PAIRED, EXPLORATORY (post hoc family of {len(P)}, Holm-adjusted; difference = A - B) ===")
    print(P[["comparison", "n_pairs", "dsuccess_pct", "dsuccess_ci_lo", "dsuccess_ci_hi", "p_success_holm"]].round(4).to_string(index=False))
    print("\nReading it: if relayD at K=1 is close to GNN K=1, the Day 6 gap came from what uninformed robots do, not from learning to communicate.")
    print("If relayD stays far below the GNN, the GNN adds something a hand-written relay with a sensible default does not.")

if __name__ == "__main__":        # required on macOS (workers re-import this file)
    main()
