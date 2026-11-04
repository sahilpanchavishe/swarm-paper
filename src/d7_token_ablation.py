"""Day 7a: does the GNN really use the gesture token that is passed to it? (answers the "formation cues from positions?" question)
Three modes, same episodes (test seeds 10000+), same observers:
  normal : true gesture token to the observers (control)
  blank  : observers receive an all-zero token (gesture removed)
  wrong  : observers receive the token of a DIFFERENT formation than the one used for scoring "true"
Per mode we report: success against the TRUE formation, success against the SHOWN (wrong) formation, and the share of NON-observers
that end within 0.05 m of their slot in the SHOWN formation (= the token reached them through message passing and they obeyed it).
Reading it:  blank -> success collapses  => the policy needs the token (it cannot guess the shape from positions).
             wrong -> success_vs_shown high, also for non-observers => the token really travels through the graph and drives behaviour.
Run: python d7_token_ablation.py [--n 20] [--models d5_K1_s0 d5_K2_s0 d5_K3_s0 d5_K2_latched_s0]      -> ../results/d7_token_ablation.csv"""
import argparse, os, re, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, torch
import swarm_sim as S
from d4_common import GNN, reach, gesture_onehot, FORM_ID, T_STEPS, START_HALF, R_COMM, V_MAX, N
from mj_env import SwarmEnv

ROOT = Path(__file__).resolve().parent.parent
_env = []

def episode(args):
    sd, K, latched, f_true, f_shown, seed, pct, mode, name = args
    torch.set_num_threads(1)
    if not _env: _env.append(SwarmEnv())
    env = _env[0]; model = GNN(K, latched); model.load_state_dict(sd); model.eval()
    rng = np.random.default_rng(seed); p, th = S.random_start(rng, half=START_HALF)
    obs = np.zeros(N, bool); obs[np.random.default_rng([seed, 99]).choice(N, {100: 8, 50: 4, 25: 2}[pct], replace=False)] = True
    env.reset(p, th); p, th = env.poses(); latch = obs.copy()
    g = torch.zeros(1, 4) if mode == "blank" else torch.tensor(gesture_onehot(FORM_ID[f_shown if mode == "wrong" else f_true]))[None]
    with torch.no_grad():
        for _ in range(T_STEPS):
            if latched: latch = reach(S.comm_adjacency(p, R_COMM), latch, max(K, 1)); know = latch
            else: know = obs
            u = model(torch.tensor(p, dtype=torch.float32)[None], g, torch.tensor(know, dtype=torch.float32)[None])[0].numpy() * V_MAX
            v, w = S.to_unicycle(u, th); p, th = env.step(v, w)
    e_true = np.linalg.norm(p - S.slots(f_true), axis=1); e_shown = np.linalg.norm(p - S.slots(f_shown), axis=1)
    no = ~obs
    return dict(model=name, mode=mode, obs_pct=pct, formation=f_true, shown=f_shown, seed=seed, success_true=bool(e_true.mean() < 0.025),
                success_shown=bool(e_shown.mean() < 0.025), nonobs_follow_shown=float((e_shown[no] < 0.05).mean()) if no.any() else np.nan,
                obs_follow_shown=float((e_shown[obs] < 0.05).mean()))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--models", nargs="+", default=["d5_K1_s0", "d5_K2_s0", "d5_K3_s0", "d5_K2_latched_s0"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--out", default=str(ROOT / "results" / "d7_token_ablation.csv"))
    a = ap.parse_args(); rows = []
    for name in a.models:
        m = re.match(r"d5_K(\d)(_latched)?_s(\d+)$", name); K, lat = int(m.group(1)), bool(m.group(2))
        sd = torch.load(ROOT / "runs" / f"{name}.pt", map_location="cpu")
        jobs = []
        for pct in (100, 25):
            for fi, f in enumerate(S.FORMATIONS):
                for s in range(a.n):
                    f_shown = S.FORMATIONS[(fi + 1 + (s % 3)) % 4]                  # always a different formation
                    for mode in ("normal", "blank", "wrong"): jobs.append((sd, K, lat, f, f_shown, 10000 + s, pct, mode, name))
        t0 = time.time(); print(f"{name}: {len(jobs)} episodes ...", flush=True)
        with Pool(a.workers) as pool: rows += pool.map(episode, jobs, chunksize=4)
        print(f"  {time.time()-t0:.0f} s", flush=True)
    df = pd.DataFrame(rows); out = Path(a.out); out.parent.mkdir(exist_ok=True); df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    g = df.groupby(["model", "obs_pct", "mode"]).agg(success_true_pct=("success_true", lambda s: 100 * s.mean()), success_shown_pct=("success_shown", lambda s: 100 * s.mean()),
                                                      nonobs_follow_shown_pct=("nonobs_follow_shown", lambda s: 100 * s.mean()), n=("success_true", "size")).round(1)
    print("\n", g.to_string()); print(f"\nsaved {out}")

if __name__ == "__main__":
    main()
