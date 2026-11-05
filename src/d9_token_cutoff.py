"""Day 9b: does the memoryless GNN use its own POSITION as an implicit memory of the gesture? EXPLORATORY (post hoc, say so in the paper).

Open question in the Discussion draft: why does the memoryless GNN beat the memoryless relay at 25 % observability? One explanation: near its slot, a robot's
position already hints which formation is being built, so the network can keep going after the token has gone. The Day 7 ablation removed the token for the whole
episode, so it cannot tell. Here the token is TRUE until control step t_cut and then BLANK (observers keep their "I see a gesture" flag but the token is all zeros,
exactly the Day 7 "blank" mode; variant "flagon"), or, in the cleaner variant "flagoff", the flag is switched off for EVERY robot too, which is exactly the input an
uninformed robot sees (the in-distribution "nobody told me anything" state). Same test seeds, starts and observers as everywhere else (10000+), 4 formations, 25 % observability by default.
t_cut = 0 reproduces the Day 7 blank mode (expect about 0 % success); t_cut = 1100 is the normal run (the control).
Per cut-off we report: success at the end (against the TRUE formation), the formation error at the cut and at the end, and how often the error got WORSE after the cut.
Reading it:
  success stays high once t_cut is late (formation mostly built)  -> position carries the formation (implicit memory), the network does not need the live token to finish.
  success falls back toward 0 whatever t_cut                       -> the policy needs the live token all the way; no implicit memory.
  late cut-offs fine but early cut-offs fail                       -> position helps only once the robots are already close to their slots.
Run:  python d9_token_cutoff.py [--n 20] [--models d5_K2_s0 d5_K2_s1 ...] [--cuts 0 300 700 1100] [--pcts 25] [--variants flagon flagoff]     -> ../results/d9_token_cutoff.csv
(default 5 models x 80 episodes x 4 cut-offs x 2 variants, about 3,000 episodes, roughly 20 to 30 minutes on 8 cores)"""
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
    sd, K, latched, f, seed, pct, t_cut, variant, name = args
    torch.set_num_threads(1)
    if not _env: _env.append(SwarmEnv())
    env = _env[0]; model = GNN(K, latched); model.load_state_dict(sd); model.eval()
    rng = np.random.default_rng(seed); p, th = S.random_start(rng, half=START_HALF); tg = S.slots(f)
    obs = np.zeros(N, bool); obs[np.random.default_rng([seed, 99]).choice(N, {100: 8, 50: 4, 25: 2}[pct], replace=False)] = True
    env.reset(p, th); p, th = env.poses(); latch = obs.copy()
    g_true = torch.tensor(gesture_onehot(FORM_ID[f]))[None]; g_blank = torch.zeros(1, 4); err_cut = np.nan
    with torch.no_grad():
        for t in range(T_STEPS):
            if t == t_cut: err_cut = S.formation_error(p, tg)
            if latched: latch = reach(S.comm_adjacency(p, R_COMM), latch, max(K, 1)); know = latch
            else: know = obs
            cut = t >= t_cut
            g = g_blank if cut else g_true
            kn = np.zeros(N, bool) if (cut and variant == "flagoff") else know
            u = model(torch.tensor(p, dtype=torch.float32)[None], g, torch.tensor(kn, dtype=torch.float32)[None])[0].numpy() * V_MAX
            v, w = S.to_unicycle(u, th); p, th = env.step(v, w)
    fin = S.formation_error(p, tg)
    if np.isnan(err_cut): err_cut = fin                                  # t_cut >= T_STEPS: token never removed
    m = re.match(r"d5_K(\d)(_latched)?_s(\d+)$", name)
    return dict(model=name, K=K, latched=latched, train_seed=int(m.group(3)), obs_pct=pct, formation=f, seed=seed, t_cut=t_cut, variant=variant,
                err_at_cut=err_cut, final_err=fin, success=bool(fin < 0.025), success_05=bool(fin < 0.05), worse_after_cut=bool(fin > err_cut + 0.01))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--models", nargs="+", default=["d5_K1_s0", "d5_K2_s0", "d5_K2_s1", "d5_K2_s2", "d5_K3_s0"])
    ap.add_argument("--cuts", nargs="+", type=int, default=[0, 300, 700, 1100]); ap.add_argument("--pcts", nargs="+", type=int, default=[25])
    ap.add_argument("--variants", nargs="+", default=["flagon", "flagoff"], choices=["flagon", "flagoff"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--out", default=str(ROOT / "results" / "d9_token_cutoff.csv"))
    a = ap.parse_args(); rows = []
    for name in a.models:
        m = re.match(r"d5_K(\d)(_latched)?_s(\d+)$", name); K, lat = int(m.group(1)), bool(m.group(2))
        sd = torch.load(ROOT / "runs" / f"{name}.pt", map_location="cpu")
        jobs = [(sd, K, lat, f, 10000 + s, pct, tc, v, name) for pct in a.pcts for f in S.FORMATIONS for s in range(a.n) for tc in a.cuts for v in a.variants
                if not (tc >= T_STEPS and v != a.variants[0])]          # a never-removed token is the same run in both variants: do it once
        t0 = time.time(); print(f"{name}: {len(jobs)} episodes ...", flush=True)
        with Pool(a.workers) as pool: rows += pool.map(episode, jobs, chunksize=4)
        print(f"  {time.time()-t0:.0f} s", flush=True)
    df = pd.DataFrame(rows); out = Path(a.out); out.parent.mkdir(exist_ok=True); df.to_csv(out, index=False)
    df["policy"] = df.apply(lambda r: f"GNN K{r.K}" + ("+latched" if r.latched else ""), axis=1)
    pd.set_option("display.width", 220)
    g = df.groupby(["policy", "obs_pct", "variant", "t_cut"]).agg(train_seeds=("train_seed", "nunique"), n=("success", "size"), success_pct=("success", lambda s: 100 * s.mean()),
                                                      err_at_cut_m=("err_at_cut", "mean"), final_err_m=("final_err", "mean"), worse_after_cut_pct=("worse_after_cut", lambda s: 100 * s.mean())).round(3)
    print("\n=== token removed at control step t_cut (0.032 s per step); success against the TRUE formation ===\n", g.to_string())
    print(f"\nsaved {out}")

if __name__ == "__main__":        # required on macOS (workers re-import this file)
    main()
