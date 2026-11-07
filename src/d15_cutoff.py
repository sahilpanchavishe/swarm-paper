"""Day 15 Study C: token cut-off redo (CONFIRMATORY, protocol addendum 4b). Same mechanics as d9_token_cutoff.py: the token is TRUE until control step t_cut,
then blank. Variant flagon keeps the observers' flag; flagoff switches the flag off for every robot (the in-distribution 'uninformed' input).
Fresh seed block, plus position-only readouts recorded in every episode:
  near_cut_ok / near_final(_ok): nearest-template classifier (which formation's slots the swarm is closest to) at the cut and at the end
  x0..x7, y0..y7: swarm positions at the cut (for an optional learned probe later, exploratory)
Run (from src/): python d15_cutoff.py --models d5_K2_s0 ... --cuts 0 100 ... --variants flagoff flagon --out ../results/d15_cutoff_primary.csv [--n 20] [--seed0 60000]"""
import argparse, os, re, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, torch
import swarm_sim as S
from d4_common import GNN, reach, gesture_onehot, FORM_ID, T_STEPS, START_HALF, R_COMM, V_MAX, N
from mj_env import SwarmEnv
ROOT = Path(__file__).resolve().parent.parent
FORMS = list(S.FORMATIONS)
_env = []

def nearest(p):
    return int(np.argmin([S.formation_error(p, S.slots(f)) for f in FORMS]))

def episode(args):
    sd, K, latched, f, seed, pct, t_cut, variant, name = args
    torch.set_num_threads(1)
    if not _env: _env.append(SwarmEnv())
    env = _env[0]; model = GNN(K, latched); model.load_state_dict(sd); model.eval()
    rng = np.random.default_rng(seed); p, th = S.random_start(rng, half=START_HALF); tg = S.slots(f)
    obs = np.zeros(N, bool); obs[np.random.default_rng([seed, 99]).choice(N, {100: 8, 50: 4, 25: 2}[pct], replace=False)] = True
    env.reset(p, th); p, th = env.poses(); latch = obs.copy()
    g_true = torch.tensor(gesture_onehot(FORM_ID[f]))[None]; g_blank = torch.zeros(1, 4)
    err_cut = np.nan; near_cut = -1; pos_cut = None
    with torch.no_grad():
        for t in range(T_STEPS):
            if t == t_cut: err_cut = S.formation_error(p, tg); near_cut = nearest(p); pos_cut = np.array(p, float).copy()
            if latched: latch = reach(S.comm_adjacency(p, R_COMM), latch, max(K, 1)); know = latch
            else: know = obs
            cut = t >= t_cut
            g = g_blank if cut else g_true
            kn = np.zeros(N, bool) if (cut and variant == "flagoff") else know
            u = model(torch.tensor(p, dtype=torch.float32)[None], g, torch.tensor(kn, dtype=torch.float32)[None])[0].numpy() * V_MAX
            v, w = S.to_unicycle(u, th); p, th = env.step(v, w)
    fin = S.formation_error(p, tg); nf = nearest(p)
    if np.isnan(err_cut): err_cut = fin; near_cut = nf; pos_cut = np.array(p, float).copy()      # t_cut >= T_STEPS: token never removed
    m = re.match(r"d5_K(\d)(_latched)?_s(\d+)$", name); ti = FORMS.index(f)
    row = dict(model=name, K=K, latched=latched, train_seed=int(m.group(3)), obs_pct=pct, formation=f, seed=seed, t_cut=t_cut, variant=variant,
               err_at_cut=err_cut, final_err=fin, success=bool(fin < 0.025), near_cut_ok=bool(near_cut == ti), near_final=FORMS[nf], near_final_ok=bool(nf == ti))
    for i in range(N): row[f"x{i}"] = pos_cut[i, 0]; row[f"y{i}"] = pos_cut[i, 1]
    return row

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed0", type=int, default=60000)
    ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--cuts", nargs="+", type=int, required=True)
    ap.add_argument("--pcts", nargs="+", type=int, default=[25]); ap.add_argument("--variants", nargs="+", default=["flagoff", "flagon"], choices=["flagon", "flagoff"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--out", required=True)
    a = ap.parse_args(); rows = []
    for name in a.models:
        m = re.match(r"d5_K(\d)(_latched)?_s(\d+)$", name); K, lat = int(m.group(1)), bool(m.group(2))
        sd = torch.load(ROOT / "runs" / f"{name}.pt", map_location="cpu")
        jobs = [(sd, K, lat, f, a.seed0 + s, pct, tc, v, name) for pct in a.pcts for f in S.FORMATIONS for s in range(a.n) for tc in a.cuts for v in a.variants
                if not (tc >= T_STEPS and v != a.variants[0])]
        t0 = time.time(); print(f"{name}: {len(jobs)} episodes ...", flush=True)
        with Pool(a.workers) as pool: rows += pool.map(episode, jobs, chunksize=4)
        print(f"  {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows); out = Path(a.out); out.parent.mkdir(exist_ok=True); df.to_csv(out, index=False); print(f"saved {out}")

if __name__ == "__main__":        # required on macOS
    main()
