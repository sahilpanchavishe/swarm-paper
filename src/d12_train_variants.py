"""Day 12b: train a capacity-control model (behaviour cloning, then DAgger) with EXACTLY the Day 4 / Day 5 recipe and seed discipline.
  python d12_train_variants.py --arch wide1   --seed 0     -> runs/d12_wide1_s0.pt     (K=1, width raised to K=2's parameter count)
  python d12_train_variants.py --arch shared2 --seed 0     -> runs/d12_shared2_s0.pt   (K=2, one set of weights used in both layers)
  python d12_train_variants.py --arch mlpwide --seed 0     -> runs/d12_mlpwide_s0.pt   (K=0, no communication, width raised to K=2's count)
Same data (data/raw/d4_demos.npz, same train/val split), BC 30 epochs lr 2e-3, DAgger 3 rounds x 240 episodes x 10 epochs lr 5e-4, rollout seeds 1300-4299.
No dev checks here (they only logged numbers); the final evaluation is `python d12_eval.py capacity`."""
import argparse, os, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, torch, torch.nn.functional as Fn
import swarm_sim as S
from d4_common import oracle_label, gesture_onehot, FORM_ID, T_STEPS, START_HALF, V_MAX, N
from d12_models import make_model, ARCHS, n_params
from mj_env import SwarmEnv

ROOT = Path(__file__).resolve().parent.parent
SUB = 4
_env = []

def rollout(args):
    """The learner drives (memoryless: only observers know the gesture); returns subsampled positions, oracle labels and know flags."""
    sd, arch, K, f, seed, obs_pct = args
    torch.set_num_threads(1)
    if not _env: _env.append(SwarmEnv())
    env = _env[0]; model = make_model(arch, K); model.load_state_dict(sd); model.eval()
    rng = np.random.default_rng(seed); p, th = S.random_start(rng, half=START_HALF)
    obs = np.zeros(N, bool); obs[np.random.default_rng([seed, 99]).choice(N, {100: 8, 50: 4, 25: 2}[obs_pct], replace=False)] = True
    env.reset(p, th); p, th = env.poses(); g = torch.tensor(gesture_onehot(FORM_ID[f]))[None]; P, Y, Kn = [], [], []
    with torch.no_grad():
        for t in range(T_STEPS):
            if t % SUB == 0: P.append(p.copy()); Y.append(oracle_label(p, f)); Kn.append(obs.astype(np.float32))
            u = model(torch.tensor(p, dtype=torch.float32)[None], g, torch.tensor(obs, dtype=torch.float32)[None])[0].numpy() * V_MAX
            v, w = S.to_unicycle(u, th); p, th = env.step(v, w)
    return dict(P=np.array(P, np.float32), Y=np.array(Y, np.float32), K=np.array(Kn, np.float32), g=gesture_onehot(FORM_ID[f]))

def fit(model, data, epochs, lr, bs, tag, t0):
    opt = torch.optim.Adam(model.parameters(), lr=lr); steps = epochs * (len(data[0]) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps); model.train(); last = float("nan")
    for ep in range(epochs):
        idx = torch.randperm(len(data[0])); tot = nb = 0
        for i in range(0, len(idx) - bs + 1, bs):
            b = idx[i:i + bs]
            loss = Fn.smooth_l1_loss(model(data[0][b], data[1][b], data[2][b]), data[3][b], beta=0.1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item(); nb += 1
        last = tot / nb
        if ep % 5 == 0 or ep == epochs - 1: print(f"  {tag} epoch {ep+1:2d}/{epochs}  train {last:.5f}  ({time.time()-t0:.0f} s)", flush=True)
    model.eval(); return last

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arch", required=True, choices=list(ARCHS)); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--eps", type=int, default=240)
    ap.add_argument("--dagger-epochs", type=int, default=10); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args(); assert a.eps <= 600 and a.rounds <= 5, "keeps DAgger seeds inside 1300-4299"
    spec = ARCHS[a.arch]; K = spec["K"]; name = f"d12_{a.arch}_s{a.seed}"; (ROOT / "runs").mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    d = np.load(ROOT / "data" / "raw" / "d4_demos.npz"); P, Y, OBS, F = d["P"], d["Y"], d["obs"], d["f"]; E, T = P.shape[:2]
    perm = np.random.default_rng(1234).permutation(E); nval = max(1, E // 10); tr_e = perm[nval:]
    k = np.repeat(OBS[tr_e][:, None, :], T, axis=1).astype(np.float32)
    data = [torch.tensor(P[tr_e].reshape(-1, N, 2)), torch.tensor(np.eye(4, dtype=np.float32)[np.repeat(F[tr_e], T)]), torch.tensor(k.reshape(-1, N)), torch.tensor(Y[tr_e].reshape(-1, N, 2))]
    torch.manual_seed(a.seed); np.random.seed(a.seed); model = make_model(spec["arch"], K)
    print(f"{name}: arch={spec['arch']} K={K} parameters={n_params(model)} | {len(tr_e)} demo episodes, {len(data[0])} samples", flush=True); t0 = time.time()
    bc_loss = fit(model, data, a.epochs, 2e-3, a.bs, "BC", t0); torch.save(model.state_dict(), ROOT / "runs" / f"d11bc_{a.arch}_s{a.seed}.pt")
    torch.manual_seed(a.seed); np.random.seed(a.seed); log = [dict(model=name, stage="BC", round=0, train_loss=bc_loss, dataset=len(data[0]))]
    for r in range(1, a.rounds + 1):
        sd = {kk: v.cpu() for kk, v in model.state_dict().items()}; t1 = time.time()
        jobs = [(sd, spec["arch"], K, S.FORMATIONS[i % 4], 1300 + (r - 1) * 600 + i, [100, 50, 25][(i // 4) % 3]) for i in range(a.eps)]
        print(f"round {r}: rolling out the learner on {len(jobs)} training episodes ...", flush=True)
        with Pool(a.workers) as pool: ro = pool.map(rollout, jobs, chunksize=2)
        new = [torch.tensor(np.concatenate([x["P"] for x in ro])), torch.tensor(np.concatenate([np.repeat(x["g"][None], len(x["P"]), 0) for x in ro])),
               torch.tensor(np.concatenate([x["K"] for x in ro])), torch.tensor(np.concatenate([x["Y"] for x in ro]))]
        data = [torch.cat([o, n]) for o, n in zip(data, new)]; print(f"  +{len(new[0])} learner states, dataset {len(data[0])}  ({time.time()-t1:.0f} s)", flush=True)
        loss = fit(model, data, a.dagger_epochs, 5e-4, a.bs, f"DAgger r{r}", t0); log.append(dict(model=name, stage="DAgger", round=r, train_loss=loss, dataset=len(data[0])))
        torch.save(model.state_dict(), ROOT / "runs" / f"d11part_{a.arch}_s{a.seed}.pt")      # partial checkpoint; the final file below marks 'finished'
    torch.save(model.state_dict(), ROOT / "runs" / f"{name}.pt")
    lp = ROOT / "results" / "d12_train_log.csv"; pd.DataFrame(log).to_csv(lp, mode="a", index=False, header=not lp.exists())
    print(f"\nsaved runs/{name}.pt  (total {(time.time()-t0)/60:.0f} min)")

if __name__ == "__main__":        # required on macOS and Windows (workers re-import this file)
    main()
