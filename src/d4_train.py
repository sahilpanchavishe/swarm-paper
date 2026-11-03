"""Day 4b: train the GNN policy by behaviour cloning and evaluate it closed-loop in MuJoCo.
Split is BY EPISODE (90% train / 10% validation). Test seeds (10000+) are used only for the closed-loop evaluation.
Usage: python d4_train.py --K 2 [--latched] [--seed 0] [--epochs 30] [--eval 10] [--workers 7]
  --K        message-passing layers (0 = no communication, MLP only)
  --latched  ablation: add the 1-bit latched token
  --eval     closed-loop test episodes per formation per observability level (0 = skip)
Writes ../runs/d4_K{K}[_latched]_s{seed}.pt, appends ../results/d4_train_log.csv and ../results/d4_eval.csv."""
import argparse, time, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, torch, torch.nn.functional as Fn
import swarm_sim as S
from d4_common import GNN, closed_loop_episode, N

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, default=2); ap.add_argument("--latched", action="store_true"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--eval", type=int, default=10); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--data", default="d4_demos.npz")
ap.add_argument("--eval-only", action="store_true", help="skip training; load the saved model and run only the closed-loop evaluation")

def main():
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    name = f"d4_K{a.K}{'_latched' if a.latched else ''}_s{a.seed}"

    (ROOT / "runs").mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True)
    if a.eval_only:
        model = GNN(a.K, a.latched); model.load_state_dict(torch.load(ROOT / "runs" / f"{name}.pt")); model.eval()
        print(f"{name}: loaded saved model, skipping training", flush=True)
    else:
        d = np.load(ROOT / "data" / "raw" / a.data)
        P, Y, L, OBS, F = d["P"], d["Y"], d["L"], d["obs"], d["f"]
        E, T = P.shape[:2]
        perm = np.random.default_rng(1234).permutation(E); nval = max(1, E // 10); val_e, tr_e = perm[:nval], perm[nval:]   # fixed split for all models
        def know_mask(e_idx):
            """(len(e), T, N) float 'knows the gesture' flag fed to the network."""
            k = np.repeat(OBS[e_idx][:, None, :], T, axis=1)
            if a.latched and a.K >= 1: k = k | L[e_idx][..., min(a.K, 3) - 1]
            return k.astype(np.float32)
        def flat(e_idx):
            return (torch.tensor(P[e_idx].reshape(-1, N, 2)), torch.tensor(np.eye(4, dtype=np.float32)[np.repeat(F[e_idx], T)]),
                    torch.tensor(know_mask(e_idx).reshape(-1, N)), torch.tensor(Y[e_idx].reshape(-1, N, 2)))
        Ptr, Gtr, Ktr, Ytr = flat(tr_e); Pva, Gva, Kva, Yva = flat(val_e)
        print(f"{name}: {len(tr_e)} train / {len(val_e)} val episodes, {len(Ptr)} / {len(Pva)} graph samples", flush=True)

        model = GNN(a.K, a.latched); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        steps = a.epochs * (len(Ptr) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
        def val_loss():
            model.eval()
            with torch.no_grad():
                out = torch.cat([model(Pva[i:i+2048], Gva[i:i+2048], Kva[i:i+2048]) for i in range(0, len(Pva), 2048)])
                mse = Fn.mse_loss(out, Yva).item(); rmse_ms = ((out - Yva).norm(dim=-1) ** 2).mean().sqrt().item() * S.V_MAX
            model.train(); return mse, rmse_ms
        log = []; t0 = time.time()
        for ep in range(a.epochs):
            idx = torch.randperm(len(Ptr)); tot = 0; nb = 0
            for i in range(0, len(idx) - a.bs + 1, a.bs):
                b = idx[i:i+a.bs]
                loss = Fn.smooth_l1_loss(model(Ptr[b], Gtr[b], Ktr[b]), Ytr[b], beta=0.1)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item(); nb += 1
            vm, vr = val_loss(); log.append(dict(model=name, epoch=ep + 1, train_loss=tot / nb, val_mse=vm, val_rmse_ms=vr))
            if ep % 5 == 0 or ep == a.epochs - 1: print(f"  epoch {ep+1:3d}  train {tot/nb:.5f}  val mse {vm:.5f}  val rmse {vr*1000:.2f} mm/s  ({time.time()-t0:.0f} s)", flush=True)
        (ROOT / "runs").mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True)
        torch.save(model.state_dict(), ROOT / "runs" / f"{name}.pt")
        pd.DataFrame(log).to_csv(ROOT / "results" / "d4_train_log.csv", mode="a", index=False, header=not (ROOT / "results" / "d4_train_log.csv").exists())

    if a.eval > 0:
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        jobs = [(sd, a.K, a.latched, f, 10000 + s, pct) for pct in (100, 50, 25) for f in S.FORMATIONS for s in range(a.eval)]
        print(f"closed-loop evaluation: {len(jobs)} MuJoCo episodes on {a.workers} workers ...", flush=True)
        with Pool(a.workers) as pool: rows = pool.map(closed_loop_episode, jobs, chunksize=2)
        df = pd.DataFrame(rows); df.insert(0, "model", name); df["train_seed"] = a.seed
        df.to_csv(ROOT / "results" / "d4_eval.csv", mode="a", index=False, header=not (ROOT / "results" / "d4_eval.csv").exists())
        g = df.groupby("obs_pct").agg(success_pct=("success", lambda s: 100 * s.mean()), success05_pct=("success_05", lambda s: 100 * s.mean()),
                                      mean_final_err=("final_err", "mean"), collisions=("collision", "sum"), n=("success", "size"))
        print(f"\n{name} closed-loop (test seeds 10000+):"); print(g.round(3).to_string())


if __name__ == "__main__":      # required: on macOS worker processes re-import this file, so nothing may run at import time
    main()
