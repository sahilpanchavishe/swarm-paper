"""Day 6a: MAIN closed-loop experiment. Every trained model is tested on the SAME episodes as the Day 3 baselines:
test seeds 10000..10000+n-1, 4 formations, observability 100/50/25 %  (n = 50 -> 600 episodes per model).
Same seed => same start poses and same observers for every policy, so comparisons are paired.
Nothing is trained or tuned here. The Day 5 final-test seeds (10000-10009) are re-used on purpose: they were evaluated once, never used for tuning.
Run:  python d6_main_eval.py                    # all runs/d5_K*_s*.pt not yet evaluated   (DAgger models)
      python d6_main_eval.py --prefix d4        # behaviour-cloning-only models (for the DAgger ablation)
      python d6_main_eval.py --n 5 --out /tmp/x.csv     # quick smoke test
Writes/appends ../results/d6_main_eval.csv (one row per episode). Models already in that file are skipped, so it is safe to re-run."""
import argparse, os, re, time
from pathlib import Path
from multiprocessing import Pool
import pandas as pd, torch
import swarm_sim as S
from d4_common import closed_loop_episode

ROOT = Path(__file__).resolve().parent.parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="d5", choices=["d4", "d5"]); ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--out", default=str(ROOT / "results" / "d6_main_eval.csv"))
    a = ap.parse_args(); out = Path(a.out)
    done = set(pd.read_csv(out).model) if out.exists() else set()
    files = sorted((ROOT / "runs").glob(f"{a.prefix}_K*_s*.pt"))
    if not files: raise SystemExit(f"no runs/{a.prefix}_K*_s*.pt files found")
    for f in files:
        name = f.stem; m = re.match(rf"{a.prefix}_K(\d)(_latched)?_s(\d+)$", name)
        if not m: continue
        if name in done: print(f"{name}: already evaluated, skipping"); continue
        K, latched, seed = int(m.group(1)), bool(m.group(2)), int(m.group(3))
        sd = torch.load(f, map_location="cpu")
        jobs = [(sd, K, latched, form, 10000 + s, pct) for pct in (100, 50, 25) for form in S.FORMATIONS for s in range(a.n)]
        t0 = time.time(); print(f"{name}: {len(jobs)} episodes on {a.workers} workers ...", flush=True)
        with Pool(a.workers) as pool: rows = pool.map(closed_loop_episode, jobs, chunksize=4)
        df = pd.DataFrame(rows); df.insert(0, "model", name); df["train_seed"] = seed; df["method"] = "DAgger" if a.prefix == "d5" else "BC"
        df.to_csv(out, mode="a", index=False, header=not out.exists())
        g = df.groupby("obs_pct").agg(success_pct=("success", lambda s: 100 * s.mean()), mean_final_err=("final_err", "mean"),
                                      contact_eps=("collision", "sum"), overlap5mm_eps=("min_clearance", lambda s: int((s < 0.069).sum())), n=("success", "size"))
        print(f"  done in {time.time()-t0:.0f} s"); print(g.round(3).to_string(), flush=True)

if __name__ == "__main__":     # required on macOS
    main()
