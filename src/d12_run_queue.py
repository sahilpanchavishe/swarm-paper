"""Day 12: run a whole training queue unattended, one model after another. Resume-safe (finished models are skipped), works on Mac, Windows and Linux.
  python d12_run_queue.py seeds34     # main models: K = 2, 1, 3 for training seeds 3 and 4 (BC then DAgger, same recipe as d6_more_seeds.sh)
  python d12_run_queue.py controls    # capacity controls: wide1, shared2, mlpwide for training seeds 0, 1, 2
  python d12_run_queue.py seeds34 --plan      # just list what would run
Stop it with Ctrl-C at any time; run the same command again to continue. Output is also appended to ../logs/d12_train.log.
Keep the computer awake while it runs (Mac: run it as `caffeinate -i python d12_run_queue.py seeds34`; Windows: set Sleep to Never)."""
import subprocess, sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parent; ROOT = SRC.parent; LOG = ROOT / "logs" / "d12_train.log"

class _Marker:
    """d5_dagger.py saves its .pt after EVERY round, so the file alone does not mean 'finished'. It logs the model in
    results/d5_dagger_log.csv only after the last round: that log line is the finished-marker."""
    def __init__(self, name): self.name = name
    def exists(self):
        p = ROOT / "results" / "d5_dagger_log.csv"
        if not p.exists(): return False
        import pandas as pd
        return self.name in set(pd.read_csv(p, usecols=["model"]).model)

def jobs(queue):
    out = []
    if queue == "seeds34":
        for seed in (3, 4):
            for K in (2, 1, 3):
                done = _Marker(f"d5_K{K}_s{seed}"); bc = ROOT / "runs" / f"d4_K{K}_s{seed}.pt"; steps = []
                if not bc.exists(): steps.append([sys.executable, "d4_train.py", "--K", str(K), "--seed", str(seed), "--epochs", "30", "--eval", "0"])
                steps.append([sys.executable, "d5_dagger.py", "--K", str(K), "--seed", str(seed), "--rounds", "3", "--eps", "240", "--epochs", "10", "--dev", "5", "--eval", "0"])
                out.append((f"K={K} seed={seed}", done, steps))
    elif queue == "controls":
        for seed in (0, 1, 2):
            for arch in ("wide1", "shared2", "mlpwide"):
                out.append((f"{arch} seed={seed}", ROOT / "runs" / f"d12_{arch}_s{seed}.pt", [[sys.executable, "d12_train_variants.py", "--arch", arch, "--seed", str(seed)]]))
    else: raise SystemExit("queue must be seeds34 or controls")
    return out

def run(cmd, log):
    p = subprocess.Popen(cmd, cwd=SRC, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: print(line, end="", flush=True); log.write(line); log.flush()
    return p.wait()

def main():
    queue = sys.argv[1] if len(sys.argv) > 1 else ""; todo = [j for j in jobs(queue) if not j[1].exists()]
    print(f"{queue}: {len(todo)} model(s) to train: {', '.join(j[0] for j in todo) or 'nothing, all done'}")
    if "--plan" in sys.argv or not todo: return
    LOG.parent.mkdir(exist_ok=True); t0 = time.time()
    with open(LOG, "a") as log:
        for i, (label, done, steps) in enumerate(todo, 1):
            print(f"\n===== [{i}/{len(todo)}] {label}  (elapsed {(time.time()-t0)/60:.0f} min) =====", flush=True); log.write(f"\n===== {label} =====\n")
            for cmd in steps:
                if run(cmd, log) != 0: raise SystemExit(f"FAILED: {' '.join(cmd)}\nFix the error above (send it to me) and run the same command again; finished models are kept.")
    print(f"\nALL DONE in {(time.time()-t0)/60:.0f} min")

if __name__ == "__main__":
    main()
