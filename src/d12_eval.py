"""Day 12: evaluation driver for every new experiment. One command per suite; each suite writes ../results/d12_<suite>.csv (one row per episode).
Resume-safe: a (policy, training seed, condition) unit already in the CSV is skipped, so Ctrl-C and re-run is fine. Nothing is trained here.

  python d12_eval.py repro          # ~5 min.  RUN FIRST. Re-runs a few Day 3 / Day 6 / Day 9 episodes with the new engine and compares.
  python d12_eval.py confirm        # fresh confirmatory block: seeds 20000-20049, all main models and baselines, no perturbation (~3 h on 8 cores)
  python d12_eval.py capacity       # capacity controls (needs runs/d12_*.pt), same fresh block
  python d12_eval.py safety_dev     # pick the safety-filter parameters on DEV seeds 5000-5009 (never the confirmatory block)
  python d12_eval.py safety_final --dsafe 0.09 --gamma 10   # chosen filter, fresh block
  python d12_eval.py robust         # packet loss, delay, noise; exploratory seeds 30000+
  python d12_eval.py widebox        # 1.8 m start box; exploratory seeds 30000+
  python d12_eval.py change         # mid-episode command change; exploratory seeds 30000+
Add --plan to any command to see the units and episode counts without running. Seed blocks: demos 0-1199, DAgger 1300-4299, dev 5000+,
Day 6 test 10000-10049 (already reused in exploratory work), CONFIRMATORY 20000-20049 (use once), exploratory 30000+."""
import argparse, os, re, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, torch
import swarm_sim as S
from d12_common import run_episode, BASE_CFG

ROOT = Path(__file__).resolve().parent.parent
MAIN = ["GNN K0", "GNN K1", "GNN K2", "GNN K3", "GNN K2+latched"]
RELAYS = ["oracle", "relay K0"] + [f"relay K{K}{s}" for K in (1, 2, 3, 7) for s in ("", "+mem")]
RELAYD = [f"relayD K{K}" for K in (0, 1, 2, 3, 7)]
ROBUST_CFGS = [{}] + [dict(loss=q) for q in (0.1, 0.3, 0.5)] + [dict(delay=d) for d in (5, 10)] + [dict(noise=s) for s in (0.005, 0.01)]
SUITES = {
    "repro": dict(policies=["GNN K1", "GNN K2", "GNN K3", "GNN K2+latched", "oracle", "relay K2", "relay K2+mem", "relay K7+mem", "relayD K2"],
                  model_seeds=(0,), seed0=10000, n=5, cfgs=[{}], obs=None),
    "confirm": dict(policies=MAIN + RELAYS + RELAYD, model_seeds=None, seed0=20000, n=50, cfgs=[{}], obs=None),
    "capacity": dict(policies=["GNN K1-wide", "GNN K2-shared", "GNN K0-wide"], model_seeds=None, seed0=20000, n=50, cfgs=[{}], obs=None),
    "safety_dev": dict(policies=["GNN K2", "GNN K3"], model_seeds=(0,), seed0=5000, n=10, obs=(100, 25),
                       cfgs=[{}] + [dict(safety=True, dsafe=d, gamma=g) for d, g in ((0.085, 5), (0.09, 10), (0.09, 20), (0.10, 10), (0.10, 20))]),
    "safety_final": dict(policies=["GNN K1", "GNN K2", "GNN K3", "GNN K2+latched", "relay K2+mem"], model_seeds=None, seed0=20000, n=50,
                         cfgs=[dict(safety=True)], obs=None),
    "robust": dict(policies=["GNN K2", "GNN K3", "GNN K2+latched", "relay K2+mem", "relay K3+mem", "relay K2", "relayD K2"],
                   model_seeds=(0, 1, 2), seed0=30000, n=20, cfgs=ROBUST_CFGS, obs=(50, 25)),
    "widebox": dict(policies=["GNN K2", "GNN K3", "GNN K2+latched", "relay K2+mem", "relay K7+mem"], model_seeds=(0, 1, 2), seed0=30000, n=25,
                    cfgs=[dict(start_half=0.9)], obs=(50, 25)),
    "widebox_safe": dict(policies=["GNN K2", "GNN K3", "GNN K2+latched", "relayD K2+mem"], model_seeds=(0, 1, 2), seed0=30000, n=25,
                         cfgs=[dict(start_half=0.9, safety=True)], obs=(50, 25)),
    "change": dict(policies=["oracle", "relay K2", "relay K2+naive", "relay K2+ver", "relay K3+ver", "GNN K1", "GNN K2", "GNN K3",
                             "GNN K2+latched/naive", "GNN K2+latched/ver"],
                   model_seeds=(0, 1, 2), seed0=30000, n=25, cfgs=[dict(change=True)], obs=(50, 25)),
}

from d15_suites import SUITES15
SUITES.update(SUITES15)

def parse_token(tok):
    """Policy name -> spec dict. 'relay K2+mem' and 'relay K2+ver' are the same latest-wins rule; '+naive' keeps the first token it received.
    'relayD' = uninformed robots head to the mean slot instead of holding still. 'GNN K2+latched/naive' = latched GNN fed per-robot stale tokens."""
    if tok == "oracle": return dict(kind="oracle", name=tok, K=0, mem="none", default="hold", obs=(100,))
    m = re.fullmatch(r"relay(D?) K(\d)(?:\+(mem|ver|naive))?", tok)
    if m: return dict(kind="relay", name=tok, K=int(m.group(2)), mem={None: "none", "mem": "fresh", "ver": "fresh", "naive": "naive"}[m.group(3)],
                      default="mean" if m.group(1) else "hold", obs=(50, 25))
    m = re.fullmatch(r"GNN K(\d)(-wide|-shared)?(\+latched)?(?:/(ver|naive))?", tok)
    if m:
        K, suf, lat = int(m.group(1)), m.group(2), bool(m.group(3))
        if suf == "-wide": arch, stem = "wide", {1: "d12_wide1", 0: "d12_mlpwide"}[K]
        elif suf == "-shared": arch, stem = "shared", {2: "d12_shared2"}[K]
        else: arch, stem = "base", (f"d5_K{K}_latched" if lat else f"d5_K{K}")
        return dict(kind="gnn", name=tok, K=K, arch=arch, latched=lat, reg=("naive" if m.group(4) == "naive" else "fresh") if lat else "none",
                    stem=stem, obs=(100, 50, 25))
    raise SystemExit(f"unknown policy name: {tok!r}")

def cond_tag(cfg):
    d = {k: v for k, v in cfg.items() if BASE_CFG.get(k) != v}
    return "|".join(f"{k}={v}" for k, v in sorted(d.items())) or "clean"

def build_units(S_, args):
    units = []
    for tok in S_["policies"]:
        spec = parse_token(tok); files = [(None, None)]
        if spec["kind"] == "gnn":
            files = []
            for f in sorted((ROOT / "runs").glob(f"{spec['stem']}_s*.pt")):
                m = re.fullmatch(rf"{re.escape(spec['stem'])}_s(\d+)", f.stem)
                if m and (S_["model_seeds"] is None or int(m.group(1)) in S_["model_seeds"]): files.append((int(m.group(1)), f))
            if not files: print(f"  [skip] no trained models found for {tok} (looked for runs/{spec['stem']}_s*.pt)")
        obs = (100,) if spec["kind"] == "oracle" else tuple(o for o in (S_["obs"] or spec["obs"]) if o in spec["obs"])
        for ts, path in files:
            for cfg in S_["cfgs"]:
                cfg = dict(cfg)
                if cfg.get("safety"):
                    if args.dsafe is not None and "dsafe" not in cfg: cfg["dsafe"] = args.dsafe
                    if args.gamma is not None and "gamma" not in cfg: cfg["gamma"] = args.gamma
                key = f"{tok}|s{ts}|{cond_tag(cfg)}|n{S_['n']}|b{S_['seed0']}"
                units.append(dict(key=key, spec=spec, path=path, train_seed=ts, cfg=cfg, obs=obs))
    return units

def repro_report(out):
    new = pd.read_csv(out); r = ROOT / "results"; key = ["formation", "seed", "obs_pct"]
    d6 = pd.read_csv(r / "d6_main_eval.csv"); d6 = d6[d6.method == "DAgger"]; d3 = pd.read_csv(r / "d3_baselines.csv"); d9 = pd.read_csv(r / "d9_relay_default.csv")
    print("\n=== REPRODUCTION CHECK: new engine (defaults) vs the committed Day 3 / 6 / 9 results, same episodes ===")
    good = True
    for pol, x in new.groupby("policy"):
        sp = parse_token(pol)
        if sp["kind"] == "gnn":
            ref = d6[(d6.K == sp["K"]) & (d6.latched == sp["latched"])]; keys = key + ["train_seed"]
        elif sp["kind"] == "oracle": ref = d3[d3.policy == "oracle"].assign(train_seed=np.nan); keys = key
        elif pol.startswith("relayD"): ref = d9[d9.policy == pol].assign(train_seed=np.nan); keys = key
        else: ref = d3[d3.policy == pol[len("relay "):]].assign(train_seed=np.nan); keys = key
        j = x.merge(ref[keys + ["final_err", "success", "collision"]], on=keys, suffixes=("", "_ref"))
        if j.empty: print(f"  {pol:18s} NO MATCHING REFERENCE EPISODES"); good = False; continue
        agree, cagree, de = (j.success == j.success_ref).mean(), (j.collision == j.collision_ref).mean(), (j.final_err - j.final_err_ref).abs()
        flag = "ok" if agree >= 0.97 and de.median() < 1e-3 else "CHECK"; good &= flag == "ok"
        print(f"  {pol:18s} n={len(j):4d}  success agree {100*agree:5.1f}%  contact agree {100*cagree:5.1f}%  |d final_err| median {1000*de.median():.3f} mm, max {1000*de.max():.2f} mm  [{flag}]")
    print("\nREPRO PASS: the new engine reproduces the old results." if good else
          "\nREPRO CHECK: differences found. Small differences are expected only if this is a different computer from the one that made the Day 6 files "
          "(floating point); anything larger means a bug. Send me this output before running anything else.")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("suite", choices=list(SUITES)); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--n", type=int); ap.add_argument("--seed0", type=int); ap.add_argument("--model-seeds", help="comma list, e.g. 0,1,2")
    ap.add_argument("--obs", help="comma list, e.g. 25,50"); ap.add_argument("--policies", help="comma list overriding the suite's policies")
    ap.add_argument("--dsafe", type=float); ap.add_argument("--gamma", type=float); ap.add_argument("--out"); ap.add_argument("--plan", action="store_true")
    a = ap.parse_args(); S_ = dict(SUITES[a.suite])
    if a.n: S_["n"] = a.n
    if a.seed0 is not None: S_["seed0"] = a.seed0
    if a.model_seeds: S_["model_seeds"] = tuple(int(s) for s in a.model_seeds.split(","))
    if a.obs: S_["obs"] = tuple(int(s) for s in a.obs.split(","))
    if a.policies: S_["policies"] = [p.strip() for p in a.policies.split(",")]
    if a.suite == "safety_final" and (a.dsafe is None or a.gamma is None): raise SystemExit("safety_final needs --dsafe and --gamma (the values chosen on the dev seeds)")
    out = ROOT / "results" / (a.out or f"d12_{a.suite}.csv"); (ROOT / "results").mkdir(exist_ok=True)
    done = set(pd.read_csv(out, usecols=["unit"]).unit) if out.exists() else set()
    units = build_units(S_, a); todo = [u for u in units if u["key"] not in done]
    per = lambda u: len(u["obs"]) * 4 * S_["n"]; total = sum(per(u) for u in todo)
    print(f"suite {a.suite}: {len(units)} units, {len(units)-len(todo)} already done, {len(todo)} to run = {total} episodes, seeds {S_['seed0']}..{S_['seed0']+S_['n']-1}, {a.workers} workers")
    if a.plan or not todo: return
    t0 = time.time(); ndone = 0
    with Pool(a.workers) as pool:
        for i, u in enumerate(todo, 1):
            pol = dict(u["spec"]); pol["train_seed"] = u["train_seed"] if u["train_seed"] is not None else np.nan
            if pol["kind"] == "gnn": pol["sd"] = {k: v.cpu() for k, v in torch.load(u["path"], map_location="cpu").items()}
            jobs = [(pol, u["cfg"], f, S_["seed0"] + s, pct) for pct in u["obs"] for f in S.FORMATIONS for s in range(S_["n"])]
            t1 = time.time(); rows = pool.map(run_episode, jobs, chunksize=4)
            df = pd.DataFrame(rows); df.insert(0, "unit", u["key"]); df.insert(1, "suite", a.suite)
            df.to_csv(out, mode="a", index=False, header=not out.exists()); ndone += len(jobs)
            el = time.time() - t0; eta = el / ndone * (total - ndone)
            print(f"[{i}/{len(todo)}] {u['key']}: success {100*df.success.mean():5.1f}%  contact {100*df.collision.mean():5.1f}%  ({time.time()-t1:.0f} s; elapsed {el/60:.0f} min, about {eta/60:.0f} min left)", flush=True)
    print(f"\nsaved {out}")
    if a.suite == "repro": repro_report(out)

if __name__ == "__main__":        # required on macOS and Windows (workers re-import this file)
    main()
