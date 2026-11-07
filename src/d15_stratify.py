"""Day 15 (EXPLORATORY, no new simulations): success of the existing runs stratified by the initial communication graph.
Strata: conn = 1 component | disc_1 = >1 components, all observers in one component | disc_n = >1 components, observers in >1.
Positions and observers are recomputed from the seeds exactly as run_episode draws them; the recomputed component count is
checked against the saved init_comp.  Run: python src/d15_stratify.py   Writes results/d15_stratified_existing.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE))
import swarm_sim as S
from d4_common import N, R_COMM
from d15_scenarios import labels

_c = {}
def placement(seed, half, obs_pct):
    key = (int(seed), half, int(obs_pct))
    if key not in _c:
        p, _ = S.random_start(np.random.default_rng(int(seed)), half=half)
        lab, c = labels(S.comm_adjacency(p, R_COMM))
        o = np.random.default_rng([int(seed), 99]).choice(N, {100: 8, 50: 4, 25: 2}[int(obs_pct)], replace=False)
        _c[key] = (c, len(set(lab[o])))
    return _c[key]

def load(name, half, safety):
    d = pd.read_csv(ROOT / "results" / name)
    for c, v in dict(loss=0, delay=0, noise=0, change=False, safety=safety, start_half=half).items():
        if c in d: d = d[d[c] == v]
    return d.copy()

out = []
for tag, name, half, safety in (("standard box", "d12_confirm.csv", 0.5, False), ("wide box", "d12_widebox.csv", 0.9, False),
                                ("wide box + filter", "d12_widebox_safe.csv", 0.9, True)):
    d = load(name, half, safety); d = d[d.obs_pct < 100]
    pl = [placement(s, half, o) for s, o in zip(d.seed, d.obs_pct)]
    d["comp"] = [a for a, _ in pl]; d["obs_comp"] = [b for _, b in pl]
    print(f"{tag}: recomputed components match saved init_comp in {100 * (d.comp == d.init_comp).mean():.1f}% of {len(d)} episodes")
    d["stratum"] = np.where(d.comp == 1, "conn", np.where(d.obs_comp == 1, "disc_1", "disc_n"))
    d["cf"] = d.success & ~d.collision
    g = d.groupby(["policy", "obs_pct", "stratum"]).agg(n=("success", "size"), success_pct=("success", lambda x: 100 * x.mean()),
                                                         cf_success_pct=("cf", lambda x: 100 * x.mean()), informed_final=("informed_final", "mean")).reset_index()
    g.insert(0, "set", tag); out.append(g)
R = pd.concat(out, ignore_index=True); R.to_csv(ROOT / "results" / "d15_stratified_existing.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
for tag in R.set.unique():
    P = R[R.set == tag].assign(cell=lambda x: x.success_pct.round(1).astype(str) + " (n=" + x.n.astype(str) + ")")
    print(f"\n=== {tag}: success % by initial graph ===")
    print(P.pivot_table(index=["policy", "obs_pct"], columns="stratum", values="cell", aggfunc="first").to_string())
