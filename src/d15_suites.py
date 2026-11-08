"""Day 15 suites, merged into d12_eval.SUITES. Filter constants are the ones fixed on the dev seeds on Day 14."""
from d4_common import V_MAX
import swarm_sim as S

DSAFE, GAMMA, SENSE = 0.10, 20.0, 0.30
def margin(delay=0, noise=0.0):
    """Worst-case closing distance during a sensing delay (both robots at V_MAX) plus 3-sigma position error on each of two robots."""
    return 2 * V_MAX * S.DT * delay + 6 * noise
def plain(**kw): return dict(safety=True, dsafe=DSAFE, gamma=GAMMA, sense=SENSE, **kw)
def margined(**kw):
    m = margin(kw.get("delay", 0), kw.get("noise", 0.0))
    return dict(safety=True, dsafe=round(DSAFE + m, 4), gamma=GAMMA, sense=round(SENSE + m, 4), **kw)

A_POL = ["GNN K1", "GNN K2", "GNN K3", "GNN K2+latched", "relay K2+mem", "relayD K2+mem", "relay K2", "relayD K2"]
CHANGE_POL = ["oracle", "relay K2", "relay K2+naive", "relay K2+ver", "relay K3+ver", "GNN K1", "GNN K2", "GNN K3",
              "GNN K2+latched/naive", "GNN K2+latched/ver"]
SUITES15 = {
    "d15_margin_dev": dict(policies=["GNN K2", "GNN K2+latched", "relayD K2+mem"], model_seeds=(0,), seed0=5000, n=10, obs=(50, 25),
                           cfgs=[plain(delay=10), margined(delay=10), plain(noise=0.01), margined(noise=0.01)]),
    "d15_filter_dist": dict(policies=["GNN K2", "GNN K3", "GNN K2+latched", "relayD K2+mem"], model_seeds=(0, 1, 2), seed0=40000, n=25, obs=(50, 25),
                            cfgs=[plain(loss=0.3), plain(loss=0.5), plain(delay=5), margined(delay=5), plain(delay=10), margined(delay=10),
                                  plain(noise=0.005), margined(noise=0.005), plain(noise=0.01), margined(noise=0.01)]),
    "d15_change": dict(policies=CHANGE_POL, model_seeds=(0, 1, 2), seed0=70000, n=25, obs=(50, 25),
                       cfgs=[dict(change=True), plain(change=True)]),
}
for _k in ("conn1", "comp2_same", "comp2_split", "comp3_same"):
    SUITES15[f"d15_A_{_k}"] = dict(policies=A_POL, model_seeds=(0, 1, 2), seed0=50000, n=25, obs=(50, 25),
                                   cfgs=[plain(start_half=0.7, scenario=_k)])

def vm(**kw): return plain(vmargin=1.0, **kw)      # velocity-dependent delay margin (revised rule, see protocol 3b)
SUITES15["d15_filter_dist"]["cfgs"] = [plain(loss=0.3), plain(loss=0.5), plain(delay=5), plain(delay=10), plain(noise=0.005), plain(noise=0.01)]   # margins failed on dev (protocol 3b, 3c)
SUITES15["d15_vm_dev"] = dict(policies=["GNN K2", "GNN K2+latched", "relayD K2+mem"], model_seeds=(0,), seed0=5000, n=10, obs=(50, 25),
                              cfgs=[plain(delay=5), vm(delay=5), vm(delay=10)])

SUITES15["d15_frontier"] = dict(policies=["GNN K2", "GNN K2+latched", "relayD K2+mem", "relay K2+mem"], model_seeds=(0, 1, 2), seed0=5000, n=10, obs=(50, 25),
                                cfgs=[plain(start_half=h) for h in (0.5, 0.7, 0.9, 1.1, 1.3)])

if __name__ == "__main__":
    for d in (5, 10): print(f"delay {d}: margin {1000 * margin(delay=d):.1f} mm")
    for s in (0.005, 0.01): print(f"noise {1000 * s:.0f} mm: margin {1000 * margin(noise=s):.1f} mm")
