"""Section 17 patch. Run from the repo root AFTER src/d17_fusion.py is in place and no run is using src/*.py:  python src/patch_d17.py
Edits src/d12_common.py (three exact-match replacements, each asserted to match once) and adds suites d17_relay and d17_gnn to src/d15_suites.py.
Behaviour is unchanged when the keys fusion/conf_ok/conf_bad are unset. Nothing is written unless every assertion passes."""
import os
assert os.path.exists("src/d17_fusion.py"), "src/d17_fusion.py missing"

edits = [
    # 1. confidence vector and persistent fusion state, right after con_idx is built
    ('[:int(cfg.get("contra_n") or n_con)]',
     '[:int(cfg.get("contra_n") or n_con)]\n'
     '    conf_arr = np.full(N, float(cfg.get("conf_ok") or 0.7)); fz = {}      # section 17: observer confidences, fusion state\n'
     '    if cfg.get("contra") and cfg.get("conf_bad"): conf_arr[con_idx] = float(cfg["conf_bad"])'),
    # 2. relay-with-memory register update: new fusion rules when cfg["fusion"] is set, the old call otherwise
    ('update_registers(rtok, rstamp, obs_e, cmd, cst, A, K, pol["mem"])',
     '(__import__("d17_fusion").step(cfg, fz, obs_e, cmd, rtok, rstamp, A, K, conf_arr, N) if cfg.get("fusion") in ("maxconf", "prov") '
     'else update_registers(rtok, rstamp, obs_e, cmd, cst, A, K, pol["mem"]))'),
    # 3. new keys in the result row
    ('("miss", "wrong", "contra", "contra_n", "contra_rand") if k in cfg',
     '("miss", "wrong", "contra", "contra_n", "contra_rand", "fusion", "conf_ok", "conf_bad") if k in cfg'),
]
p = "src/d12_common.py"; s = open(p).read()
for old, new in edits:
    assert s.count(old) == 1, ("d12_common anchor count", old[:60], s.count(old))
    s = s.replace(old, new)

p2 = "src/d15_suites.py"; s2 = open(p2).read(); marker = 'if __name__ == "__main__":'
assert s2.count(marker) == 1 and "d17_relay" not in s2, "suites file not as expected"
suites = '''def _d17cfg(fusion, cond):
    ck = dict(C0=dict(contra=False, conf_ok=0.7, conf_bad=0.7), C1=dict(contra=True, conf_ok=0.7, conf_bad=0.7),
              C2=dict(contra=True, conf_ok=0.5, conf_bad=0.9), C3=dict(contra=True, conf_ok=0.9, conf_bad=0.3))[cond]
    return {**plain(), **dict(miss=0.0, wrong=0.0, contra_n=1, contra_rand=True, fusion=fusion), **ck}

SUITES15["d17_relay"] = dict(policies=["relay K2+mem"], model_seeds=(0,), seed0=140000, n=25, obs=(50,),
                             cfgs=[_d17cfg(f, c) for f in ("first", "maxconf", "prov") for c in ("C0", "C1", "C2", "C3")])
SUITES15["d17_gnn"] = dict(policies=["GNN K2+latched"], model_seeds=(0, 1, 2), seed0=140000, n=25, obs=(50,),
                           cfgs=[_d17cfg("first", c) for c in ("C0", "C1", "C2", "C3")])

'''
s2 = s2.replace(marker, suites + marker)
open(p, "w").write(s); open(p2, "w").write(s2)
print("patched d12_common.py and d15_suites.py")
