"""Section 17: command-fusion rules for the relay with memory (used through cfg["fusion"] in d12_common.run_episode).
  maxconf: highest-confidence-wins. Register stamp = observer confidence (x100, integer); a robot adopts the neighbour token with a strictly higher stamp.
  prov:    count-by-source. Every robot keeps the set of distinct observer sources that vouched for each formation (sets are merged along the
           communication graph and kept in memory) and adopts the formation with the most distinct sources. Ties keep the token already held.
Self-test: cd src && python d17_fusion.py"""
import numpy as np


def step(cfg, fz, obs_e, cmd, rtok, rstamp, A, K, conf, N):
    """One control step. obs_e (N,) bool observers reading the command now; cmd (obs_e.sum(),) formation ids they read; conf (N,) confidences.
    fz is a dict that persists over the episode. Returns new (rtok, rstamp); rstamp >= 0 means the robot holds a token."""
    if cfg["fusion"] == "maxconf":
        from d12_common import update_registers
        cst = np.rint(100 * conf[obs_e]).astype(int)
        return update_registers(rtok, rstamp, obs_e, cmd, cst, A, max(K, 1), "fresh")
    if cfg["fusion"] == "prov":
        if "v" not in fz: fz["v"] = np.zeros((N, 4, N), bool)         # v[i, f, s]: robot i knows that source s vouched for formation f
        v = fz["v"]; ids = np.nonzero(obs_e)[0]; v[ids, cmd, ids] = True
        Ai = A.astype(np.int32)
        for _ in range(max(K, 1)):
            v = v | ((Ai @ v.reshape(N, -1).astype(np.int32)) > 0).reshape(N, 4, N)
        fz["v"] = v
        cnt = v.sum(2); best = cnt.max(1); know = best > 0
        held = np.where(rtok >= 0, rtok, 0)
        keep = (rtok >= 0) & (cnt[np.arange(N), held] == best)
        tok = np.where(know, np.where(keep, rtok, cnt.argmax(1)), -1)
        return tok.astype(rtok.dtype), np.where(know, 0, -1).astype(rstamp.dtype)
    raise ValueError(cfg["fusion"])


if __name__ == "__main__":
    N = 8; A = ~np.eye(N, dtype=bool)
    obs_e = np.zeros(N, bool); obs_e[:4] = True; cmd = np.array([1, 1, 1, 2]); conf = np.array([.5, .5, .5, .9, 0, 0, 0, 0])
    rtok, rstamp = np.full(N, -1), np.full(N, -1)
    t, s = step(dict(fusion="prov"), {}, obs_e, cmd, rtok, rstamp, A, 2, conf, N)
    ok1 = bool((t == 1).all() and (s >= 0).all())
    t2, s2 = step(dict(fusion="maxconf"), {}, obs_e, cmd, rtok, rstamp, A, 2, conf, N)
    ok2 = bool((t2 == 2).all() and (s2 >= 0).all())
    # disconnected graph: only robots that can reach an observer know anything
    A3 = np.zeros((N, N), bool); A3[0, 4] = A3[4, 0] = True
    t3, s3 = step(dict(fusion="prov"), {}, obs_e, cmd, rtok, rstamp, A3, 1, conf, N)
    ok3 = bool(t3[4] == 1 and t3[5] == -1)
    print("prov majority:", ok1, "| maxconf highest confidence:", ok2, "| disconnected robot informed only through its neighbour:", ok3)
    print("PASS" if ok1 and ok2 and ok3 else "FAIL")
