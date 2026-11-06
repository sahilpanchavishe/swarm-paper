"""Day 12: capacity-control architectures. d4_common.py is untouched; these are SUBCLASSES / re-parameterisations of its GNN.

Why: K=2 beats K=1 even at 100 % observability, where every robot already knows the gesture, and K=2 has ~47 % more parameters
than K=1 (44,034 vs 64,898). So depth and capacity are tangled. Three controls separate them:
  wide1   K=1, hidden width raised so the parameter count matches K=2     (1 hop, K=2 capacity)
  shared2 K=2 with the SAME weights in both message-passing layers        (2 hops, K=1 capacity)
  mlpwide K=0 (no communication at all), width raised to match K=2         (capacity only, no neighbour information)
All three use exactly the same inputs, training data, DAgger schedule and evaluation as the main models.
"""
import torch.nn as nn
from d4_common import GNN, N

FIN = 2 + N + 4 + 1          # position, role one-hot, gesture token, know flag (same as d4_common.GNN)

def count(K, h):
    """Parameter count of d4_common.GNN(K, hid=h), written out so a width can be searched without building networks."""
    enc = FIN * h + h + h * h + h
    msg = K * ((h + 2) * h + h + h * h + h)
    upd = K * (2 * h * h + h + h * h + h)
    head = (h + 2 + N) * 2 * h + 2 * h + 2 * h * h + h + h * 2 + 2
    return enc + msg + upd + head

def n_params(model):
    return sum(p.numel() for p in model.parameters())           # parameters() already de-duplicates shared weights

def wide_hid(K, target=None):
    """Hidden width that makes GNN(K) as close as possible to the parameter count of the base K=2 model."""
    target = count(2, 64) if target is None else target
    return min(range(16, 400), key=lambda h: abs(count(K, h) - target))

class GNNShared(GNN):
    """K message-passing layers that all use the SAME message and update networks (parameters of one layer, applied K times)."""
    def __init__(self, K, latched=False, hid=64):
        super().__init__(K, latched, hid)
        m, u = self.msg[0], self.upd[0]
        self.msg = nn.ModuleList([m] * K); self.upd = nn.ModuleList([u] * K)

ARCHS = {"wide1": dict(arch="wide", K=1), "shared2": dict(arch="shared", K=2), "mlpwide": dict(arch="wide", K=0)}

def make_model(arch, K, latched=False):
    if arch == "base": return GNN(K, latched)
    if arch == "wide": return GNN(K, latched, hid=wide_hid(K))
    if arch == "shared": return GNNShared(K, latched)
    raise ValueError(arch)
