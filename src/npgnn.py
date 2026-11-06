"""Torch-free loader + numpy forward pass for the repo's GNN (identical maths to d4_common.GNN).
Only used here because the sandbox cannot run PyTorch; verified against saved results below."""
import zipfile, pickle, numpy as np, io
class _T:  # placeholder tensor
    def __init__(s, arr): s.arr = arr
def load_pt(path):
    z = zipfile.ZipFile(path); names = z.namelist()
    pkl = [n for n in names if n.endswith('data.pkl')][0]; root = pkl[:-len('data.pkl')]
    dt = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64}
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == '_rebuild_tensor_v2':
                def f(storage, off, size, stride, *a):
                    arr = np.lib.stride_tricks.as_strided(storage[off:], shape=size, strides=[s*storage.itemsize for s in stride]) if len(size) else storage[off:off+1]
                    return _T(np.array(arr))
                return f
            if name == 'OrderedDict':
                from collections import OrderedDict; return OrderedDict
            if mod == 'torch' and name.endswith('Storage'): return name
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            _, st, key, loc, n = pid; t = dt[st if isinstance(st, str) else st.__name__]
            return np.frombuffer(z.read(f'{root}data/{key}'), dtype=t)
    sd = U(io.BytesIO(z.read(pkl))).load()
    return {k: v.arr for k, v in sd.items()}
def lin(sd, p, x): return x @ sd[p+'.weight'].T + sd[p+'.bias']
def relu(x): return np.maximum(x, 0)
def mlp(sd, p, x, n):  # Sequential of Linear/ReLU: layers at even idx
    for i in range(n):
        x = lin(sd, f'{p}.{2*i}', x)
        if i < n-1: x = relu(x)
    return x
def forward(sd, K, P, gest, know, R_COMM=0.5, N=8):
    D = np.linalg.norm(P[:, None]-P[None], axis=-1); A = ((D <= R_COMM) & (D > 0)).astype(np.float32)
    ids = np.eye(N, dtype=np.float32); tok = gest[None, :]*know[:, None]
    x = np.concatenate([P/1.1, ids, tok, know[:, None]], -1)
    h = relu(mlp(sd, 'enc', x, 2))
    rel = (P[None]-P[:, None])/R_COMM
    for k in range(K):
        hj = np.broadcast_to(h[None], (N, N, h.shape[-1]))
        m = mlp(sd, f'msg.{k}', np.concatenate([hj, rel], -1), 2)*A[..., None]
        agg = m.sum(1)/4.0
        h = h + mlp(sd, f'upd.{k}', np.concatenate([h, agg], -1), 2)
    return mlp(sd, 'head', np.concatenate([h, P/1.1, ids], -1), 3)
