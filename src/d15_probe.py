"""Day 15 (EXPLORATORY): can swarm positions at the cut alone identify the formation? Learned probe, grouped CV by start seed. Run: python src/d15_probe.py"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(ROOT / "results" / "d15_cutoff_primary.csv"); d = d[(d.variant == "flagoff") & (~d.latched.astype(bool)) & (d.K == 2)]
iu = np.triu_indices(8, 1)
def feat(r):
    P = np.array([[r[f"x{i}"], r[f"y{i}"]] for i in range(8)]); P = P - P.mean(0)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)[iu]
    return np.concatenate([np.sort(np.linalg.norm(P, axis=1)), np.sort(D)])
rows = []
for tc, x in d.groupby("t_cut"):
    X = np.array([feat(r) for _, r in x.iterrows()]); y = x.formation.values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    pred = cross_val_predict(clf, X, y, groups=x.seed.values, cv=GroupKFold(5)); ok = pred == y
    rows.append(dict(t_cut=tc, n=len(x), probe_acc_pct=100 * ok.mean(), template_acc_pct=100 * x.near_cut_ok.mean(), success_pct=100 * x.success.mean(),
                     success_if_probe_right=100 * x.success[ok].mean() if ok.any() else np.nan, success_if_probe_wrong=100 * x.success[~ok].mean() if (~ok).any() else np.nan))
R = pd.DataFrame(rows); R.to_csv(ROOT / "results" / "d15_probe.csv", index=False); print(R.round(1).to_string(index=False))
print("\nt_cut=0 is the control: every formation starts from the same positions, so accuracy there should be near 25%.")
