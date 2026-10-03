"""Round 14: is the energy response to extreme (spiked / dipped) inputs exactly linear? Train complete rows only.
Deviation d = observed value - target-free consistency prediction (experiments/feature_z_pooled.csv, in SD units).
Fit OOF residual of V5 ~ d + d^2 (+ |d|) and test curvature; also residual variance vs |d|."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd

def ols_hc3(X, y):
    """OLS with HC3 robust standard errors -> (params, t-values)."""
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtXi, X); w = (e / (1 - h)) ** 2
    V = XtXi @ (X.T * w) @ X @ XtXi; return b, b / np.sqrt(np.diag(V))
from common import load
from features import NUMERIC
tr, te = load(); y = tr.energy_usage.to_numpy()
oof = np.load("experiments/oof/e_V5_5x4.npz")["oof"].mean(0); r = y - oof
Z = pd.read_csv("experiments/feature_z_pooled.csv").iloc[:len(tr)].reset_index(drop=True)
ok = tr[NUMERIC].notna().all(1).to_numpy()
for c in ["previous_usage", "occupancy", "temperature", "humidity"]:
    z = Z[c].to_numpy(); m = ok & np.isfinite(z)
    X = np.c_[np.ones(m.sum()), z[m], z[m] ** 2]; pr, tv = ols_hc3(X, r[m])
    zb = pd.cut(z[m], [-99, -4, -3, -2, -1, 1, 2, 3, 4, 99])
    g = pd.DataFrame({"zb": zb, "r": r[m]}).groupby("zb", observed=True).r.agg(["count", "mean", "std"])
    print(f"\n{c}: resid ~ z + z^2 (HC3): lin {pr[1]:+.4f} (t {tv[1]:+.1f}), quad {pr[2]:+.4f} (t {tv[2]:+.1f})")
    print(g.round(2).T.to_string())
# variance vs |z| (heteroscedasticity) using max |z| over features
zm = Z[["previous_usage", "occupancy"]].abs().max(1).to_numpy()
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]:
    s = ok & (zm >= lo) & (zm < hi); print(f"max|z| in [{lo},{hi}): n {s.sum():5d}  resid sd {r[s].std():.3f}  mean {r[s].mean():+.3f}")
