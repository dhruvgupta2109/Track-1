"""Round 14: screen interaction groups on V5 out-of-fold residuals (complete train rows). For each candidate group,
ridge on the residual with 5-fold x 4 CV (alpha chosen by inner GCV-free grid on the train folds) -> out-of-sample
R^2 and MSE change. Groups that don't reach a clear positive out-of-sample gain are not worth adding."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from features import clean_input, BUILDINGS, TYPES, BUILDING_TYPE
tr = pd.read_csv("train.csv"); y = tr.energy_usage.to_numpy()
oof = np.load("experiments/oof/e_V5_5x4.npz")["oof"].mean(0); r = y - oof
c = clean_input(tr); ok = c[["previous_usage", "occupancy", "temperature", "humidity"]].notna().all(1).to_numpy()
c = c[ok].reset_index(drop=True); r = r[ok]
B = pd.get_dummies(c.building_id).reindex(columns=BUILDINGS, fill_value=0).to_numpy(float)
Ty = pd.get_dummies(c.building_id.map(BUILDING_TYPE)).reindex(columns=TYPES, fill_value=0).to_numpy(float)
H = pd.get_dummies(c.hour.astype(int)).reindex(columns=range(24), fill_value=0).to_numpy(float)
D = pd.get_dummies(c.dow.astype(int)).reindex(columns=range(7), fill_value=0).to_numpy(float)
Mo = pd.get_dummies(c.month.astype(int)).reindex(columns=range(1, 13), fill_value=0).to_numpy(float)
hb = pd.get_dummies(pd.cut(c.hour, [-1, 5, 8, 11, 14, 17, 20, 23])).to_numpy(float)
we = (c.dow >= 5).to_numpy(float)[:, None]
z = lambda v: ((v - v.mean()) / v.std()).to_numpy()[:, None]
occ, prev, tmp, hum = z(c.occupancy), z(c.previous_usage), z(c.temperature), z(c.humidity)
kr = lambda A, Bm: np.einsum("ij,ik->ijk", A, Bm).reshape(len(A), -1)
G = {
    "building x hour": kr(B, H), "building x dow": kr(B, D), "building x month": kr(B, Mo),
    "type x hour x weekend": kr(kr(Ty, H), we), "building x hourband x weekend": kr(kr(B, hb), we),
    "prev x building": prev * B, "prev x hourband": prev * hb, "prev x type x hourband": kr(prev * Ty, hb),
    "occ x hourband": occ * hb, "occ x building x hourband": kr(occ * B, hb), "occ^2 x building": occ ** 2 * B,
    "occ x weekend x building": occ * we * B, "temp x hourband x type": kr(tmp * Ty, hb), "temp x building x month-season": kr(tmp * B, Mo[:, [0, 3, 6, 9]] + Mo[:, [1, 4, 7, 10]] + Mo[:, [2, 5, 8, 11]]),
    "temp x humidity x type": tmp * hum * Ty, "humidity x building": hum * B, "humidity^2, temp^2 x type": np.c_[hum ** 2 * Ty, tmp ** 2 * Ty],
    "occ x temp x type": occ * tmp * Ty, "occ x prev x type": occ * prev * Ty, "prev^2 x type": prev ** 2 * Ty,
}
base = np.mean(r ** 2); print(f"complete rows {len(r)}, V5 OOF residual MSE {base:.4f} (RMSE {np.sqrt(base):.4f})")
rows = []
for name, X in G.items():
    X = X[:, X.std(0) > 0]; d = []
    for rep in range(4):
        e = np.zeros(len(r))
        for a, b in KFold(5, shuffle=True, random_state=rep).split(X):
            m = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(X[a], r[a]); e[b] = r[b] - m.predict(X[b])
        d.append(np.mean(e ** 2) - base)
    rows.append(dict(group=name, n_cols=X.shape[1], dMSE=np.mean(d), min=np.min(d), max=np.max(d)))
out = pd.DataFrame(rows).sort_values("dMSE"); out["dRMSE"] = np.sqrt(base + out.dMSE) - np.sqrt(base)
print(out.round(4).to_string(index=False))
