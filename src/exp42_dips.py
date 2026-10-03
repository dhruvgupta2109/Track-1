"""Round 13 forensics: direction of feature anomalies (spikes up vs dips down) in train vs test,
and whether energy follows dips in train. Consistency z from a target-free LightGBM fitted on train+test features (OOF)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from features import make_features, NUMERIC
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
allx = pd.concat([tr.drop(columns="energy_usage"), te], ignore_index=True)
F = make_features(allx); F["dom"] = np.r_[np.zeros(len(tr)), np.ones(len(te))]
Z = pd.DataFrame(index=F.index, columns=NUMERIC, dtype=float)
for c in NUMERIC:
    ok = F[c].notna().to_numpy(); X = F.drop(columns=[c]); r = np.full(len(F), np.nan)
    idx = np.where(ok)[0]
    for a, b in KFold(5, shuffle=True, random_state=0).split(idx):
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, verbose=-1, n_jobs=4)
        m.fit(X.iloc[idx[a]], F[c].iloc[idx[a]])
        r[idx[b]] = F[c].iloc[idx[b]] - m.predict(X.iloc[idx[b]])
    # robust scale from train rows
    s = np.nanstd(np.where(np.abs(r[:len(tr)]) < 3 * np.nanstd(r[:len(tr)]), r[:len(tr)], np.nan))
    Z[c] = r / s
Z.to_csv("experiments/feature_z_pooled.csv", index=False)
ztr, zte = Z.iloc[:len(tr)].reset_index(drop=True), Z.iloc[len(tr):].reset_index(drop=True)
for c in NUMERIC:
    a, b = ztr[c], zte[c]
    print(f"{c:15s} train up>3 {np.mean(a>3)*100:5.2f}% dn<-3 {np.mean(a<-3)*100:5.2f}% | test up>3 {np.mean(b>3)*100:5.2f}% dn<-3 {np.mean(b<-3)*100:5.2f}%  (n tr up/dn {int((a>3).sum())}/{int((a<-3).sum())}, te {int((b>3).sum())}/{int((b<-3).sum())})")
# train residual behaviour by direction (V5 OOF, clean)
oof = np.load("experiments/oof/e_V5_5x4.npz")["oof"].mean(0); y = tr.energy_usage.to_numpy(); e = y - oof
print("\nV5 clean OOF residual by z bin (train):")
for c in ["previous_usage", "occupancy", "temperature", "humidity"]:
    zb = pd.cut(ztr[c], [-99, -6, -3, -2, 2, 3, 6, 99])
    g = pd.DataFrame({"zb": zb, "e": e}).groupby("zb", observed=True).e.agg(["count", "mean", lambda v: np.sqrt(np.mean(v**2))])
    g.columns = ["n", "bias", "rmse"]; print(c); print(g.round(2).to_string())
