"""Round 14: level drift train -> test (label-free). Train-only LightGBM models of previous_usage and occupancy from
the other features (no domain flag); residuals on test vs OOF residuals on train, by building (medians: robust to
spikes). If test prev is higher than train-predicted given occupancy etc., the test-period ENERGY level has drifted.
Also: in train, does prev track energy's level across months (prev month effect vs energy month effect)?"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from features import make_features
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
Ft = make_features(tr); Fe = make_features(te)
P = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=6)
def resid(target, drop):
    cols = [c for c in Ft.columns if c not in [target] + drop]
    ok = Ft[target].notna().to_numpy(); oof = np.full(len(Ft), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=0).split(Ft):
        a = a[ok[a]]; m = lgb.LGBMRegressor(**P).fit(Ft.iloc[a][cols], Ft[target].iloc[a]); oof[b] = m.predict(Ft.iloc[b][cols])
    m = lgb.LGBMRegressor(**P).fit(Ft[ok][cols], Ft[target][ok]); pe = m.predict(Fe[cols])
    return Ft[target].to_numpy() - oof, Fe[target].to_numpy() - pe
out = {}
for target, drop, lab in [("previous_usage", [], "prev | all other features"), ("previous_usage", ["occupancy"], "prev | no occupancy"),
                          ("occupancy", ["previous_usage"], "occupancy | no prev")]:
    rt, re_ = resid(target, drop)
    g = pd.DataFrame({"b": np.r_[tr.building_id, te.building_id], "d": np.r_[np.zeros(len(tr)), np.ones(len(te))], "r": np.r_[rt, re_]}).dropna()
    med = g.groupby(["b", "d"]).r.median().unstack(); med["shift"] = med[1.0] - med[0.0]
    print(f"\n{lab}: median residual train {np.nanmedian(rt):+.2f}, test {np.nanmedian(re_):+.2f};  per-building test-minus-train median shift:")
    print(med["shift"].round(2).to_dict()); out[lab] = med["shift"]
# energy vs prev level co-movement across months/buildings in train
y = tr.energy_usage
cell = tr.groupby(["building_id", "month"]).agg(e=("energy_usage", "mean"), p=("previous_usage", "mean"), o=("occupancy", "mean"))
cell = cell - cell.groupby(level=0).transform("mean")   # within-building deviations across months
b = np.polyfit(cell.p, cell.e, 1)
print(f"\ntrain: within-building month-level co-movement: energy deviation = {b[0]:.2f} x prev deviation (corr {np.corrcoef(cell.p, cell.e)[0,1]:.2f})")
X = np.c_[cell.p, cell.o, np.ones(len(cell))]; coef = np.linalg.lstsq(X, cell.e, rcond=None)[0]
print(f"       with occupancy: energy dev = {coef[0]:.2f} x prev dev + {coef[1]:.3f} x occ dev")
