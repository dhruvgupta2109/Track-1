import sys, copy; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
from common import load, apply_masks, test_mask_patterns, NUM
from linmodels import PatternLinear
from features import DEFAULT_SPEC
from exp01_baselines import Wrap
from exp02_missing import aug
tr, te = load(); P = test_mask_patterns(te)
S = copy.deepcopy(DEFAULT_SPEC); S.update(hour_fourier=3, weekend_hours="shared", missing_spec=dict(hour_fourier=6, weekend_hours="type"))
L = lambda: Wrap(lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.01, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4))
rows = []
for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr)):
    T = tr.iloc[a]; V = apply_masks(tr.iloc[b].drop(columns="energy_usage"), P, seed=7 + k)
    y = tr.energy_usage.values[b]
    lin = PatternLinear(3.0, S).fit(T.drop(columns="energy_usage"), T.energy_usage.values).predict(V)
    Ta = aug(1.0)(T, seed=k); g = L().fit(Ta.drop(columns="energy_usage"), Ta.energy_usage.values).predict(V)
    pat = V[NUM].isna().astype(int).astype(str).agg("".join, axis=1).values
    rows.append(pd.DataFrame({"pat": pat, "el": (y - lin) ** 2, "eg": (y - g) ** 2, "eb": (y - (0.8 * lin + 0.2 * g)) ** 2}))
d = pd.concat(rows); print("pattern = missing flags for", NUM)
r = d.groupby("pat").agg(n=("el", "size"), lin=("el", "mean"), gbm=("eg", "mean"), blend82=("eb", "mean"))
r[["lin", "gbm", "blend82"]] = np.sqrt(r[["lin", "gbm", "blend82"]]); print(r.round(3))
print("overall", np.sqrt(d[["el", "eg", "eb"]].mean()).round(4).to_dict())
