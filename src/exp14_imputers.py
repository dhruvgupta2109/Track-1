import sys, copy; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from features import clean_input, linear_design, make_features
from specs import V3
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
A = pd.concat([tr.drop(columns="energy_usage"), te], ignore_index=True); is_tr = np.r_[np.ones(len(tr)), np.zeros(len(te))].astype(bool)
C = clean_input(A); F = make_features(A)
for target, others in [("occupancy", ["previous_usage", "temperature", "humidity"]), ("previous_usage", ["occupancy", "temperature", "humidity"])]:
    ok = C[target].notna().to_numpy()
    res = {}
    for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(A)):
        a = a[ok[a]]; b = b[ok[b] & is_tr[b]]          # evaluate on train rows only, fit on train+test features
        drop = {target} | {c for c in others if C[c].isna().any() and False}
        # (1) linear design used by the missing-pattern submodel (prev/occ dropped -> missing_spec)
        Xl = lambda idx: linear_design(C.iloc[idx].fillna({c: C[c].mean() for c in others}), drop={target}, spec=V3)
        p1 = Ridge(3.0).fit(Xl(a), C[target].iloc[a]).predict(Xl(b))
        # (2) LightGBM on all other features
        Fx = F.drop(columns=[target])
        g = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10, verbose=-1, n_jobs=4)
        p2 = g.fit(Fx.iloc[a], C[target].iloc[a]).predict(Fx.iloc[b])
        y = C[target].iloc[b].to_numpy()
        for nm, p in [("linear_design", p1), ("lgb", p2)]:
            res.setdefault(nm, []).append(np.mean((y - p) ** 2))
    var = C[target][is_tr & ok].var()
    print(target, {k: f"rmse {np.sqrt(np.mean(v)):.2f} R2 {1-np.mean(v)/var:.3f}" for k, v in res.items()})
