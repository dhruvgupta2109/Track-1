"""Out-of-fold imputation errors on OBSERVED values for every train and test row (features only), per feature,
using a single c11-style imputer (domain flag, dev + cyclic features, mask-order fixed). -> conditional SD of each
feature given the others, by (domain, building)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from common import load, test_mask_patterns, apply_masks, NUM
from features import make_features, NUMERIC
from imputed import _mask_aug, IMP_PARAMS
from devimpute import make_tables, add_dev, add_cyclic, LEVELS
tr, te = load(); P = test_mask_patterns(te); PN = P[P.any(1)][:, [NUM.index(c) for c in NUMERIC]]
X = pd.concat([tr.drop(columns="energy_usage"), te[tr.columns.drop("energy_usage")]], ignore_index=True)
dom = np.r_[np.zeros(len(tr)), np.ones(len(te))]
err = {c: np.full(len(X), np.nan) for c in NUMERIC}
for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(X)):
    F0 = add_cyclic(make_features(X.iloc[a]).assign(dom=dom[a])); tabs = make_tables(F0, LEVELS)
    w0 = np.where(dom[a] == 1, 2.0, 1.0)
    H = X.iloc[b].reset_index(drop=True)
    for col in NUMERIC:
        F = _mask_aug(F0, col, PN, 2, seed=11); w = np.tile(w0, 3); ok = F[col].notna().to_numpy()
        m = lgb.LGBMRegressor(**dict(IMP_PARAMS, n_jobs=6)).fit(add_dev(F, tabs, LEVELS, col).drop(columns=[col])[ok], F.loc[ok, col], sample_weight=w[ok])
        okh = H[col].notna().to_numpy()
        Hm = apply_masks(H[okh], P, seed=100 + k); Hm[col] = np.nan
        G = add_cyclic(make_features(Hm).assign(dom=dom[b][okh]))
        pred = m.predict(add_dev(G, tabs, LEVELS, col).drop(columns=[col]))
        idx = np.asarray(b)[okh]; err[col][idx] = H.loc[okh, col].to_numpy() - pred
    print("fold", k, flush=True)
out = pd.DataFrame(err); out["domain"] = dom; out["building_id"] = X.building_id.values
out.to_csv("experiments/imputation_oof_errors.csv", index=False)
sd = out.groupby(["domain", "building_id"])[NUMERIC].agg(lambda s: np.sqrt(np.nanmean(s ** 2)))
print(sd.round(2).to_string())
print("mean bias of OOF imputation (observed values) by domain:"); print(out.groupby("domain")[NUMERIC].mean().round(2))
