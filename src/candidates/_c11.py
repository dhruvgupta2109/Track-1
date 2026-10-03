"""Shared builder for c11 variants: c09 imputers with the mask-order bug FIXED (+ optional cyclic features)
and the V5 energy model (residential day/night steps, hour-deviation scale 0.1)."""
import numpy as np, pandas as pd
from common import test_mask_patterns, ROOT, NUM
from features import NUMERIC
from linmodels import fit_linear_artifact
from imputed import IMP_PARAMS
from devimpute import fit_dev_imputers, DEV_CODE, CYC_CODE, LEVELS, NUMS
from specs import V5

DEEP = dict(IMP_PARAMS, num_leaves=63, min_child_samples=5, n_estimators=1000, learning_rate=0.02)
SLOW = dict(IMP_PARAMS, n_estimators=1500, learning_rate=0.01)
CONFIGS = [(IMP_PARAMS, 11), (DEEP, 12), (SLOW, 13)]

def build(train, cyclic=False, prev_missing_offset=0.0):
    test = pd.read_csv(ROOT / "test.csv")
    P = test_mask_patterns(test); P = P[P.any(1)][:, [NUM.index(c) for c in NUMERIC]]  # mask-order fix
    X = train.drop(columns="energy_usage")
    lin = fit_linear_artifact(X, train["energy_usage"].to_numpy(), alpha=3.0, spec=V5)
    rows = pd.concat([X, test[X.columns]], ignore_index=True)
    dom = np.r_[np.zeros(len(X)), np.ones(len(test))]
    imps, tables = fit_dev_imputers(rows, dom, CONFIGS, P, copies=5, test_weight=2.0, cyclic=cyclic)
    return {"linear": lin, "imputers": imps, "impute_cols": NUMS, "domain": 1.0, "dev_tables": tables,
            "dev_levels": LEVELS, "cyclic": bool(cyclic), "prev_missing_offset": float(prev_missing_offset)}

PREDICT_CODE = DEV_CODE + CYC_CODE + '''
import lightgbm as lgb
# 1) fill missing numeric inputs: average of 3 target-free LightGBM imputers (domain flag = test distribution),
#    with deviation features (observed value minus its building/hour profile mean) and optional cyclic time features
F = make_features(test)
F["dom"] = artifact["domain"]
if artifact["cyclic"]:
    F = add_cyclic(F)
X = test.copy()
for col in artifact["impute_cols"]:
    miss = F[col].isna().to_numpy()
    if miss.any():
        Xm = add_dev(F, artifact["dev_tables"], artifact["dev_levels"], col).drop(columns=[col])[miss]
        X[col] = F[col].to_numpy()
        X.loc[miss, col] = np.mean([lgb.Booster(model_str=s).predict(Xm) for s in artifact["imputers"][col]], axis=0)
# 2) linear energy model (extrapolates linearly in every observed feature)
preds = predict_linear(artifact["linear"], X)
# 3) optional offset for rows whose previous_usage was missing (estimated on naturally-missing train rows)
preds = preds + artifact["prev_missing_offset"] * F["previous_usage"].isna().to_numpy()'''
