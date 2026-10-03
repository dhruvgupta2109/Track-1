"""c09: c08 (domain-aware imputer trio, test x2, mask aug x5) + deviation features for the imputers:
each observed feature minus its (building, hour) and (building) profile mean (target-free lookup tables).
Proxy: single imputer 22.65 -> 21.40 with these levels; trio (building,hour,weekend) variant 21.29 vs c08 22.21."""
import numpy as np, pandas as pd
from common import test_mask_patterns, ROOT
from linmodels import fit_linear_artifact
from imputed import IMP_PARAMS
from devimpute import fit_dev_imputers, DEV_CODE, LEVELS, NUMS
from specs import V4

DEEP = dict(IMP_PARAMS, num_leaves=63, min_child_samples=5, n_estimators=1000, learning_rate=0.02)
SLOW = dict(IMP_PARAMS, n_estimators=1500, learning_rate=0.01)
CONFIGS = [(IMP_PARAMS, 11), (DEEP, 12), (SLOW, 13)]

def fit_artifact(train):
    test = pd.read_csv(ROOT / "test.csv")
    P = test_mask_patterns(test); P = P[P.any(1)]
    X = train.drop(columns="energy_usage")
    lin = fit_linear_artifact(X, train["energy_usage"].to_numpy(), alpha=3.0, spec=V4)
    rows = pd.concat([X, test[X.columns]], ignore_index=True)
    dom = np.r_[np.zeros(len(X)), np.ones(len(test))]
    imps, tables = fit_dev_imputers(rows, dom, CONFIGS, P, copies=5, test_weight=2.0)
    return {"linear": lin, "imputers": imps, "impute_cols": NUMS, "domain": 1.0, "dev_tables": tables, "dev_levels": LEVELS}

PREDICT_CODE = DEV_CODE + '''
import lightgbm as lgb
# 1) fill missing numeric inputs: average of 3 target-free LightGBM imputers (domain flag = test distribution),
#    each also given deviation features (observed value minus its building/hour profile mean)
F = make_features(test)
F["dom"] = artifact["domain"]
X = test.copy()
for col in artifact["impute_cols"]:
    miss = F[col].isna().to_numpy()
    if miss.any():
        Xm = add_dev(F, artifact["dev_tables"], artifact["dev_levels"], col).drop(columns=[col])[miss]
        X[col] = F[col].to_numpy()
        X.loc[miss, col] = np.mean([lgb.Booster(model_str=s).predict(Xm) for s in artifact["imputers"][col]], axis=0)
# 2) linear model (extrapolates linearly in every observed feature)
preds = predict_linear(artifact["linear"], X)'''
