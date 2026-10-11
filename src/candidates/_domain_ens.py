"""Shared builder for domain-aware imputer ensembles (c06, c07)."""
import numpy as np, pandas as pd
from common import test_mask_patterns, ROOT
from linmodels import fit_linear_artifact
from imputed import fit_imputers, IMP_PARAMS
from specs import V4

COLS = ["occupancy", "previous_usage", "temperature", "humidity"]
DEEP = dict(IMP_PARAMS, num_leaves=63, min_child_samples=5, n_estimators=1000, learning_rate=0.02)
SLOW = dict(IMP_PARAMS, n_estimators=1500, learning_rate=0.01)
CONFIGS = [(IMP_PARAMS, 11), (DEEP, 12), (SLOW, 13)]

def build(train, test_weight=1.0, configs=None, copies=1):
    configs = configs or CONFIGS
    test = pd.read_csv(ROOT / "test.csv")
    P = test_mask_patterns(test); P = P[P.any(1)]
    X = train.drop(columns="energy_usage")
    lin = fit_linear_artifact(X, train["energy_usage"].to_numpy(), alpha=3.0, spec=V4)
    rows = pd.concat([X, test[X.columns]], ignore_index=True)
    dom = np.r_[np.zeros(len(X)), np.ones(len(test))]
    w = np.where(dom == 1, test_weight, 1.0)
    imps = {c: [] for c in COLS}
    for params, seed in configs:
        one = fit_imputers(rows, params=params, aug_patterns=P, aug_copies=copies, cols=COLS, domain=dom, row_weight=w, aug_seed=seed)
        for c in COLS:
            imps[c].append(one[c])
    return {"linear": lin, "imputers": imps, "impute_cols": COLS, "domain": 1.0}

PREDICT_CODE = '''import lightgbm as lgb
# 1) fill missing numeric inputs: average of 3 target-free LightGBM imputers (domain flag = test distribution)
F = make_features(test)
F["dom"] = artifact["domain"]
X = test.copy()
for col in artifact["impute_cols"]:
    miss = F[col].isna().to_numpy()
    if miss.any():
        Xm = F.drop(columns=[col])[miss]
        X[col] = F[col].to_numpy()
        X.loc[miss, col] = np.mean([lgb.Booster(model_str=s).predict(Xm) for s in artifact["imputers"][col]], axis=0)
# 2) linear model (extrapolates linearly in every observed feature)
preds = predict_linear(artifact["linear"], X)'''
