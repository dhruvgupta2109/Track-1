"""c05: c04 + domain-aware imputers. Imputers get a feature 'dom' (0 = train rows, 1 = test rows) and predict
with dom=1, so missing inputs are filled with TEST-distribution conditional means (test has ~4x the spike rate).
Held-out test-row imputation (energy-weighted MSE): 26.7 -> 24.1; train-like rows 13.99 -> 13.68; occupancy bias -4.7 -> -0.4."""
import numpy as np, pandas as pd
from common import test_mask_patterns, ROOT
from linmodels import fit_linear_artifact
from imputed import fit_imputers
from specs import V4

COLS = ["occupancy", "previous_usage", "temperature", "humidity"]

def fit_artifact(train):
    test = pd.read_csv(ROOT / "test.csv")
    P = test_mask_patterns(test); P = P[P.any(1)]
    X = train.drop(columns="energy_usage")
    lin = fit_linear_artifact(X, train["energy_usage"].to_numpy(), alpha=3.0, spec=V4)
    rows = pd.concat([X, test[X.columns]], ignore_index=True)
    dom = np.r_[np.zeros(len(X)), np.ones(len(test))]
    imps = fit_imputers(rows, aug_patterns=P, aug_copies=1, cols=COLS, domain=dom)
    return {"linear": lin, "imputers": imps, "impute_cols": COLS, "domain": 1.0}

PREDICT_CODE = '''import lightgbm as lgb
# 1) fill missing numeric inputs with target-free LightGBM imputers (test-distribution conditional means)
F = make_features(test)
F["dom"] = artifact["domain"]  # imputers were trained with a train(0)/test(1) domain flag
X = test.copy()
for col in artifact["impute_cols"]:
    miss = F[col].isna().to_numpy()
    if miss.any():
        booster = lgb.Booster(model_str=artifact["imputers"][col])
        X[col] = F[col].to_numpy()
        X.loc[miss, col] = booster.predict(F.drop(columns=[col])[miss])
# 2) linear model (extrapolates linearly in every observed feature)
preds = predict_linear(artifact["linear"], X)'''
