"""c04: linear V4 (smooth type hour-profiles + shrunk per-hour deviations) on fully-imputed inputs.
Missing numeric values are filled by target-free LightGBM imputers (fitted on train+test FEATURES with
test-like mask augmentation), stored as LightGBM text models. CV 5x4: 3.048 / 3.228 / 3.364."""
import pandas as pd
from common import test_mask_patterns
from linmodels import fit_linear_artifact
from imputed import fit_imputers
from specs import V4

COLS = ["occupancy", "previous_usage", "temperature", "humidity"]

def fit_artifact(train):
    test = pd.read_csv(__import__("common").ROOT / "test.csv")
    P = test_mask_patterns(test); P = P[P.any(1)]
    X = train.drop(columns="energy_usage")
    lin = fit_linear_artifact(X, train["energy_usage"].to_numpy(), alpha=3.0, spec=V4)
    rows = pd.concat([X, test[X.columns]], ignore_index=True)
    imps = fit_imputers(rows, aug_patterns=P, aug_copies=1, cols=COLS)
    return {"linear": lin, "imputers": imps, "impute_cols": COLS}

PREDICT_CODE = '''import lightgbm as lgb
# 1) fill missing numeric inputs with target-free LightGBM imputers (conditional expectations)
F = make_features(test)
X = test.copy()
for col in artifact["impute_cols"]:
    miss = F[col].isna().to_numpy()
    if miss.any():
        booster = lgb.Booster(model_str=artifact["imputers"][col])
        X[col] = F[col].to_numpy()
        X.loc[miss, col] = booster.predict(F.drop(columns=[col])[miss])
# 2) linear model (extrapolates linearly in every observed feature)
preds = predict_linear(artifact["linear"], X)'''
