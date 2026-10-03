"""Fallback baseline: single LightGBM on row-local features (CV b03: 3.34 clean / 3.96 masked)."""
import lightgbm as lgb
from features import make_features

PARAMS = dict(n_estimators=2000, learning_rate=0.01, num_leaves=15, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4, random_state=42)

def fit_artifact(train):
    X = make_features(train); feats = list(X.columns)
    m = lgb.LGBMRegressor(**PARAMS).fit(X[feats], train["energy_usage"])
    return {"model": m, "features": feats}

PREDICT_CODE = "preds = artifact[\"model\"].predict(X_test)"
