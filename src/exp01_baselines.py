import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from common import load, cross_validate, TARGET
from features import make_features

class Wrap:
    def __init__(self, model, feats=make_features, dense=False):
        self.model, self.feats, self.dense = model, feats, dense
    def fit(self, X, y):
        F = self.feats(X)
        if self.dense: F = self.dense(F)
        self.model.fit(F, y); return self
    def predict(self, X):
        F = self.feats(X)
        if self.dense: F = self.dense(F)
        return self.model.predict(F)

class PrevOnly:
    def fit(self, X, y):
        self.fill = np.mean(y); return self
    def predict(self, X):
        return X.previous_usage.fillna(self.fill).to_numpy()

if __name__ == "__main__":
  tr, te = load()
  cross_validate("b00_prev_usage_copy", PrevOnly, tr, te)
  cross_validate("b01_hgb_default", lambda: Wrap(HistGradientBoostingRegressor(random_state=42, categorical_features="from_dtype")), tr, te)
  cross_validate("b02_lgb_default", lambda: Wrap(lgb.LGBMRegressor(verbose=-1, n_jobs=4)), tr, te)
  cross_validate("b03_lgb_slow", lambda: Wrap(lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.01, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4)), tr, te)
  cross_validate("b04_lgb_linear_tree", lambda: Wrap(lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.02, num_leaves=15, linear_tree=True, linear_lambda=1.0, verbose=-1, n_jobs=4)), tr, te)
