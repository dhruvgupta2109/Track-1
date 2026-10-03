import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from common import load, cross_validate, NUM
from features import make_features
from linmodels import PatternRidge
from exp02_missing import aug

class RidgeThenGBM:
    """Ridge base + LightGBM on residuals. Residuals for GBM training are computed out-of-fold
    inside the training data (inner 5-fold) to avoid fitting GBM to in-sample-shrunk residuals."""
    def __init__(self, params, inner_oof=True, augment=None, add_base=True):
        self.params, self.inner_oof, self.augment, self.add_base = params, inner_oof, augment, add_base
    def fit(self, X, y):
        from sklearn.model_selection import KFold
        X = X.reset_index(drop=True); y = np.asarray(y)
        self.r = PatternRidge(3.0).fit(X, y)
        if self.inner_oof:
            base = np.zeros(len(X))
            for a, b in KFold(5, shuffle=True, random_state=1).split(X):
                base[b] = PatternRidge(3.0).fit(X.iloc[a], y[a]).predict(X.iloc[b])
        else:
            base = self.r.predict(X)
        D = X.copy(); D["energy_usage"] = y; D["_base"] = base
        if self.augment is not None:
            D = self.augment(D, seed=5)
            # base prediction for masked copies must come from the ridge for their pattern
            masked = D.index >= len(X)
            D.loc[masked, "_base"] = self.r.predict(D.loc[masked, X.columns])  # approx (in-sample)
        F = make_features(D[X.columns])
        if self.add_base: F["_base"] = D["_base"].to_numpy()
        self.g = lgb.LGBMRegressor(**self.params).fit(F, D["energy_usage"].to_numpy() - D["_base"].to_numpy())
        return self
    def predict(self, X):
        b = self.r.predict(X); F = make_features(X)
        if self.add_base: F["_base"] = b
        return b + self.g.predict(F)

P = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=50, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, reg_lambda=5, verbose=-1, n_jobs=4, random_state=0)
if __name__ == "__main__":
    tr, te = load()
    cross_validate("h01_ridge_gbm_resid", lambda: RidgeThenGBM(P), tr, te)
    cross_validate("h02_ridge_gbm_resid_aug", lambda: RidgeThenGBM(P, augment=aug(1.0)), tr, te)
    P2 = dict(P, n_estimators=800, learning_rate=0.01)
    cross_validate("h03_ridge_gbm_resid_slow", lambda: RidgeThenGBM(P2), tr, te)
