"""Anomaly-aware linear model: adds each feature's deviation from its conditional expectation
d_c = x_c - E[x_c | other features]  (target-free; E[] from the LightGBM imputers)
as extra linear terms, so spike rows may respond with a different slope than ordinary variation."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from features import make_features, clean_input, linear_design
from imputed import fit_imputers, impute, IMP_PARAMS

DEV_COLS = ["previous_usage", "occupancy", "temperature"]

def expected(df, boosters, cols):
    """E[x_c | other observed features] for every row (not just missing ones)."""
    F = make_features(df); out = {}
    for c in cols:
        strs = boosters[c] if isinstance(boosters[c], list) else [boosters[c]]
        out[c] = np.mean([lgb.Booster(model_str=s).predict(F.drop(columns=[c])) for s in strs], axis=0)
    return out

def dev_block(c_clean, E, cols, mode):
    blocks = []
    for col in cols:
        d = np.nan_to_num(c_clean[col].to_numpy() - E[col], nan=0.0)   # missing -> imputed -> deviation 0
        if mode == "split":
            blocks += [np.maximum(d, 0), np.minimum(d, 0)]
        else:
            blocks.append(d)
    return np.stack(blocks, 1)

class AnomalyLinear:
    def __init__(self, spec, extra_rows, aug_patterns, alpha=3.0, cols=DEV_COLS, mode="split", dev_scale=1.0, inner=5):
        self.spec, self.extra, self.P, self.alpha, self.cols, self.mode, self.sc, self.inner = spec, extra_rows, aug_patterns, alpha, cols, mode, dev_scale, inner
        self.ALL = ["occupancy", "previous_usage", "temperature", "humidity"]
    def _design(self, X, E):
        Xi = impute(X, self.imp, cols=self.ALL)
        from features import VALID
        for col, (lo, hi) in VALID.items():
            Xi[col] = pd.to_numeric(Xi[col], errors="coerce").clip(lo, hi)
        c = clean_input(Xi)
        return np.hstack([linear_design(c, spec=self.spec), self.sc * dev_block(clean_input(X), E, self.cols, self.mode)])
    def fit(self, X, y):
        X = X.reset_index(drop=True); y = np.asarray(y, float)
        rows = pd.concat([X, self.extra[X.columns]], ignore_index=True)
        self.imp = fit_imputers(rows, aug_patterns=self.P, aug_copies=1, cols=self.ALL)
        # out-of-fold expectations for training rows (in-sample LGB would shrink deviations)
        E = {c: np.zeros(len(X)) for c in self.cols}
        for a, b in KFold(self.inner, shuffle=True, random_state=3).split(X):
            r2 = pd.concat([X.iloc[a], self.extra[X.columns]], ignore_index=True)
            bi = fit_imputers(r2, aug_patterns=self.P, aug_copies=1, cols=self.cols)
            e = expected(X.iloc[b], bi, self.cols)
            for c in self.cols: E[c][b] = e[c]
        D = self._design(X, E)
        self.m = Ridge(alpha=self.alpha).fit(D, y)
        return self
    def predict(self, X):
        E = expected(X, self.imp, self.cols)
        return np.clip(self.m.predict(self._design(X, E)), 0, 400)
