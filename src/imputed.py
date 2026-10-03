"""Linear model + nonlinear imputation of missing occupancy / previous_usage.

The linear model is linear in occupancy and previous_usage, so E[y | observed] = f(x_obs, E[occ | x_obs]).
Better conditional expectations for the missing feature translate directly into better predictions,
while extrapolation in every OBSERVED feature stays linear.

Imputers are target-free (they predict a feature from the other features), so they may be fitted on
train + test feature rows. In CV the validation rows are excluded from imputer training.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import make_features, clean_input, predict_linear
from linmodels import fit_linear_artifact

IMPUTE = ["occupancy", "previous_usage"]
IMP_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=4, random_state=0)


def _mask_aug(F, col, patterns, copies, seed):
    """Copies of F with test-like missingness on the OTHER numeric columns (target col kept)."""
    from features import NUMERIC
    parts = [F]
    rng = np.random.default_rng(seed)
    for i in range(copies):
        G = F.copy()
        m = patterns[rng.integers(0, len(patterns), len(G))]
        for j, c in enumerate(NUMERIC):
            if c != col:
                G.loc[m[:, j], c] = np.nan
        parts.append(G)
    return pd.concat(parts, ignore_index=True)


def fit_imputers(feature_rows, params=IMP_PARAMS, aug_patterns=None, aug_copies=0, cols=IMPUTE):
    F0 = make_features(feature_rows)
    boosters = {}
    for col in cols:
        F = F0 if not aug_copies else _mask_aug(F0, col, aug_patterns, aug_copies, seed=11)
        ok = F[col].notna()
        X = F.drop(columns=[col])
        m = lgb.LGBMRegressor(**params).fit(X[ok], F.loc[ok, col])
        boosters[col] = m.booster_.model_to_string()
    return boosters


def impute(df, boosters, cols=IMPUTE):
    """Fill missing occupancy / previous_usage (computed from the ORIGINAL observed features)."""
    out = df.copy()
    F = make_features(df)
    for col in cols:
        miss = F[col].isna().to_numpy()
        if miss.any():
            b = lgb.Booster(model_str=boosters[col])
            X = F.drop(columns=[col])[miss]
            out.loc[out.index[miss], col] = b.predict(X)
    return out


class ImputeLinear:
    def __init__(self, spec, alpha=3.0, extra_rows=None, cols=IMPUTE, blend=1.0, params=IMP_PARAMS,
                 aug_patterns=None, aug_copies=0, both_missing_fallback=False):
        self.spec, self.alpha, self.extra, self.cols, self.blend, self.params = spec, alpha, extra_rows, cols, blend, params
        self.aug_patterns, self.aug_copies, self.fallback = aug_patterns, aug_copies, both_missing_fallback

    def fit(self, X, y):
        self.art = fit_linear_artifact(X, y, self.alpha, self.spec)
        rows = X if self.extra is None else pd.concat([X, self.extra[X.columns]], ignore_index=True)
        self.imp = fit_imputers(rows, self.params, self.aug_patterns, self.aug_copies, self.cols)
        return self

    def predict(self, X):
        p_pat = predict_linear(self.art, X)
        p_imp = predict_linear(self.art, impute(X, self.imp, self.cols))
        p = self.blend * p_imp + (1 - self.blend) * p_pat
        if self.fallback:
            both = X[IMPUTE].isna().all(axis=1).to_numpy()
            p[both] = p_pat[both]
        return p
