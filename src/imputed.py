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


def fit_imputers(feature_rows, params=IMP_PARAMS, aug_patterns=None, aug_copies=0, cols=IMPUTE,
                 row_weight=None, n_bags=1, domain=None):
    """row_weight: optional per-row weights for feature_rows. n_bags>1 -> list of seeded boosters (averaged)."""
    F0 = make_features(feature_rows)
    if domain is not None:  # 0 = train distribution, 1 = test distribution (feature only, no target)
        F0["dom"] = np.asarray(domain, dtype=float)
    w0 = np.ones(len(F0)) if row_weight is None else np.asarray(row_weight, float)
    boosters = {}
    for col in cols:
        F = F0 if not aug_copies else _mask_aug(F0, col, aug_patterns, aug_copies, seed=11)
        w = np.tile(w0, len(F) // len(F0))
        ok = F[col].notna().to_numpy()
        X = F.drop(columns=[col])
        bags = []
        for s in range(n_bags):
            m = lgb.LGBMRegressor(**dict(params, random_state=params.get("random_state", 0) + s))
            m.fit(X[ok], F.loc[ok, col], sample_weight=w[ok])
            bags.append(m.booster_.model_to_string())
        boosters[col] = bags[0] if n_bags == 1 else bags
    return boosters


def impute(df, boosters, cols=IMPUTE, domain=None):
    """Fill missing occupancy / previous_usage (computed from the ORIGINAL observed features)."""
    out = df.copy()
    F = make_features(df)
    if domain is not None:
        F["dom"] = float(domain)
    for col in cols:
        miss = F[col].isna().to_numpy()
        if miss.any():
            X = F.drop(columns=[col])[miss]
            strs = boosters[col] if isinstance(boosters[col], list) else [boosters[col]]
            out.loc[out.index[miss], col] = np.mean([lgb.Booster(model_str=s).predict(X) for s in strs], axis=0)
    return out


class ImputeLinear:
    def __init__(self, spec, alpha=3.0, extra_rows=None, cols=IMPUTE, blend=1.0, params=IMP_PARAMS,
                 aug_patterns=None, aug_copies=0, both_missing_fallback=False, fit_weights=None, lin_kw=None):
        self.spec, self.alpha, self.extra, self.cols, self.blend, self.params = spec, alpha, extra_rows, cols, blend, params
        self.aug_patterns, self.aug_copies, self.fallback = aug_patterns, aug_copies, both_missing_fallback
        self.fit_weights, self.lin_kw = fit_weights, (lin_kw or {})

    def fit(self, X, y):
        w = None if self.fit_weights is None else np.asarray(self.fit_weights)[X.index.to_numpy()]
        self.art = fit_linear_artifact(X, y, self.alpha, self.spec, weights=w, **self.lin_kw)
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
