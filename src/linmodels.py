"""Training side of the pattern-submodel linear model (inference lives in features.py)."""
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from features import NUMERIC, DEFAULT_SPEC, clean_input, linear_design, predict_linear


def all_patterns():
    return [tuple(c for c in NUMERIC if c in s) for r in range(5) for s in itertools.combinations(NUMERIC, r)]


def fit_linear_artifact(df, y, alpha=3.0, spec=DEFAULT_SPEC, clip=(0.0, 400.0), weights=None, estimator="ridge", huber_eps=1.5,
                        building_wls=False):
    if building_wls:  # two-stage: inverse per-building residual variance from a first unweighted fit
        first = fit_linear_artifact(df, y, alpha, spec, clip)
        r = np.asarray(y, float) - predict_linear(first, df)
        var = pd.Series(r ** 2).groupby(df["building_id"].to_numpy()).mean()
        w = 1.0 / pd.Series(df["building_id"].to_numpy()).map(var).fillna(var.mean()).to_numpy()
        weights = w / w.mean() if weights is None else weights * w / w.mean()
    """One ridge per missingness pattern, each trained on all rows where its columns are observed.
    Stored as plain numpy (coef, intercept) -> no library-version deserialisation risk."""
    c = clean_input(df); y = np.asarray(y, dtype=float)
    coefs = {}
    for p in all_patterns():
        need = [col for col in NUMERIC if col not in p]
        ok = c[need].notna().all(1).to_numpy()
        D = linear_design(c[ok], drop=set(p), spec=spec)
        sw = None if weights is None else weights[ok]
        if estimator == "huber":
            from sklearn.linear_model import HuberRegressor
            m = HuberRegressor(epsilon=huber_eps, alpha=alpha / ok.sum(), max_iter=2000).fit(D, y[ok], sample_weight=sw)
        else:
            m = Ridge(alpha=alpha).fit(D, y[ok], sample_weight=sw)
        coefs[p] = (m.coef_.astype(float), float(m.intercept_))
    return {"coefs": coefs, "spec": spec, "clip": clip}


class PatternLinear:
    def __init__(self, alpha=3.0, spec=DEFAULT_SPEC):
        self.alpha, self.spec = alpha, spec
    def fit(self, X, y):
        self.art = fit_linear_artifact(X, y, self.alpha, self.spec); return self
    def predict(self, X):
        return predict_linear(self.art, X)
