"""c09 pipeline (mask order FIXED) under NATURAL missingness: OOF residual bias per naturally-missing feature."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from common import load, cross_validate, test_mask_patterns, NUM
from features import NUMERIC, predict_linear, make_features, clean_input
from linmodels import fit_linear_artifact
from imputed import IMP_PARAMS
from devimpute import fit_dev_imputers, add_dev, LEVELS
from specs import V4
import lightgbm as lgb
tr, te = load()
P = test_mask_patterns(te); P = P[P.any(1)][:, [NUM.index(c) for c in NUMERIC]]   # FIX: columns in NUMERIC order
DEEP = dict(IMP_PARAMS, num_leaves=63, min_child_samples=5, n_estimators=1000, learning_rate=0.02)
SLOW = dict(IMP_PARAMS, n_estimators=1500, learning_rate=0.01)
CONFIGS = [(IMP_PARAMS, 11), (DEEP, 12), (SLOW, 13)]
class C09:
    def __init__(self, dom_pred=0.0): self.dom_pred = dom_pred
    def fit(self, X, y):
        self.lin = fit_linear_artifact(X, y, 3.0, V4)
        rows = pd.concat([X, te[X.columns]], ignore_index=True); dom = np.r_[np.zeros(len(X)), np.ones(len(te))]
        self.imps, self.tabs = fit_dev_imputers(rows, dom, CONFIGS, P, copies=5, test_weight=2.0); return self
    def predict(self, X):
        F = make_features(X); F["dom"] = self.dom_pred; Xf = X.copy()
        for col in NUMERIC:
            miss = F[col].isna().to_numpy()
            if miss.any():
                Xm = add_dev(F, self.tabs, LEVELS, col).drop(columns=[col])[miss]
                Xf[col] = F[col].to_numpy()
                Xf.loc[Xf.index[miss], col] = np.mean([lgb.Booster(model_str=s).predict(Xm) for s in self.imps[col]], axis=0)
        return predict_linear(self.lin, Xf)
cross_validate("mnar_c09fixed_dom0_5x2", lambda: C09(0.0), tr, te, repeats=2)
