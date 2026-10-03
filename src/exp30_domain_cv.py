import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from common import load, cross_validate, test_mask_patterns
from imputed import fit_imputers, impute
from linmodels import fit_linear_artifact
from features import predict_linear
from specs import V4
tr, te = load(); P = test_mask_patterns(te); P = P[P.any(1)]
ALL = ["occupancy", "previous_usage", "temperature", "humidity"]
class Dom:
    def fit(self, X, y):
        self.art = fit_linear_artifact(X, y, 3.0, V4)
        rows = pd.concat([X, te[X.columns]], ignore_index=True)
        self.imp = fit_imputers(rows, aug_patterns=P, aug_copies=1, cols=ALL, domain=np.r_[np.zeros(len(X)), np.ones(len(te))]); return self
    def predict(self, X): return predict_linear(self.art, impute(X, self.imp, cols=ALL, domain=0.0))
cross_validate("c05_domain_cv_dom0_5x4", Dom, tr, te, repeats=4)
