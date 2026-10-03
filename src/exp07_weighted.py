import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from common import load, cross_validate
from linmodels import fit_linear_artifact
from features import predict_linear
W = pd.Series(np.load("experiments/shift_w.npy"))
class WLin:
    def __init__(self, power=1.0, alpha=3.0): self.p, self.a = power, alpha
    def fit(self, X, y):
        w = W.loc[X.index].to_numpy() ** self.p
        self.art = fit_linear_artifact(X, y, self.a, weights=w); return self
    def predict(self, X): return predict_linear(self.art, X)
tr, te = load()
cross_validate("l03b_v2_default", lambda: WLin(0.0), tr, te, save=False)
cross_validate("l10_shift_weighted_fit", lambda: WLin(1.0), tr, te)
cross_validate("l11_shift_weighted_fit_sqrt", lambda: WLin(0.5), tr, te)
