import sys; sys.path.insert(0, "src")
import numpy as np
from common import load, cross_validate
from linmodels import fit_linear_artifact
from features import predict_linear
class LogLin:
    def fit(self, X, y):
        self.art = fit_linear_artifact(X, np.log(y), 1.0, clip=(-1e9, 1e9))
        r = np.log(y) - predict_linear(self.art, X); self.s2 = r.var(); return self
    def predict(self, X): return np.exp(predict_linear(self.art, X) + self.s2 / 2)
class WLS:
    def fit(self, X, y):
        a0 = fit_linear_artifact(X, y, 1.0); f = predict_linear(a0, X)
        self.art = fit_linear_artifact(X, y, 1.0, weights=1.0 / np.maximum(f, 10) ** 2 * 2500); return self
    def predict(self, X): return predict_linear(self.art, X)
tr, te = load()
cross_validate("l08_log_target", LogLin, tr, te)
cross_validate("l09_wls_level", WLS, tr, te)
