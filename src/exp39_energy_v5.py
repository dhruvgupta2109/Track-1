import sys; sys.path.insert(0, "src")
from common import load, cross_validate
from linmodels import fit_linear_artifact
from features import predict_linear
from specs import V4, V5
tr, te = load()
class Lin:
    def __init__(self, spec, wls): self.spec, self.wls = spec, wls
    def fit(self, X, y): self.a = fit_linear_artifact(X, y, 3.0, self.spec, building_wls=self.wls); return self
    def predict(self, X): return predict_linear(self.a, X)
cross_validate("e_V4_5x4", lambda: Lin(V4, False), tr, te, repeats=4)
cross_validate("e_V4_wls_5x4", lambda: Lin(V4, True), tr, te, repeats=4)
cross_validate("e_V5_5x4", lambda: Lin(V5, False), tr, te, repeats=4)
cross_validate("e_V5_wls_5x4", lambda: Lin(V5, True), tr, te, repeats=4)
