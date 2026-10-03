import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from common import load, cross_validate, test_mask_patterns
from imputed import ImputeLinear, impute, _mask_aug
from features import make_features, predict_linear
from specs import V4
tr, te = load(); P = test_mask_patterns(te); P = P[P.any(1)]
ALL = ["occupancy", "previous_usage", "temperature", "humidity"]
def enc(F):
    F = F.copy(); F["building"] = F.building.cat.codes; F["btype"] = F.btype.cat.codes; return F.fillna(-999)
class ImpKinds(ImputeLinear):
    def __init__(self, kind, **kw): super().__init__(V4, cols=ALL, extra_rows=te, aug_patterns=P, aug_copies=1, **kw); self.kind = kind
    def fit(self, X, y):
        super().fit(X, y)
        if self.kind in ("et", "lgb+et"):
            rows = pd.concat([X, te[X.columns]], ignore_index=True); F0 = make_features(rows); self.et = {}
            for col in ALL:
                F = _mask_aug(F0, col, P, 1, seed=11); ok = F[col].notna().to_numpy()
                self.et[col] = ExtraTreesRegressor(200, min_samples_leaf=5, max_features=0.7, n_jobs=6, random_state=0).fit(enc(F.drop(columns=[col]))[ok], F.loc[ok, col])
        return self
    def _impute(self, X):
        if self.kind == "chained":   # occupancy first, then previous_usage sees imputed occupancy
            X1 = impute(X, self.imp, cols=["occupancy", "temperature", "humidity"])
            X2 = impute(X1.assign(previous_usage=X["previous_usage"]), self.imp, cols=["previous_usage"])
            return X2
        out = impute(X, self.imp, cols=ALL) if self.kind != "et" else X.copy()
        if self.kind in ("et", "lgb+et"):
            F = make_features(X)
            for col in ALL:
                miss = F[col].isna().to_numpy()
                if miss.any():
                    pe = self.et[col].predict(enc(F.drop(columns=[col]))[miss])
                    out.loc[out.index[miss], col] = pe if self.kind == "et" else 0.5 * out.loc[out.index[miss], col].astype(float).to_numpy() + 0.5 * pe
        return out
    def predict(self, X): return predict_linear(self.art, self._impute(X))
for k in ["chained", "et", "lgb+et"]:
    cross_validate(f"k_{k}", lambda: ImpKinds(k), tr, te, repeats=2)
