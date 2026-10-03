"""Imputers + natural-missingness indicators for the OTHER columns (artificial masks keep indicator 0)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from proxy import evaluate, IMP_PARAMS, PN, ALL
from imputed import _mask_aug
from exp34_devfeat import DEEP, SLOW

def _keyarr(F, lv):
    cols = [F[x].cat.codes.to_numpy() if hasattr(F[x], "cat") else F[x].to_numpy() for x in lv]
    return pd.Index(list(zip(*cols)))

def nat_member(params, seed=11, copies=5, levels=(("building", "hour"), ("building",)), use_dev=True, n_only=False):
    def feats(F, nat, tabs, skip):
        G = F.copy()
        if use_dev:
            for i, lv in enumerate(levels):
                idx = _keyarr(F, lv)
                for c in ALL:
                    if c != skip: G[f"dev{i}_{c}"] = F[c].to_numpy() - tabs[i][c].reindex(idx).to_numpy()
        others = [c for c in ALL if c != skip]
        if n_only: G["nat_n"] = nat[others].sum(1).to_numpy()
        else:
            for c in others: G["nat_" + c] = nat[c].to_numpy()
        return G
    def fit(F0, col, w):
        tabs = [{c: pd.Series(F0[c].to_numpy()).groupby(_keyarr(F0, lv)).mean() for c in ALL} for lv in levels]
        nat0 = F0[ALL].isna().astype(float)
        F = _mask_aug(F0, col, PN, copies, seed=seed); nat = pd.concat([nat0] * (copies + 1), ignore_index=True)
        ww = np.tile(w, copies + 1); ok = F[col].notna().to_numpy()
        X = feats(F, nat, tabs, col).drop(columns=[col])
        m = lgb.LGBMRegressor(**params).fit(X[ok], F.loc[ok, col], sample_weight=ww[ok])
        def pred(G):
            # at evaluation the target col is artificially hidden; natural missingness of others = their NaN status
            natG = G[ALL].isna().astype(float)
            return m.predict(feats(G, natG, tabs, col).drop(columns=[col]))
        return pred
    return fit

g = sys.argv[1]
if g == "a": evaluate([nat_member(IMP_PARAMS, 11, 2)], 2.0, label="single dev+natind (vs dev 21.40)")
if g == "b": evaluate([nat_member(IMP_PARAMS, 11, 2, n_only=True)], 2.0, label="single dev+nat count")
if g == "c": evaluate([nat_member(IMP_PARAMS, 11, 5), nat_member(DEEP, 12, 5), nat_member(SLOW, 13, 5)], 2.0, label="c09 + natind (vs c09 20.87/27.63)")
if g == "ref":
    from exp34_devfeat import multi_member
    evaluate([multi_member(IMP_PARAMS, [["building", "hour"], ["building"]], 11, 2)], 2.0, label="single dev only (ref)")
