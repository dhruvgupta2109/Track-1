"""Domain-aware imputers: add is_test flag (0 train / 1 test) to imputer features; predict test-like rows with flag=1.
Scored on held-out TEST rows (features only) and held-out TRAIN rows (flag=0), energy-weighted."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from common import load, test_mask_patterns, apply_masks
from features import make_features
from imputed import _mask_aug, IMP_PARAMS
tr, te = load(); P = test_mask_patterns(te); Pn = P[P.any(1)]
ALL = ["occupancy", "previous_usage", "temperature", "humidity"]
SLOPE = {"occupancy": 0.10, "previous_usage": 0.34, "temperature": 2.0, "humidity": 0.06}
Xtr = tr.drop(columns="energy_usage"); Xte = te[Xtr.columns]
def fit(rows, dom, w=None, use_dom=True):
    F0 = make_features(rows)
    if use_dom: F0["dom"] = dom
    B = {}
    for col in ALL:
        F = _mask_aug(F0, col, Pn, 1, seed=11); ww = np.tile(np.ones(len(F0)) if w is None else w, 2); ok = F[col].notna().to_numpy()
        B[col] = lgb.LGBMRegressor(**IMP_PARAMS).fit(F.drop(columns=[col])[ok], F.loc[ok, col], sample_weight=ww[ok])
    return B
def err(B, hold, flag, k, use_dom):
    out = {}
    for col in ALL:
        ok = hold[col].notna().to_numpy(); H = apply_masks(hold[ok], P, seed=100 + k); H[col] = np.nan
        F = make_features(H)
        if use_dom: F["dom"] = flag
        out[col] = B[col].predict(F.drop(columns=[col])) - hold.loc[ok, col].to_numpy()
    return out
kt = list(KFold(5, shuffle=True, random_state=1).split(Xtr)); ke = list(KFold(5, shuffle=True, random_state=0).split(Xte))
res = []
for name, use_dom, tw in [("c04 pooled", False, 1.0), ("domain flag", True, 1.0), ("domain flag + test x2", True, 2.0)]:
    E = {}
    for k in range(5):
        (ta, tb), (ea, eb) = kt[k], ke[k]
        rows = pd.concat([Xtr.iloc[ta], Xte.iloc[ea]], ignore_index=True); dom = np.r_[np.zeros(len(ta)), np.ones(len(ea))]
        B = fit(rows, dom, np.where(dom == 1, tw, 1.0), use_dom)
        for pool, hold, flag in [("train", Xtr.iloc[tb].reset_index(drop=True), 0), ("test", Xte.iloc[eb].reset_index(drop=True), 1)]:
            for col, e in err(B, hold, flag, k, use_dom).items(): E.setdefault((pool, col), []).append(e)
    r = {"variant": name}
    for pool in ["train", "test"]:
        r[pool + "_energyMSE"] = sum(SLOPE[c] ** 2 * np.mean(np.concatenate(E[(pool, c)]) ** 2) for c in ALL)
        for c in ["occupancy", "previous_usage"]:
            e = np.concatenate(E[(pool, c)]); r[f"{pool}_{c[:4]}_rmse"] = np.sqrt(np.mean(e ** 2)); r[f"{pool}_{c[:4]}_bias"] = e.mean()
    res.append(r); print(pd.DataFrame([r]).round(3).to_string(index=False, header=len(res) == 1))
