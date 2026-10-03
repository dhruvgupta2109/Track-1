"""Can c04's OOF residual be predicted out-of-sample? (2nd-level CV on OOF residuals)"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from evalkit import load_oof, Y, TR, applied_masks
from features import make_features
from common import NUM
oof, oofm = load_oof("final_cmp_C04_5x4")
F = make_features(TR)
Fn = F.copy(); Fn["building"] = Fn.building.cat.codes; Fn["btype"] = Fn.btype.cat.codes
def score(X, r, mk, name):
    p = np.zeros(len(r))
    for a, b in KFold(5, shuffle=True, random_state=42).split(X):
        p[b] = mk().fit(X.iloc[a], r[a]).predict(X.iloc[b])
    gain = np.sqrt(np.mean(r ** 2)) - np.sqrt(np.mean((r - p) ** 2))
    best = max(((a_, np.sqrt(np.mean((r - a_ * p) ** 2))) for a_ in np.linspace(0, 1, 11)), key=lambda t: -t[1])
    print(f"{name:45s} resid rmse {np.sqrt(np.mean(r**2)):.4f} -> {np.sqrt(np.mean((r-p)**2)):.4f} (gain {gain:+.4f}); best shrink a={best[0]:.1f} -> {best[1]:.4f}; R2 {1-np.mean((r-p)**2)/np.var(r):+.4f}")
L = lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, n_jobs=6)
E = lambda: ExtraTreesRegressor(300, min_samples_leaf=30, max_features=0.6, n_jobs=6, random_state=0)
# clean residuals (avg over repeats reduces noise of the target slightly; use repeat 0 and mean)
r0 = Y - oof[0]; rbar = Y - oof.mean(0)
X = F.copy(); X["pred"] = oof.mean(0)
score(X, rbar, L, "clean | LGB on features+pred")
score(Fn.assign(pred=oof.mean(0)).fillna(-999), rbar, E, "clean | ExtraTrees")
# masked residuals: include mask flags as features (masked inputs)
M = applied_masks(4)
for r in [0]:
    Xm = F.copy()
    for j, c in enumerate(NUM):
        Xm.loc[M[r][:, j], c] = np.nan; Xm["m_" + c] = M[r][:, j].astype(float)
    Xm["pred"] = oofm[r]
    score(Xm, Y - oofm[r], L, f"masked rep{r} | LGB on masked features+flags")
    Xe = Xm.copy(); Xe["building"] = Xe.building.cat.codes; Xe["btype"] = Xe.btype.cat.codes
    score(Xe.fillna(-999), Y - oofm[r], E, f"masked rep{r} | ExtraTrees")
