"""Tune domain-aware imputers by energy-weighted imputation MSE on held-out TEST rows (features only)."""
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
kt = list(KFold(5, shuffle=True, random_state=1).split(Xtr)); ke = list(KFold(5, shuffle=True, random_state=0).split(Xte))
def run(params, copies=1, tw=1.0, label=""):
    E = {c: [] for c in ALL}; Et = {c: [] for c in ALL}
    for k in range(5):
        (ta, tb), (ea, eb) = kt[k], ke[k]
        rows = pd.concat([Xtr.iloc[ta], Xte.iloc[ea]], ignore_index=True); dom = np.r_[np.zeros(len(ta)), np.ones(len(ea))]
        F0 = make_features(rows); F0["dom"] = dom
        for col in ALL:
            F = _mask_aug(F0, col, Pn, copies, seed=11); w = np.tile(np.where(dom == 1, tw, 1.0), copies + 1); ok = F[col].notna().to_numpy()
            m = lgb.LGBMRegressor(**params).fit(F.drop(columns=[col])[ok], F.loc[ok, col], sample_weight=w[ok])
            for pool, hold, flag, store in [("test", Xte.iloc[eb].reset_index(drop=True), 1, E), ("train", Xtr.iloc[tb].reset_index(drop=True), 0, Et)]:
                ok2 = hold[col].notna().to_numpy(); H = apply_masks(hold[ok2], P, seed=100 + k); H[col] = np.nan
                G = make_features(H); G["dom"] = flag
                store[col].append(m.predict(G.drop(columns=[col])) - hold.loc[ok2, col].to_numpy())
    te_mse = sum(SLOPE[c] ** 2 * np.mean(np.concatenate(E[c]) ** 2) for c in ALL)
    tr_mse = sum(SLOPE[c] ** 2 * np.mean(np.concatenate(Et[c]) ** 2) for c in ALL)
    print(f"{label:40s} test energyMSE {te_mse:.3f} | train {tr_mse:.3f} | occ {np.sqrt(np.mean(np.concatenate(E['occupancy'])**2)):.2f} prev {np.sqrt(np.mean(np.concatenate(E['previous_usage'])**2)):.2f}", flush=True)
B = dict(IMP_PARAMS)
run(B, label="c05 default")
run(dict(B, num_leaves=63, min_child_samples=5, n_estimators=1000, learning_rate=0.02), label="deeper")
run(dict(B, num_leaves=15, min_child_samples=20, n_estimators=1200, learning_rate=0.02), label="shallower longer")
run(dict(B, n_estimators=1500, learning_rate=0.01), label="slow")
run(B, copies=0, label="no mask aug")
run(B, copies=2, label="mask aug x2")
run(dict(B, objective="huber", alpha=30.0), label="huber obj")
