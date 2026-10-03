"""Imputer combos scored on held-out TEST rows (and train rows); configs averaged when given as a list."""
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
DEEP = dict(IMP_PARAMS, num_leaves=63, min_child_samples=5, n_estimators=1000, learning_rate=0.02)
SLOW = dict(IMP_PARAMS, n_estimators=1500, learning_rate=0.01)
def run(configs, label):
    E = {c: [] for c in ALL}; Et = {c: [] for c in ALL}
    for k in range(5):
        (ta, tb), (ea, eb) = kt[k], ke[k]
        rows = pd.concat([Xtr.iloc[ta], Xte.iloc[ea]], ignore_index=True); dom = np.r_[np.zeros(len(ta)), np.ones(len(ea))]
        F0 = make_features(rows); F0["dom"] = dom
        for col in ALL:
            preds = {"test": [], "train": []}
            for params, tw, seed in configs:
                F = _mask_aug(F0, col, Pn, 1, seed=seed); w = np.tile(np.where(dom == 1, tw, 1.0), 2); ok = F[col].notna().to_numpy()
                m = lgb.LGBMRegressor(**params).fit(F.drop(columns=[col])[ok], F.loc[ok, col], sample_weight=w[ok])
                for pool, hold, flag in [("test", Xte.iloc[eb].reset_index(drop=True), 1), ("train", Xtr.iloc[tb].reset_index(drop=True), 0)]:
                    ok2 = hold[col].notna().to_numpy(); H = apply_masks(hold[ok2], P, seed=100 + k); H[col] = np.nan
                    G = make_features(H); G["dom"] = flag; preds[pool].append((m.predict(G.drop(columns=[col])), hold.loc[ok2, col].to_numpy()))
            for pool, store in [("test", E), ("train", Et)]:
                store[col].append(np.mean([p for p, _ in preds[pool]], 0) - preds[pool][0][1])
    f = lambda S: sum(SLOPE[c] ** 2 * np.mean(np.concatenate(S[c]) ** 2) for c in ALL)
    print(f"{label:38s} test energyMSE {f(E):.3f} | train {f(Et):.3f}", flush=True)
run([(IMP_PARAMS, 1.0, 11)], "c05 default (ref)")
run([(DEEP, 1.0, 11)], "deep")
run([(DEEP, 2.0, 11)], "deep + test x2")
run([(IMP_PARAMS, 1.0, 11), (DEEP, 1.0, 12), (SLOW, 1.0, 13)], "avg(default, deep, slow)")
run([(IMP_PARAMS, 2.0, 11), (DEEP, 2.0, 12), (SLOW, 2.0, 13)], "avg(...) + test x2")
