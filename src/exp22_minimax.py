"""Imputation error under two private-test hypotheses: test-like rows vs train-like rows.
Both held-out sets are excluded from imputer training (5-fold over each pool jointly)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from common import load, test_mask_patterns, apply_masks
from imputed import fit_imputers, impute
tr, te = load(); P = test_mask_patterns(te); Pn = P[P.any(1)]
ALL = ["occupancy", "previous_usage", "temperature", "humidity"]
SLOPE = {"occupancy": 0.10, "previous_usage": 0.35, "temperature": 2.0, "humidity": 0.06}
Xtr = tr.drop(columns="energy_usage"); Xte = te[Xtr.columns]
kt = list(KFold(5, shuffle=True, random_state=1).split(Xtr)); ke = list(KFold(5, shuffle=True, random_state=0).split(Xte))
res = []
for name, tw in [("train-only", 0.0), ("train+test x1 (c04)", 1.0), ("x2", 2.0), ("x4", 4.0)]:
    E = {("train", c): [] for c in ALL} | {("test", c): [] for c in ALL}
    for k in range(5):
        (ta, tb), (ea, eb) = kt[k], ke[k]
        rows = pd.concat([Xtr.iloc[ta], Xte.iloc[ea]], ignore_index=True)
        w = np.r_[np.ones(len(ta)), np.full(len(ea), tw)]
        keep = w > 0
        imp = fit_imputers(rows[keep].reset_index(drop=True), aug_patterns=Pn, aug_copies=1, cols=ALL, row_weight=w[keep])
        for pool, hold in [("train", Xtr.iloc[tb].reset_index(drop=True)), ("test", Xte.iloc[eb].reset_index(drop=True))]:
            for col in ALL:
                ok = hold[col].notna().to_numpy()
                H = apply_masks(hold[ok], P, seed=100 + k); H[col] = np.nan
                E[(pool, col)].append(impute(H, imp, cols=[col])[col].to_numpy() - hold.loc[ok, col].to_numpy())
    r = {"variant": name}
    for pool in ["train", "test"]:
        r[pool + "_energyMSE"] = sum(SLOPE[c] ** 2 * np.mean(np.concatenate(E[(pool, c)]) ** 2) for c in ALL)
        r[pool + "_occ_rmse"] = np.sqrt(np.mean(np.concatenate(E[(pool, "occupancy")]) ** 2))
        r[pool + "_prev_rmse"] = np.sqrt(np.mean(np.concatenate(E[(pool, "previous_usage")]) ** 2))
    res.append(r)
print(pd.DataFrame(res).round(3).to_string(index=False))
