"""Imputer variants judged by imputation error on held-out PUBLIC-TEST rows (features only)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from common import load, test_mask_patterns, apply_masks
from imputed import fit_imputers, impute, IMP_PARAMS
tr, te = load(); P = test_mask_patterns(te); Pn = P[P.any(1)]
ALL = ["occupancy", "previous_usage", "temperature", "humidity"]
SLOPE = {"occupancy": 0.10, "previous_usage": 0.35, "temperature": 2.0, "humidity": 0.06}  # approx c04 energy slopes
Xtr = tr.drop(columns="energy_usage")
V = {
 "c04 (train+test)": dict(),
 "test rows x2": dict(test_w=2.0),
 "test rows x4": dict(test_w=4.0),
 "bag5": dict(n_bags=5),
 "shallow reg": dict(params=dict(IMP_PARAMS, num_leaves=15, min_child_samples=30, reg_lambda=5.0)),
 "deeper": dict(params=dict(IMP_PARAMS, num_leaves=63, min_child_samples=5, n_estimators=1000, learning_rate=0.02)),
 "aug2": dict(aug_copies=2),
}
out = []
for name, cfg in V.items():
    errs = {c: [] for c in ALL}
    for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(te)):
        hold = te.iloc[b].reset_index(drop=True)
        rows = pd.concat([Xtr, te.iloc[a][Xtr.columns]], ignore_index=True)
        w = np.r_[np.ones(len(Xtr)), np.full(len(a), cfg.get("test_w", 1.0))]
        imp = fit_imputers(rows, params=cfg.get("params", IMP_PARAMS), aug_patterns=Pn, aug_copies=cfg.get("aug_copies", 1),
                           cols=ALL, row_weight=w, n_bags=cfg.get("n_bags", 1))
        for col in ALL:
            ok = hold[col].notna().to_numpy()
            H = apply_masks(hold[ok], P, seed=100 + k); H[col] = np.nan
            errs[col].append(impute(H, imp, cols=[col])[col].to_numpy() - hold.loc[ok, col].to_numpy())
    r = {"variant": name}
    for c in ALL:
        e = np.concatenate(errs[c]); r[c] = np.sqrt(np.mean(e ** 2)); r[c[:4] + "_bias"] = e.mean()
    r["energy_mse_sum"] = sum(SLOPE[c] ** 2 * r[c] ** 2 for c in ALL)
    out.append(r); print(pd.DataFrame([r]).round(3).to_string(index=False, header=len(out) == 1))
