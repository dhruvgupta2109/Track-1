"""Imputation accuracy on held-out PUBLIC-TEST rows (features only; no labels involved).
5-fold over test rows: imputer fit on train (+ the other 4/5 of test for the transductive variant),
then each observed feature of held-out test rows is hidden (one at a time and with test-like co-masks)
and re-predicted. Errors are converted to their energy impact using the c04 linear slopes."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from common import load, test_mask_patterns, apply_masks, NUM
from imputed import fit_imputers, impute
tr, te = load(); P = test_mask_patterns(te); Pn = P[P.any(1)]
ALL = ["occupancy", "previous_usage", "temperature", "humidity"]
Xtr = tr.drop(columns="energy_usage")
res = {}
for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(te)):
    hold = te.iloc[b].reset_index(drop=True)
    variants = {"train_only": Xtr, "train+test": pd.concat([Xtr, te.iloc[a][Xtr.columns]], ignore_index=True)}
    for vn, rows in variants.items():
        imp = fit_imputers(rows, aug_patterns=Pn, aug_copies=1, cols=ALL)
        for col in ALL:
            ok = hold[col].notna().to_numpy()
            H = apply_masks(hold[ok], P, seed=100 + k)  # test-like co-missingness on other columns
            H[col] = np.nan
            pred = impute(H, imp, cols=[col])[col].to_numpy()
            err = pred - hold.loc[ok, col].to_numpy()
            res.setdefault((vn, col), []).append(err)
rows = []
for (vn, col), errs in res.items():
    e = np.concatenate(errs); rows.append((col, vn, len(e), np.sqrt(np.mean(e ** 2)), np.mean(e), np.median(np.abs(e))))
print(pd.DataFrame(rows, columns=["feature", "imputer", "n", "rmse", "bias", "medAE"]).sort_values(["feature", "imputer"]).round(3).to_string(index=False))
