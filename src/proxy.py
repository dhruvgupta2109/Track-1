"""Label-free proxy validated by the public LB: energy-weighted imputation MSE on held-out TEST rows.

For each fold over test rows, imputers are fitted on train + the other test rows; every observed numeric value of the
held-out test rows is hidden (with test-like co-missingness on the other columns) and re-predicted. Errors are weighted
by the squared c04 energy slope of that feature for that row's building, since the energy model is linear.
Also reports the same on held-out train rows (domain flag 0) as a robustness check.
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.model_selection import KFold
sys.path.insert(0, str(Path(__file__).parent))
from common import load, test_mask_patterns, apply_masks
from features import make_features, BUILDINGS
from imputed import _mask_aug, IMP_PARAMS as _IP
IMP_PARAMS = dict(_IP, n_jobs=3)

TR, TE = load(); P = test_mask_patterns(TE)
# NOTE: P columns follow common.NUM; _mask_aug iterates features.NUMERIC -> reorder (mask-order bug fix).
# Set PROXY_LEGACY_MASK=1 to reproduce the original (buggy) order used for c05-c09.
import os as _os
from features import NUMERIC as _NUMERIC
from common import NUM as _NUM
PN = P[P.any(1)] if _os.environ.get("PROXY_LEGACY_MASK") == "1" else P[P.any(1)][:, [_NUM.index(c) for c in _NUMERIC]]
ALL = ["occupancy", "previous_usage", "temperature", "humidity"]
XTR = TR.drop(columns="energy_usage"); XTE = TE[XTR.columns]
KT = list(KFold(5, shuffle=True, random_state=1).split(XTR)); KE = list(KFold(5, shuffle=True, random_state=0).split(XTE))

def _slopes():
    import joblib
    from interpret import effects
    e = effects(joblib.load(Path(__file__).parents[1] / "package" / "c04_linear_imputed" / "model.pkl"))
    occ = (e["per_100_occupants"] / 100).to_dict(); tmp = e["per_degC_at_28"].to_dict()
    return {"occupancy": occ, "temperature": tmp, "previous_usage": {b: 0.34 for b in BUILDINGS}, "humidity": {b: 0.06 for b in BUILDINGS}}
SL = _slopes()

def lgb_member(params, seed=11, copies=1):
    def fit(F0, col, w):
        F = _mask_aug(F0, col, PN, copies, seed=seed); ww = np.tile(w, copies + 1); ok = F[col].notna().to_numpy()
        m = lgb.LGBMRegressor(**params).fit(F.drop(columns=[col])[ok], F.loc[ok, col], sample_weight=ww[ok])
        return lambda G: m.predict(G.drop(columns=[col]))
    return fit

def evaluate(members, test_w=1.0, train_w=1.0, use_dom=True, label="", folds=5, verbose=True):
    E = {("test", c): [] for c in ALL} | {("train", c): [] for c in ALL}
    for k in range(folds):
        (ta, tb), (ea, eb) = KT[k], KE[k]
        rows = pd.concat([XTR.iloc[ta], XTE.iloc[ea]], ignore_index=True); dom = np.r_[np.zeros(len(ta)), np.ones(len(ea))]
        keep = np.where(dom == 1, test_w, train_w) > 0
        F0 = make_features(rows[keep].reset_index(drop=True)); w = np.where(dom == 1, test_w, train_w)[keep]
        if use_dom: F0["dom"] = dom[keep]
        for col in ALL:
            preds = [m(F0, col, w) for m in members]
            for pool, hold, flag in [("test", XTE.iloc[eb].reset_index(drop=True), 1), ("train", XTR.iloc[tb].reset_index(drop=True), 0)]:
                ok = hold[col].notna().to_numpy(); H = apply_masks(hold[ok], P, seed=100 + k); H[col] = np.nan
                G = make_features(H)
                if use_dom: G["dom"] = flag
                p = np.mean([f(G) for f in preds], 0)
                s = H.building_id.map(SL[col]).fillna(np.mean(list(SL[col].values()))).to_numpy()
                E[(pool, col)].append((s * (p - hold.loc[ok, col].to_numpy())) ** 2)
                if pool == "test":  # rows that ALSO have a genuinely missing value elsewhere (like real missing rows)
                    nat = hold[ok][[c for c in ALL if c != col]].isna().any(axis=1).to_numpy()
                    E.setdefault(("natmiss", col), []).append(((s * (p - hold.loc[ok, col].to_numpy())) ** 2)[nat])
    res = {pool: sum(np.mean(np.concatenate(E[(pool, c)])) for c in ALL) for pool in ["test", "train", "natmiss"]}
    per = {c: np.mean(np.concatenate(E[("test", c)])) for c in ALL}
    if verbose:
        print(f"{label:42s} TEST proxy {res['test']:.3f} | NATMISS {res['natmiss']:.3f} | train {res['train']:.3f} | test per-feature "
              + " ".join(f"{c[:4]} {v:.2f}" for c, v in per.items()), flush=True)
    return res
