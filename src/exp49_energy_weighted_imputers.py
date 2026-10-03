"""Match imputer training loss to the pre-existing slope-weighted proxy.

Fixed squared per-building slopes weight each training feature error. Compare
the complete c11 trio against its cached paired held-out errors from exp47.
Feature values, rows, folds, augmentation and model settings are unchanged.
Weights derive from the previously frozen c04 energy fit on labelled training
data. No public-test energy labels or leaderboard fitting are used. The train
proxy is a feature holdout, not fully nested validation of the energy slopes.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates._c11 import CONFIGS
from common import ROOT, NUM, apply_masks, load, test_mask_patterns
from devimpute import LEVELS, add_cyclic, add_dev, make_tables
from features import NUMERIC, make_features
from imputed import _mask_aug
from proxy import SL


def main():
    tr, te = load()
    XTR = tr.drop(columns="energy_usage"); XTE = te[XTR.columns]
    P = test_mask_patterns(te)
    PN = P[P.any(1)][:, [NUM.index(c) for c in NUMERIC]]
    kt = list(KFold(5, shuffle=True, random_state=1).split(XTR))
    ke = list(KFold(5, shuffle=True, random_state=0).split(XTE))
    base = pd.read_csv(ROOT / "experiments/extrapolating_imputers_5fold_trio_errors.csv.gz")
    base = base[base.variant == "c11_baseline"].copy()
    rows = []
    for k, ((ta, tb), (ea, eb)) in enumerate(zip(kt, ke)):
        train = pd.concat([XTR.iloc[ta], XTE.iloc[ea]], ignore_index=True)
        dom = np.r_[np.zeros(len(ta)), np.ones(len(ea))]
        F0 = add_cyclic(make_features(train).assign(dom=dom))
        tabs = make_tables(F0, LEVELS)
        w0 = np.where(dom == 1, 2., 1.)
        for col in NUMERIC:
            # Shared slopes imply identical objectives; use the exact cached baseline.
            if len(set(SL[col].values())) == 1:
                continue
            sw = train.building_id.map(SL[col]).to_numpy() ** 2
            sw /= np.average(sw, weights=w0)
            models = []
            for params, seed in CONFIGS:
                F = _mask_aug(F0, col, PN, 5, seed=seed)
                w = np.tile(w0 * sw, 6); ok = F[col].notna().to_numpy()
                A = add_dev(F, tabs, LEVELS, col).drop(columns=col)
                m = lgb.LGBMRegressor(**params).fit(A[ok], F.loc[ok, col], sample_weight=w[ok])
                models.append(m)
            for pool, hold, ids, domain in [("test", XTE.iloc[eb].reset_index(drop=True), eb, 1),
                                           ("train", XTR.iloc[tb].reset_index(drop=True), tb, 0)]:
                valid = hold[col].notna().to_numpy()
                H = apply_masks(hold[valid], P, seed=100 + k); H[col] = np.nan
                G = add_cyclic(make_features(H).assign(dom=domain))
                A = add_dev(G, tabs, LEVELS, col).drop(columns=col)
                pred = np.mean([m.predict(A) for m in models], axis=0)
                true = hold.loc[valid, col].to_numpy()
                natural = hold.loc[valid, [c for c in NUMERIC if c != col]].isna().any(axis=1).to_numpy()
                slope = H.building_id.map(SL[col]).to_numpy()
                for i, e, s, nat in zip(np.asarray(ids)[valid], pred - true, slope, natural):
                    rows.append(dict(fold=k, pool=pool, feature=col, row=int(i), variant="energy_weighted",
                                     error=float(e), energy_error=float(e * s), naturally_incomplete=bool(nat)))
            print(f"fold {k + 1}/5 {col}: held out", flush=True)
    W = pd.DataFrame(rows)
    shared = base[~base.feature.isin(W.feature.unique())].assign(variant="energy_weighted")
    W = pd.concat([W, shared], ignore_index=True)
    blend = base.copy(); blend["variant"] = "blend_weighted"
    # Explicit index alignment; never rely on merge or input ordering for pairing.
    idx = ["fold", "pool", "feature", "row"]
    weighted = W.set_index(idx).reindex(pd.MultiIndex.from_frame(base[idx]))
    blend["error"] = .5 * (base.error.to_numpy() + weighted.error.to_numpy())
    blend["energy_error"] = .5 * (base.energy_error.to_numpy() + weighted.energy_error.to_numpy())
    E = pd.concat([base, W, blend], ignore_index=True)
    E.to_csv(ROOT / "experiments/energy_weighted_imputers_errors.csv.gz", index=False)
    result = {}
    for pool, sel in [("test", E.pool == "test"), ("natmiss", (E.pool == "test") & E.naturally_incomplete),
                      ("train", E.pool == "train")]:
        T = E[sel].assign(mse=lambda d: d.energy_error ** 2).groupby(["variant", "feature"]).mse.mean().unstack()
        T["total"] = T.sum(axis=1)
        result[pool] = T.to_dict(orient="index")
        print(f"\n{pool} energy-weighted imputation MSE:\n{T.round(4).to_string()}")
    (ROOT / "experiments/energy_weighted_imputers.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
