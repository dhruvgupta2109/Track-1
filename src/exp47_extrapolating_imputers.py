"""Held-out, target-free comparison of c11 imputers with extrapolating imputers.

No test energy labels enter fitting or evaluation. Energy slopes used to score
feature errors come from the previously frozen, labelled c04 fit. Each fold excludes held-out train
and test rows, including from deviation tables. The linear component predicts
the missing feature; a LightGBM residual component can correct its bias. Fixed
50/50 blends are diagnostics, not fitted leaderboard weights. Cache errors so
paired uncertainty and natural-missingness slices remain inspectable.
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates._c11 import CONFIGS
from common import ROOT, NUM, apply_masks, load, test_mask_patterns
from devimpute import LEVELS, add_cyclic, add_dev, make_tables
from features import BUILDINGS, NUMERIC, make_features
from imputed import _mask_aug
from proxy import SL


def linear_inputs(F, col, means, scales):
    """Building schedules + linear observed deviations; NaNs get flags.

    All numeric centering/scaling comes from this training fold. Domain terms
    allow train/test means and slopes to differ. This design has no target input.
    """
    b = F.building.cat.codes.to_numpy()
    B = (b[:, None] == np.arange(len(BUILDINGS))).astype(float)
    h = F.hour.to_numpy()
    H = (h[:, None] == np.arange(24)).astype(float)
    dom = F.dom.to_numpy()[:, None]
    we = F.weekend.fillna(2 / 7).to_numpy()[:, None]
    parts = [B, B * dom, (B[:, :, None] * H[:, None, :]).reshape(len(F), -1) * .5,
             B * we, H * we * .5, F[["m_sin", "m_cos"]].fillna(0).to_numpy()]
    for c in NUMERIC:
        if c == col:
            continue
        v = F[c].to_numpy()
        mu = np.array([means[c].get(int(k), means[c][-1]) for k in b])
        sd = np.array([scales[c].get(int(k), scales[c][-1]) for k in b])
        z = np.nan_to_num((v - mu) / np.maximum(sd, 1))[:, None]
        miss = np.isnan(v).astype(float)[:, None]
        parts.extend([B * z, B * z * dom * .5, B * miss, B * miss * dom * .5])
    return np.concatenate(parts, axis=1)


def fit_linear(F0, F, col, ok, weights):
    means, scales = {}, {}
    for c in NUMERIC:
        g = F0[c].groupby(F0.building.cat.codes)
        means[c] = {**g.mean().to_dict(), -1: float(F0[c].mean())}
        scales[c] = {**g.std().to_dict(), -1: float(F0[c].std())}
    D = linear_inputs(F, col, means, scales)
    model = Ridge(alpha=20).fit(D[ok], F.loc[ok, col], sample_weight=weights[ok])
    return model.predict(D), lambda G: model.predict(linear_inputs(G, col, means, scales))


def main(nfolds, trio):
    tr, te = load()
    XTR = tr.drop(columns="energy_usage")
    XTE = te[XTR.columns]
    P = test_mask_patterns(te)
    PN = P[P.any(1)][:, [NUM.index(c) for c in NUMERIC]]
    kt = list(KFold(5, shuffle=True, random_state=1).split(XTR))
    ke = list(KFold(5, shuffle=True, random_state=0).split(XTE))
    configs = CONFIGS if trio else CONFIGS[:1]
    rows = []
    for k in range(nfolds):
        (ta, tb), (ea, eb) = kt[k], ke[k]
        train = pd.concat([XTR.iloc[ta], XTE.iloc[ea]], ignore_index=True)
        dom = np.r_[np.zeros(len(ta)), np.ones(len(ea))]
        F0 = add_cyclic(make_features(train).assign(dom=dom))
        tabs = make_tables(F0, LEVELS)
        w0 = np.where(dom == 1, 2., 1.)
        for col in NUMERIC:
            predictors = []
            for params, seed in configs:
                F = _mask_aug(F0, col, PN, 5, seed=seed)
                w = np.tile(w0, 6)
                ok = F[col].notna().to_numpy()
                A = add_dev(F, tabs, LEVELS, col).drop(columns=col)
                baseline = lgb.LGBMRegressor(**params).fit(A[ok], F.loc[ok, col], sample_weight=w[ok])
                lp, linear = fit_linear(F0, F, col, ok, w)
                residual = lgb.LGBMRegressor(**params).fit(A[ok], F.loc[ok, col] - lp[ok], sample_weight=w[ok])
                predictors.append((baseline, linear, residual))
            for pool, hold, ids, domain in [("test", XTE.iloc[eb].reset_index(drop=True), eb, 1),
                                           ("train", XTR.iloc[tb].reset_index(drop=True), tb, 0)]:
                valid = hold[col].notna().to_numpy()
                H = apply_masks(hold[valid], P, seed=100 + k)
                H[col] = np.nan
                G = add_cyclic(make_features(H).assign(dom=domain))
                A = add_dev(G, tabs, LEVELS, col).drop(columns=col)
                bp = np.mean([b.predict(A) for b, _, _ in predictors], axis=0)
                lp = np.mean([f(G) for _, f, _ in predictors], axis=0)
                rp = lp + np.mean([r.predict(A) for _, _, r in predictors], axis=0)
                true = hold.loc[valid, col].to_numpy()
                natural = hold.loc[valid, [c for c in NUMERIC if c != col]].isna().any(axis=1).to_numpy()
                slope = H.building_id.map(SL[col]).to_numpy()
                for variant, pred in [("c11_baseline", bp), ("linear", lp), ("linear_residual", rp),
                                      ("blend_linear", .5 * (bp + lp)), ("blend_residual", .5 * (bp + rp))]:
                    for i, e, s, nat in zip(np.asarray(ids)[valid], pred - true, slope, natural):
                        rows.append(dict(fold=k, pool=pool, feature=col, row=int(i), variant=variant,
                                         error=float(e), energy_error=float(e * s), naturally_incomplete=bool(nat)))
            print(f"fold {k + 1}/{nfolds} {col}: fitted and held out", flush=True)
    E = pd.DataFrame(rows)
    stem = f"extrapolating_imputers_{nfolds}fold_{'trio' if trio else 'single'}"
    E.to_csv(ROOT / "experiments" / f"{stem}_errors.csv.gz", index=False)
    results = {}
    for pool, select in [("test", E.pool == "test"), ("natmiss", (E.pool == "test") & E.naturally_incomplete),
                         ("train", E.pool == "train")]:
        S = E[select]
        T = S.assign(mse=S.energy_error ** 2).groupby(["variant", "feature"]).mse.mean().unstack()
        T["total"] = T.sum(axis=1)
        results[pool] = T.to_dict(orient="index")
        print(f"\n{pool} energy-weighted imputation MSE:\n{T.round(4).to_string()}", flush=True)
    for variant in E.variant.unique():
        if variant == "c11_baseline":
            continue
        T = E[E.pool == "test"].pivot(index=["fold", "feature", "row"], columns="variant", values="energy_error")
        d = T[variant] ** 2 - T.c11_baseline ** 2
        # Correlated feature errors within a row: aggregate before estimating uncertainty.
        byrow = d.groupby(level=["fold", "row"]).sum()
        results.setdefault("paired_test", {})[variant] = {
            "sum_error_mse_delta": float(byrow.mean()),
            "row_paired_se": float(byrow.std(ddof=1) / np.sqrt(len(byrow))),
            "fold_deltas": {int(k): float(v) for k, v in byrow.groupby(level="fold").mean().items()},
        }
    (ROOT / "experiments" / f"{stem}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--trio", action="store_true")
    args = parser.parse_args()
    main(args.folds, args.trio)
