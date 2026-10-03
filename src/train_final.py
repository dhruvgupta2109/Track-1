"""Rebuild c17's complete statistical pipeline from train.csv and test.csv.

Reproduces the fixed experiments 41/43 without loading their fitted outputs or
any frozen model. Statistical settings and seeds are preserved. Public scores
do not enter calibration. Prediction wrapper changes live in c18_private_safe.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates._c11 import CONFIGS
from common import NUM, apply_masks, folds, test_mask_patterns
from devimpute import LEVELS, NUMS, add_cyclic, add_dev, make_tables, fit_dev_imputers
from features import NUMERIC, make_features, predict_linear
from imputed import IMP_PARAMS, _mask_aug
from linmodels import fit_linear_artifact
from specs import V5

STEPS = {"occupancy": 5., "previous_usage": 1., "temperature": .5, "humidity": 1.}


def impute_inputs(rows, art, domain):
    F = add_cyclic(make_features(rows).assign(dom=domain))
    out = rows.copy()
    for col in art["impute_cols"]:
        miss = F[col].isna().to_numpy()
        if miss.any():
            Xm = add_dev(F, art["dev_tables"], art["dev_levels"], col).drop(columns=col)[miss]
            out[col] = F[col].to_numpy()
            out.loc[out.index[miss], col] = np.mean([
                lgb.Booster(model_str=s).predict(Xm) for s in art["imputers"][col]], axis=0)
    return out


def conditional_errors(train, test):
    """Exact experiment-41 five-fold feature errors; no energy labels."""
    P = test_mask_patterns(test)
    PN = P[P.any(1)][:, [NUM.index(c) for c in NUMERIC]]
    X = pd.concat([train.drop(columns="energy_usage"), test[train.columns.drop("energy_usage")]], ignore_index=True)
    domain = np.r_[np.zeros(len(train)), np.ones(len(test))]
    errors = {c: np.full(len(X), np.nan) for c in NUMERIC}
    for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(X)):
        F0 = add_cyclic(make_features(X.iloc[a]).assign(dom=domain[a]))
        tables = make_tables(F0, LEVELS)
        w0 = np.where(domain[a] == 1, 2., 1.)
        H = X.iloc[b].reset_index(drop=True)
        for col in NUMERIC:
            F = _mask_aug(F0, col, PN, 2, seed=11)
            w = np.tile(w0, 3); observed = F[col].notna().to_numpy()
            A = add_dev(F, tables, LEVELS, col).drop(columns=col)
            m = lgb.LGBMRegressor(**dict(IMP_PARAMS, n_jobs=6)).fit(
                A[observed], F.loc[observed, col], sample_weight=w[observed])
            valid = H[col].notna().to_numpy()
            masked = apply_masks(H[valid], P, seed=100 + k); masked[col] = np.nan
            G = add_cyclic(make_features(masked).assign(dom=domain[b][valid]))
            pred = m.predict(add_dev(G, tables, LEVELS, col).drop(columns=col))
            errors[col][np.asarray(b)[valid]] = H.loc[valid, col].to_numpy() - pred
        print(f"conditional-variance fold {k + 1}/5", flush=True)
    E = pd.DataFrame(errors).assign(domain=domain, building_id=X.building_id.to_numpy())
    return E


def estimate_mnar(train, core, errors):
    """Exact skeptical zero-centred empirical-Bayes rule used by c17."""
    X = train.drop(columns="energy_usage"); y = train.energy_usage.to_numpy()
    XI = impute_inputs(X, core, 0.)
    missing = X[NUMERIC].isna().to_numpy()
    any_missing = missing.any(axis=1)
    oof = np.zeros((4, len(X)))
    for r, k, a, b in folds(len(X), 5, 4):
        art = fit_linear_artifact(X.iloc[a], y[a], alpha=3., spec=V5)
        oof[r, b] = predict_linear(art, XI.iloc[b])
        if k == 4:
            print(f"energy calibration repeat {r + 1}/4", flush=True)
    residual = y - oof.mean(axis=0)
    noise_var = float(np.mean(residual[~any_missing] ** 2))
    SD = {d: errors[errors.domain == d].groupby("building_id")[NUMERIC].agg(
        lambda s: np.sqrt(np.nanmean(s ** 2))) for d in (0, 1)}
    estimates = {}
    for j, col in enumerate(NUMERIC):
        single = missing[:, j] & (missing.sum(axis=1) == 1)
        up = XI.loc[single].copy(); down = up.copy(); h = STEPS[col]
        up[col] += h; down[col] -= h
        slope = (predict_linear(core["linear"], up) - predict_linear(core["linear"], down)) / (2 * h)
        g = slope * np.array([SD[0].loc[b, col] for b in X.loc[single, "building_id"]])
        weights = 1 / (noise_var + g ** 2)
        denominator = np.sum(weights * g ** 2)
        estimates[col] = (float(np.sum(weights * g * residual[single]) / denominator),
                          float(np.sqrt(1 / denominator)))
    kappa = np.array([estimates[c][0] for c in NUMERIC])
    variances = np.array([estimates[c][1] ** 2 for c in NUMERIC])
    grid = np.linspace(0, 1., 2001)
    v = variances[None, :] + grid[:, None]
    ll = -.5 * np.log(v).sum(axis=1) - .5 * (kappa[None, :] ** 2 / v).sum(axis=1)
    tau2 = float(grid[np.argmax(ll)])
    shrunk = kappa * tau2 / (tau2 + variances)
    return {
        "mnar_kappa": dict(zip(NUMERIC, shrunk.tolist())),
        "mnar_sd": {c: {b: float(v) for b, v in SD[1][c].items()} for c in NUMERIC},
        "mnar_sd_default": {c: float(SD[1][c].mean()) for c in NUMERIC},
        "mnar_step": dict(STEPS),
        "fallback_mean": float(y.mean()),
    }, {"raw_kappa_se": estimates, "tau2": tau2, "complete_oof_rmse": float(np.sqrt(noise_var))}


def rebuild_statistical_model(train, test):
    # Match the fixed c11 design while respecting the caller's supplied test frame.
    # The older c11 package builder reads test.csv directly from the repository.
    P = test_mask_patterns(test)
    PN = P[P.any(1)][:, [NUM.index(c) for c in NUMERIC]]
    X = train.drop(columns="energy_usage")
    linear = fit_linear_artifact(X, train.energy_usage.to_numpy(), alpha=3., spec=V5)
    rows = pd.concat([X, test[X.columns]], ignore_index=True)
    domain = np.r_[np.zeros(len(X)), np.ones(len(test))]
    imputers, tables = fit_dev_imputers(rows, domain, CONFIGS, PN, copies=5, test_weight=2., cyclic=True)
    core = {"linear": linear, "imputers": imputers, "impute_cols": NUMS, "domain": 1.,
            "dev_tables": tables, "dev_levels": LEVELS, "cyclic": True, "prev_missing_offset": 0.}
    errors = conditional_errors(train, test)
    correction, diagnostics = estimate_mnar(train, core, errors)
    return {**core, **correction}, diagnostics
