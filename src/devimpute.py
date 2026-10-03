"""Domain-aware imputers with deviation features (c09).

Deviation feature: x_c - mean of x_c in the row's profile cell (e.g. building x hour), from a target-free lookup
table built on train + test feature rows. Spikes co-occur across features, so an observed feature's deviation is
evidence about a missing one. Tables are plain dicts so the artifact has no pandas-pickle dependency.
The DEV_CODE string is the exact inference implementation copied into the prediction notebook.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import make_features
from imputed import _mask_aug

NUMS = ["occupancy", "previous_usage", "temperature", "humidity"]
LEVELS = [["building", "hour"], ["building"]]

DEV_CODE = '''
def _dev_keys(F, level):
    cols = [F[x].cat.codes.to_numpy() if hasattr(F[x], "cat") else F[x].to_numpy() for x in level]
    return list(zip(*cols))

def add_dev(F, tables, levels, skip):
    """Append deviation features (fixed order: level, then column) - identical in training and inference."""
    G = F.copy()
    for i, level in enumerate(levels):
        keys = _dev_keys(F, level)
        for c in ["occupancy", "previous_usage", "temperature", "humidity"]:
            if c != skip:
                mu = np.array([tables[i][c].get(k, np.nan) for k in keys], dtype=float)
                G[f"dev{i}_{c}"] = F[c].to_numpy(dtype=float) - mu
    return G
'''
exec(DEV_CODE)


def make_tables(F, levels=LEVELS):
    out = []
    for level in levels:
        keys = _dev_keys(F, level)
        t = {}
        for c in NUMS:
            s = pd.Series(F[c].to_numpy(dtype=float)).groupby(pd.Series(keys)).mean()
            t[c] = {tuple(int(v) if isinstance(v, (np.integer, int)) else float(v) for v in (k if isinstance(k, tuple) else (k,))): float(m)
                    for k, m in s.items() if np.isfinite(m)}
        out.append(t)
    return out




CYC_CODE = '''
def add_cyclic(F):
    """sin/cos of hour and month for the imputers (appended after 'dom')."""
    G = F.copy(); h = G["hour"].to_numpy(dtype=float); m = G["month"].to_numpy(dtype=float)
    G["h_sin"], G["h_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    G["m_sin"], G["m_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    return G
'''
exec(CYC_CODE)


def fit_dev_imputers(rows, domain, configs, aug_patterns, copies, test_weight, levels=LEVELS, cyclic=False):
    F0 = make_features(rows)
    F0["dom"] = np.asarray(domain, dtype=float)
    if cyclic:
        F0 = add_cyclic(F0)
    tables = make_tables(F0, levels)
    w0 = np.where(np.asarray(domain) == 1, test_weight, 1.0)
    imps = {c: [] for c in NUMS}
    for params, seed in configs:
        for col in NUMS:
            F = _mask_aug(F0, col, aug_patterns, copies, seed=seed)
            w = np.tile(w0, copies + 1); ok = F[col].notna().to_numpy()
            X = add_dev(F, tables, levels, col).drop(columns=[col])
            m = lgb.LGBMRegressor(**params).fit(X[ok], F.loc[ok, col], sample_weight=w[ok])
            imps[col].append(m.booster_.model_to_string())
    return imps, tables
