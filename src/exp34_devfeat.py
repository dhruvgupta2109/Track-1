"""Imputers with deviation features: x_c - mean_c(building, hour, weekend), table from feature rows only."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from proxy import evaluate, lgb_member, IMP_PARAMS, PN, ALL
from imputed import _mask_aug

def profile(F):
    k = [F.building.cat.codes, F.hour, F.weekend]
    return {c: F.groupby(k, observed=True)[c].mean() for c in ALL}

def add_dev(F, tab, skip):
    G = F.copy(); idx = pd.MultiIndex.from_arrays([F.building.cat.codes, F.hour, F.weekend])
    for c in ALL:
        if c != skip:
            G["dev_" + c] = F[c].to_numpy() - tab[c].reindex(idx).to_numpy()
    return G

def dev_member(params, seed=11, copies=1):
    def fit(F0, col, w):
        tab = profile(F0)
        F = _mask_aug(F0, col, PN, copies, seed=seed); ww = np.tile(w, copies + 1); ok = F[col].notna().to_numpy()
        F = add_dev(F, tab, col)
        m = lgb.LGBMRegressor(**params).fit(F.drop(columns=[col])[ok], F.loc[ok, col], sample_weight=ww[ok])
        return lambda G: m.predict(add_dev(G, tab, col).drop(columns=[col]))
    return fit

grp = sys.argv[1] if __name__ == "__main__" else ""
DEEP = dict(IMP_PARAMS, num_leaves=63, min_child_samples=5, n_estimators=1000, learning_rate=0.02)
SLOW = dict(IMP_PARAMS, n_estimators=1500, learning_rate=0.01)
if grp == "a":
    evaluate([lgb_member(IMP_PARAMS, 11, 2)], 2.0, label="single default aug2 (ref)")
    evaluate([dev_member(IMP_PARAMS, 11, 2)], 2.0, label="single default aug2 + dev feats")
if grp == "b":
    evaluate([dev_member(IMP_PARAMS, 11, 5), dev_member(DEEP, 12, 5), dev_member(SLOW, 13, 5)], 2.0, label="trio w2 aug5 + dev feats (vs c08 22.21)")
def profile2(F, keys, std=False, by_dom=False):
    k = [F.building.cat.codes] + [F[x] for x in keys] + ([F["dom"]] if by_dom else [])
    return {c: (F.groupby(k, observed=True)[c].mean(), F.groupby(k, observed=True)[c].std() if std else None) for c in ALL}
def add_dev2(F, tab, skip, keys, std=False, by_dom=False):
    G = F.copy(); idx = pd.MultiIndex.from_arrays([F.building.cat.codes] + [F[x] for x in keys] + ([F["dom"]] if by_dom else []))
    for c in ALL:
        if c != skip:
            mu, sd = tab[c]; d = F[c].to_numpy() - mu.reindex(idx).to_numpy()
            G["dev_" + c] = d
            if std: G["z_" + c] = d / np.maximum(sd.reindex(idx).to_numpy(), 1e-6)
    return G
def dev2_member(params, keys, std=False, by_dom=False, seed=11, copies=2):
    def fit(F0, col, w):
        tab = profile2(F0, keys, std, by_dom)
        F = _mask_aug(F0, col, PN, copies, seed=seed); ww = np.tile(w, copies + 1); ok = F[col].notna().to_numpy()
        F = add_dev2(F, tab, col, keys, std, by_dom)
        m = lgb.LGBMRegressor(**params).fit(F.drop(columns=[col])[ok], F.loc[ok, col], sample_weight=ww[ok])
        return lambda G: m.predict(add_dev2(G, tab, col, keys, std, by_dom).drop(columns=[col]))
    return fit
if grp == "c":
    evaluate([dev2_member(IMP_PARAMS, ["hour", "dow"])], 2.0, label="dev (bldg,hour,dow)")
    evaluate([dev2_member(IMP_PARAMS, ["hour", "weekend"], std=True)], 2.0, label="dev + z (bldg,hour,weekend)")
if grp == "d":
    evaluate([dev2_member(IMP_PARAMS, ["hour", "weekend"], by_dom=True)], 2.0, label="dev profile per domain")
    evaluate([dev2_member(IMP_PARAMS, ["hour"])], 2.0, label="dev (bldg,hour)")
def multi_member(params, levels, seed=11, copies=2):
    """levels: list of key-lists over columns of F (e.g. ['building','hour']); one dev feature per (level, column)."""
    def keyarr(F, lv): return pd.MultiIndex.from_arrays([F[x].cat.codes if hasattr(F[x], "cat") else F[x] for x in lv]) if len(lv) > 1 else pd.Index(F[lv[0]].cat.codes if hasattr(F[lv[0]], "cat") else F[lv[0]])
    def feats(F, tabs, skip):
        G = F.copy()
        for i, lv in enumerate(levels):
            idx = keyarr(F, lv)
            for c in ALL:
                if c != skip: G[f"dev{i}_{c}"] = F[c].to_numpy() - tabs[i][c].reindex(idx).to_numpy()
        return G
    def fit(F0, col, w):
        tabs = [{c: F0.groupby(keyarr(F0, lv), observed=True)[c].mean() if False else F0[c].groupby(keyarr(F0, lv)).mean() for c in ALL} for lv in levels]
        F = _mask_aug(F0, col, PN, copies, seed=seed); ww = np.tile(w, copies + 1); ok = F[col].notna().to_numpy()
        F = feats(F, tabs, col)
        m = lgb.LGBMRegressor(**params).fit(F.drop(columns=[col])[ok], F.loc[ok, col], sample_weight=ww[ok])
        return lambda G: m.predict(feats(G, tabs, col).drop(columns=[col]))
    return fit
if grp == "e":
    evaluate([multi_member(IMP_PARAMS, [["btype", "hour"]])], 2.0, label="dev (type,hour)")
    evaluate([multi_member(IMP_PARAMS, [["building"]])], 2.0, label="dev (building)")
if grp == "f":
    evaluate([multi_member(IMP_PARAMS, [["building", "hour"], ["building"]])], 2.0, label="dev (bldg,hour)+(bldg)")
    evaluate([multi_member(IMP_PARAMS, [["building", "hour"], ["building", "hour", "weekend"]])], 2.0, label="dev (bldg,hour)+(bldg,hour,we)")
if grp == "g":
    L = [["building", "hour"], ["building"]]
    evaluate([multi_member(IMP_PARAMS, L, 11, 5), multi_member(DEEP, L, 12, 5), multi_member(SLOW, L, 13, 5)], 2.0, label="c09 cfg: trio w2 aug5 dev(bh)+(b)")
if grp == "h":
    L = [["building", "hour"], ["building"]]
    evaluate([lgb_member(IMP_PARAMS, 11, 5), lgb_member(DEEP, 12, 5), lgb_member(SLOW, 13, 5),
              multi_member(IMP_PARAMS, L, 11, 5), multi_member(DEEP, L, 12, 5), multi_member(SLOW, L, 13, 5)], 2.0, label="c10 cfg: c08 trio + c09 trio")
