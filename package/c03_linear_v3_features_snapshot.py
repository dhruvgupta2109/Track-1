"""Row-local feature engineering + linear-model inference. This exact code is copied into the
prediction notebook, so it may only use numpy / pandas.

Robustness contract (the private test "may contain different observations and edge cases"):
every row gets a finite prediction even with missing / malformed / unseen values in ANY column.
"""
import numpy as np
import pandas as pd

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
BUILDINGS = ["ADM_A", "BUS_A", "BUS_B", "ENG_A", "ENG_B", "LEC_A", "LIB_A",
             "RES_A", "RES_B", "SCI_A", "SCI_B", "SPT_A"]
TYPES = ["Administration", "Business", "Engineering", "LectureHall", "Library",
         "Residential", "Science", "Sports"]
BUILDING_TYPE = {"ADM_A": "Administration", "BUS_A": "Business", "BUS_B": "Business",
                 "ENG_A": "Engineering", "ENG_B": "Engineering", "LEC_A": "LectureHall",
                 "LIB_A": "Library", "RES_A": "Residential", "RES_B": "Residential",
                 "SCI_A": "Science", "SCI_B": "Science", "SPT_A": "Sports"}
NUMERIC = ["previous_usage", "occupancy", "temperature", "humidity"]
# physically plausible ranges; values outside are treated as sensor errors -> missing
VALID = {"previous_usage": (0, 1000), "occupancy": (0, 10000), "temperature": (-20, 60), "humidity": (0, 100)}


def _num(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def clean_input(df):
    """Sanitise raw columns; anything unusable becomes NaN (handled downstream)."""
    out = pd.DataFrame(index=df.index)
    get = lambda c: df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
    out["building_id"] = get("building_id").astype(str).str.strip().str.upper()
    out["building_type"] = get("building_type").astype(str).str.strip()
    hour = _num(get("hour"))
    out["hour"] = hour.where(hour.between(0, 23)).round()
    month = _num(get("month"))
    out["month"] = month.where(month.between(1, 12)).round()
    day = get("day_of_week").astype(str).str.strip().str[:3].str.lower()
    out["dow"] = day.map({d[:3].lower(): i for i, d in enumerate(DAYS)}).astype(float)
    for c in NUMERIC:
        v = _num(get(c))
        lo, hi = VALID[c]
        out[c] = v.where(v.between(lo, hi))
    return out


def make_features(df):
    """Tabular features for tree models (used by the GBM baseline)."""
    c = clean_input(df)
    out = pd.DataFrame(index=df.index)
    out["building"] = pd.Categorical(c["building_id"], categories=BUILDINGS)
    out["btype"] = pd.Categorical(c["building_type"], categories=TYPES)
    out["hour"] = c["hour"]
    out["dow"] = c["dow"]
    out["weekend"] = (c["dow"] >= 5).astype(float).where(c["dow"].notna())
    out["month"] = c["month"]
    for col in ["temperature", "humidity", "occupancy", "previous_usage"]:
        out[col] = c[col]
    return out


def _soft_building(c):
    """n x 12 building weights: one-hot if known, else uniform over buildings of the given type,
    else uniform over all buildings (=> prediction is the average building's expectation)."""
    n = len(c)
    W = np.zeros((n, len(BUILDINGS)))
    bid = c["building_id"].to_numpy(); bt = c["building_type"].to_numpy()
    for i in range(n):
        if bid[i] in BUILDINGS:
            W[i, BUILDINGS.index(bid[i])] = 1.0
        else:
            same = [j for j, b in enumerate(BUILDINGS) if BUILDING_TYPE[b] == bt[i]]
            W[i, same if same else slice(None)] = 1.0 / (len(same) if same else len(BUILDINGS))
    return W


def _soft_onehot(v, k):
    v = np.asarray(v, dtype=float)
    out = np.full((len(v), k), 1.0 / k)
    ok = ~np.isnan(v)
    out[ok] = 0.0
    out[np.where(ok)[0], v[ok].astype(int)] = 1.0
    return out


DEFAULT_SPEC = {"hours": "type", "weekend": "building",
                "slopes": {"previous_usage": "shared", "occupancy": "building",
                           "temperature": "building", "humidity": "shared"},
                "temp2": True, "season": True}


def linear_design(c, drop=(), spec=DEFAULT_SPEC):
    """Design matrix from a clean_input() frame. `drop` = numeric columns unavailable for this
    missingness-pattern sub-model. Categorical unknowns enter as expectations (soft one-hots)."""
    if spec.get("missing_spec") and set(drop) & {"occupancy", "previous_usage"}:
        # sub-models without occupancy/previous_usage must rebuild them from time x building -> richer design
        spec = {**spec, **spec["missing_spec"]}
    n = len(c)
    B = _soft_building(c)
    T = B @ np.array([[BUILDING_TYPE[b] == t for t in TYPES] for b in BUILDINGS], dtype=float)
    G = {"building": B, "type": T, "shared": np.ones((n, 1))}
    K = spec.get("hour_fourier", 0)
    if K:  # smooth daily profile: K harmonics (missing hour -> 0 = daily average)
        hh = c["hour"].to_numpy()
        H = np.concatenate([f(2 * np.pi * k * hh / 24)[:, None] for k in range(1, K + 1) for f in (np.sin, np.cos)], 1)
        H = np.nan_to_num(H, nan=0.0)
    else:
        H = _soft_onehot(c["hour"], 24)
    dow = c["dow"].to_numpy()
    we = np.where(np.isnan(dow), 2 / 7, (dow >= 5).astype(float))[:, None]
    cols = [B, (G[spec["hours"]][:, :, None] * H[:, None, :]).reshape(n, -1), G[spec["weekend"]] * we]
    if spec.get("weekend_hours"):  # separate daily shape on weekends
        Gw = G[spec["weekend_hours"]]
        cols.append((Gw[:, :, None] * (H * we)[:, None, :]).reshape(n, -1))
    for col in NUMERIC:
        if col not in drop:
            cols.append(G[spec["slopes"][col]] * c[col].to_numpy()[:, None])
    if spec.get("season", True):
        m = c["month"].to_numpy()
        sc = np.stack([np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)], 1)
        cols.append(np.nan_to_num(sc, nan=0.0))
    if spec.get("temp2", True) and "temperature" not in drop:
        cols.append(((c["temperature"].to_numpy() - 28.0) ** 2)[:, None])
    return np.concatenate(cols, 1)


def pattern_keys(c):
    miss = c[NUMERIC].isna().to_numpy()
    return [tuple(col for col, mm in zip(NUMERIC, row) if mm) for row in miss]


def predict_linear(art, df):
    """Pattern-submodel linear prediction: each row uses the model fitted without its missing
    numeric columns. art = {"coefs": {pattern: (coef, intercept)}, "spec": ..., "clip": (lo, hi)}."""
    c = clean_input(df)
    keys = pattern_keys(c)
    preds = np.empty(len(c))
    for k in set(keys):
        idx = np.array([kk == k for kk in keys])
        coef, icpt = art["coefs"][k]
        preds[idx] = linear_design(c[idx], drop=set(k), spec=art["spec"]) @ coef + icpt
    return np.clip(preds, *art["clip"])
