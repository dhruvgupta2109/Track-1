"""Shared data loading, row-local features, and validation harness for Track 1.

All features are strictly row-local: the platform test file has no id column and is shuffled,
so nothing may depend on other rows or on the id.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
TARGET = "energy_usage"
NUM = ["temperature", "humidity", "occupancy", "previous_usage"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
BUILDINGS = ["ADM_A", "BUS_A", "BUS_B", "ENG_A", "ENG_B", "LEC_A", "LIB_A",
             "RES_A", "RES_B", "SCI_A", "SCI_B", "SPT_A"]
TYPES = ["Administration", "Business", "Engineering", "LectureHall", "Library",
         "Residential", "Science", "Sports"]
OOF_DIR = ROOT / "experiments" / "oof"
LOG = ROOT / "experiments" / "log.jsonl"


def load():
    return pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def metrics(y, p):
    return dict(rmse=rmse(y, p), mae=float(mean_absolute_error(y, p)), r2=float(r2_score(y, p)))


def test_mask_patterns(test):
    """Empirical joint missingness patterns of the test file (bool array n x 4)."""
    return test[NUM].isna().to_numpy()


def apply_masks(df, patterns, seed):
    """Overwrite NUM columns with NaN using patterns sampled from the test distribution."""
    rng = np.random.default_rng(seed)
    m = patterns[rng.integers(0, len(patterns), len(df))]
    out = df.copy()
    vals = out[NUM].to_numpy(dtype=float)
    vals[m] = np.nan
    out[NUM] = vals
    return out


def folds(n, n_splits=5, repeats=2, seed=0):
    for r in range(repeats):
        kf = KFold(n_splits, shuffle=True, random_state=seed + r)
        for k, (a, b) in enumerate(kf.split(np.arange(n))):
            yield r, k, a, b


def cross_validate(name, make_model, train, test, n_splits=5, repeats=2, augment=None,
                   notes="", save=True, verbose=True):
    """make_model() -> object with fit(X_df, y) / predict(X_df) on RAW columns.

    Reports: clean OOF metrics, and 'masked' metrics where validation rows receive
    test-like missingness masks (the test file has ~3x the train missing rate).
    augment: optional fn(train_fold_df, seed) -> augmented train df (applied inside fold only).
    """
    t0 = time.time()
    pats = test_mask_patterns(test)
    y = train[TARGET].to_numpy()
    n = len(train)
    oof = np.zeros((repeats, n)); oofm = np.zeros((repeats, n))
    for r, k, a, b in folds(n, n_splits, repeats):
        tr = train.iloc[a]
        if augment is not None:
            tr = augment(tr, seed=1000 * r + k)
        m = make_model()
        m.fit(tr.drop(columns=[TARGET]), tr[TARGET].to_numpy())
        va = train.iloc[b].drop(columns=[TARGET])
        oof[r, b] = m.predict(va)
        oofm[r, b] = m.predict(apply_masks(va, pats, seed=7 + 100 * r + k))
    res = dict(name=name, notes=notes, n_splits=n_splits, repeats=repeats,
               clean=metrics(np.tile(y, repeats), oof.ravel()),
               masked=metrics(np.tile(y, repeats), oofm.ravel()),
               clean_rep_rmse=[rmse(y, o) for o in oof],
               shift=_shift(y, oof, oofm),
               seconds=round(time.time() - t0, 1))
    if save:
        OOF_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OOF_DIR / f"{name}.npz", oof=oof, oofm=oofm)
        with open(LOG, "a") as f:
            f.write(json.dumps(res) + "\n")
    if verbose:
        print(f"{name:40s} clean {res['clean']['rmse']:.4f} (mae {res['clean']['mae']:.3f}, r2 {res['clean']['r2']:.4f}) "
              f"| masked {res['masked']['rmse']:.4f} | shiftW masked {res['shift'].get('masked', float('nan')):.4f} | reps {np.round(res['clean_rep_rmse'], 4)} | {res['seconds']}s")
    return res


def _shift(y, oof, oofm):
    """RMSE with train rows re-weighted so feature-spike frequency matches test (src/shift_weights.py)."""
    p = ROOT / "experiments" / "shift_w.npy"
    if not p.exists():
        return {}
    w = np.load(p)
    f = lambda o: float(np.sqrt(np.mean([np.average((y - oi) ** 2, weights=w) for oi in o])))
    return {"clean": f(oof), "masked": f(oofm)}
