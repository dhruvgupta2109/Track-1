"""OOF evaluation toolkit: segment tables + paired candidate-vs-reference comparison.

Masked OOF rows are reconstructed exactly: same folds / seeds as common.cross_validate.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, NUM, load, folds, test_mask_patterns  # noqa: E402

TR, TE = load()
Y = TR["energy_usage"].to_numpy()
W = np.load(ROOT / "experiments" / "shift_w.npy")
Z = pd.read_csv(ROOT / "experiments" / "train_feature_z.csv")  # target-free spike z-scores
SPIKE_Z = Z[["previous_usage", "occupancy", "temperature"]].abs().max(axis=1).fillna(0).to_numpy()
SPIKE = SPIKE_Z > 3


def applied_masks(repeats, n_splits=5):
    """Bool (repeats, n, 4): the test-like masks applied to each validation row in cross_validate."""
    pats = test_mask_patterns(TE)
    out = np.zeros((repeats, len(TR), 4), bool)
    for r, k, a, b in folds(len(TR), n_splits, repeats):
        rng = np.random.default_rng(7 + 100 * r + k)
        out[r, b] = pats[rng.integers(0, len(pats), len(b))]
    return out | TR[NUM].isna().to_numpy()[None]


def load_oof(name):
    d = np.load(ROOT / "experiments" / "oof" / f"{name}.npz")
    return d["oof"], d["oofm"]


def _rm(e, w=None):
    return float(np.sqrt(np.average(e ** 2, weights=w)))


def summary(name):
    oof, oofm = load_oof(name)
    R = oof.shape[0]
    res = {}
    for lab, o in [("clean", oof), ("masked", oofm)]:
        res[lab] = np.mean([_rm(Y - oi) for oi in o]) ** 1
        res[lab + "_shiftW"] = np.mean([_rm(Y - oi, W) for oi in o])
        res[lab + "_spike"] = np.mean([_rm((Y - oi)[SPIKE]) for oi in o])
        res[lab + "_nonspike"] = np.mean([_rm((Y - oi)[~SPIKE]) for oi in o])
    return res


def paired(ref, cand, n_boot=2000, seed=0):
    """Per-repeat RMSE deltas (cand - ref; negative = cand better) + row-bootstrap CI on masked MSE."""
    a, am = load_oof(ref); b, bm = load_oof(cand)
    R = min(a.shape[0], b.shape[0])
    rows = {}
    for lab, A, B, w in [("clean", a, b, None), ("masked", am, bm, None), ("shiftW", am, bm, W),
                         ("spike_masked", am, bm, SPIKE.astype(float)), ("nonspike_masked", am, bm, (~SPIKE).astype(float))]:
        d = np.array([_rm(Y - B[r], w) - _rm(Y - A[r], w) for r in range(R)])
        rows[lab] = dict(ref=np.mean([_rm(Y - A[r], w) for r in range(R)]), cand=np.mean([_rm(Y - B[r], w) for r in range(R)]),
                         mean_delta=d.mean(), median_delta=float(np.median(d)), std_delta=d.std(), wins=f"{int((d < 0).sum())}/{R}")
    e = ((Y - bm[:R]) ** 2 - (Y - am[:R]) ** 2).mean(0)
    rng = np.random.default_rng(seed)
    bs = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(n_boot)]
    df = pd.DataFrame(rows).T
    ci = np.quantile(bs, [.025, .975])
    return df, dict(masked_mse_delta=float(e.mean()), ci95=(float(ci[0]), float(ci[1])))


def segment_frame(name, use_masked=True):
    """Long frame: one row per (repeat, train row) with residual and segment keys."""
    oof, oofm = load_oof(name)
    O = oofm if use_masked else oof
    R = O.shape[0]
    M = applied_masks(R) if use_masked else np.repeat(TR[NUM].isna().to_numpy()[None], R, 0)
    parts = []
    for r in range(R):
        d = TR.copy()
        d["pred"] = O[r]; d["resid"] = Y - O[r]; d["rep"] = r
        for j, c in enumerate(NUM):
            d["miss_" + c] = M[r][:, j]
        d["n_missing"] = M[r].sum(1)
        d["pattern"] = ["".join("1" if x else "0" for x in row) for row in M[r]]
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["weekend"] = d.day_of_week.isin(["Saturday", "Sunday"])
    zt = np.tile(SPIKE_Z, R)
    d["spike"] = np.where(zt > 5, "extreme", np.where(zt > 3, "mild", "none"))
    for c in ["previous_usage", "occupancy", "temperature"]:
        d["z_" + c] = np.tile(Z[c].to_numpy(), R)
    return d


def seg(d, by):
    g = d.groupby(by, observed=True)
    out = g.agg(n=("resid", "size"), rmse=("resid", lambda e: np.sqrt(np.mean(e ** 2))),
                mae=("resid", lambda e: np.mean(np.abs(e))), bias=("resid", "mean"), sd=("resid", "std"),
                y=("energy_usage", "mean"), pred=("pred", "mean"))
    out["n"] = (out["n"] / d["rep"].nunique()).round().astype(int)
    out["bias_se"] = out["sd"] / np.sqrt(out["n"])
    return out.round(3)
