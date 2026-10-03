"""Test parsimonious synthetic-generator shapes and coefficient quantization.

Hypotheses, not recovered source code. Three repeated five-fold complete-row
tests compare every formula to refitted V5 on exactly the same held-out rows.
All parameter rounding is performed within each training fold.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from common import ROOT, load
from features import BUILDINGS, TYPES, BUILDING_TYPE, clean_input, linear_design, _soft_building
from specs import V5


def compact(c):
    B = _soft_building(c); T = B@np.array([[BUILDING_TYPE[b] == t for t in TYPES] for b in BUILDINGS], float)
    h = c.hour.to_numpy(); we = (c.dow.to_numpy() >= 5).astype(float)[:, None]; m = c.month.to_numpy()
    H = np.stack([np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)], axis=1)
    occ = B*c.occupancy.to_numpy()[:, None]; temp = B*(c.temperature.to_numpy()[:, None]-28)
    # Numeric slopes are placed first so quantization has an explicit layout.
    numeric = np.c_[c.previous_usage.to_numpy(), c.humidity.to_numpy()-78, occ, temp]
    parts = [numeric, B, B*we, (T[:, :, None]*H[:, None, :]).reshape(len(c), -1), H*we]
    parts.extend([((h >= 8) & (h < 21))[:, None]*T[:, TYPES.index("Residential")][:, None],
                  np.stack([np.sin(4*np.pi*h/24), np.cos(4*np.pi*h/24)], 1)*T[:, TYPES.index("Sports")][:, None],
                  np.exp(-.5*((h-20.5)/2)**2)[:, None]*T[:, TYPES.index("Library")][:, None],
                  np.stack([np.sin(6*np.pi*h/24), np.cos(6*np.pi*h/24)], 1)*T[:, TYPES.index("LectureHall")][:, None]*.5,
                  np.stack([np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)], 1),
                  ((c.temperature.to_numpy()-28)**2)[:, None]])
    return np.concatenate(parts, 1)


def rounded_fit(D, y):
    model = Ridge(alpha=3).fit(D, y)
    # Coarse simple constants are plausible in hand-written synthetic generators.
    steps = np.r_[.01, .01, np.full(12, .005), np.full(12, .1)]
    coef = np.round(model.coef_[:26]/steps)*steps
    rest = Ridge(alpha=3).fit(D[:, 26:], y-D[:, :26]@coef)
    return lambda Z: Z[:, :26]@coef + rest.predict(Z[:, 26:])


def main():
    threadpool_limits(4); tr, _ = load(); tr = tr.dropna().reset_index(drop=True)
    c = clean_input(tr); y = tr.energy_usage.to_numpy(); old = linear_design(c, spec=V5); small = compact(c)
    p = {v: np.zeros((3, len(tr))) for v in ["v5", "compact", "compact_rounded"]}
    for repeat in range(3):
        for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=repeat).split(tr)):
            p["v5"][repeat, b] = Ridge(alpha=3).fit(old[a], y[a]).predict(old[b])
            p["compact"][repeat, b] = Ridge(alpha=3).fit(small[a], y[a]).predict(small[b])
            p["compact_rounded"][repeat, b] = rounded_fit(small[a], y[a])(small[b])
        print(repeat, {v: float(np.sqrt(np.mean((y-p[v][repeat])**2))) for v in p}, flush=True)
    result = {"n": len(tr), "v5_columns": old.shape[1], "compact_columns": small.shape[1], "metrics": {}}
    for v, preds in p.items():
        dr = np.mean((y-preds)**2-(y-p["v5"])**2, axis=0)
        result["metrics"][v] = dict(rmse=float(np.sqrt(np.mean((y-preds)**2))), delta_mse=float(dr.mean()),
                                   row_paired_se=float(dr.std(ddof=1)/np.sqrt(len(dr))))
    np.savez_compressed(ROOT/"experiments/compact_formula_oof.npz", **p, y=y)
    (ROOT/"experiments/compact_formula.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
