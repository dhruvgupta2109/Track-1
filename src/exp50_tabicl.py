"""New foundation-model pilot, held-out rows and frozen c19 comparator.

Research dependencies live in tmp/venv-tabicl; final environment is untouched.
Checkpoint downloads are public weights, never training-file uploads. No test
energy labels are available. Save per-row feature errors for paired scoring.
"""
import argparse
import gc
import json
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from tabicl import TabICLRegressor

from common import ROOT, load, apply_masks, test_mask_patterns
from devimpute import LEVELS, make_tables, add_dev, add_cyclic
from features import make_features, clean_input, linear_design, NUMERIC
from proxy import SL
from specs import V5


def regressor():
    return TabICLRegressor(n_estimators=8, batch_size=1, device="cuda",
                          use_amp=True, use_fa3=False, kv_cache=False,
                          offload_mode="auto", n_jobs=4, random_state=42)


def numerical(F):
    """Fixed category codes retain identifiers without proprietary encoders."""
    return F.assign(**{c: F[c].cat.codes.astype(float).replace(-1, np.nan)
                       for c in F.select_dtypes("category").columns})


def main(args):
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.72)
    tr, te = load(); XTR = tr.drop(columns="energy_usage"); XTE = te[XTR.columns]
    kt = list(KFold(5, shuffle=True, random_state=1).split(XTR))
    ke = list(KFold(5, shuffle=True, random_state=0).split(XTE))
    P = test_mask_patterns(te); saved = []; energy = []
    stem = f"tabicl_{args.folds}fold"
    for k, ((ta, tb), (ea, eb)) in enumerate(zip(kt[:args.folds], ke[:args.folds])):
        rows = pd.concat([XTR.iloc[ta], XTE.iloc[ea]], ignore_index=True)
        dom = np.r_[np.zeros(len(ta)), np.ones(len(ea))]
        F0 = add_cyclic(make_features(rows).assign(dom=dom)); tables = make_tables(F0, LEVELS)
        if args.tasks in ("both", "impute"):
            for col in NUMERIC:
                t0 = time.time(); ok = F0[col].notna().to_numpy()
                A = numerical(add_dev(F0, tables, LEVELS, col).drop(columns=col))
                m = regressor().fit(A[ok], F0.loc[ok, col].to_numpy())
                for pool, hold, ids, domain in [("test", XTE.iloc[eb].reset_index(drop=True), eb, 1),
                                               ("train", XTR.iloc[tb].reset_index(drop=True), tb, 0)]:
                    valid = hold[col].notna().to_numpy()
                    H = apply_masks(hold[valid], P, seed=100 + k); H[col] = np.nan
                    G = add_cyclic(make_features(H).assign(dom=domain))
                    Z = numerical(add_dev(G, tables, LEVELS, col).drop(columns=col))
                    pred = m.predict(Z)
                    natural = hold.loc[valid, [c for c in NUMERIC if c != col]].isna().any(axis=1).to_numpy()
                    slope = H.building_id.map(SL[col]).to_numpy()
                    true = hold.loc[valid, col].to_numpy()
                    for i, e, s, nat in zip(np.asarray(ids)[valid], pred - true, slope, natural):
                        saved.append(dict(fold=k, pool=pool, feature=col, row=int(i), variant="tabicl",
                                          error=float(e), energy_error=float(e*s), naturally_incomplete=bool(nat)))
                pd.DataFrame(saved).to_csv(ROOT / f"experiments/{stem}_errors.csv.gz", index=False)
                print(f"fold {k+1} {col}: {time.time()-t0:.1f}s", flush=True)
                del m; gc.collect(); torch.cuda.empty_cache()
        if args.tasks in ("both", "energy"):
            # Independent complete-row energy test: all fits exclude held-out y.
            a = XTR.iloc[ta].dropna(subset=NUMERIC).index
            b = XTR.iloc[tb].dropna(subset=NUMERIC).index
            F = numerical(add_cyclic(make_features(XTR)))
            D = linear_design(clean_input(XTR.loc[a]), spec=V5)
            Db = linear_design(clean_input(XTR.loc[b]), spec=V5)
            y = tr.energy_usage.to_numpy()
            base = Ridge(alpha=3).fit(D, y[a]); bp = base.predict(Db)
            for task, target in [("direct", y[a]), ("residual", y[a]-base.predict(D))]:
                t0 = time.time(); m = regressor().fit(F.loc[a], target)
                p = m.predict(F.loc[b]); p = p if task == "direct" else bp + p
                energy.append(dict(fold=k, task=task, n=len(b), rmse=float(np.sqrt(np.mean((y[b]-p)**2))),
                                   ridge_rmse=float(np.sqrt(np.mean((y[b]-bp)**2))), seconds=time.time()-t0))
                print(energy[-1], flush=True)
                del m; gc.collect(); torch.cuda.empty_cache()
    result = {"tabicl": version("tabicl"), "torch": torch.__version__, "energy": energy}
    if saved:
        E = pd.DataFrame(saved)
        W = pd.read_csv(ROOT / "experiments/energy_weighted_imputers_errors.csv.gz")
        # c19 is occupancy blend, c11 on the remaining features.
        R = W[((W.feature == "occupancy") & (W.variant == "blend_weighted")) |
              ((W.feature != "occupancy") & (W.variant == "c11_baseline"))]
        R = R[R.fold < args.folds].assign(variant="c19")
        E = pd.concat([E, R], ignore_index=True)
        for pool, sel in [("test", E.pool == "test"), ("natmiss", (E.pool == "test") & E.naturally_incomplete),
                          ("train", E.pool == "train")]:
            T = E[sel].assign(mse=lambda d: d.energy_error**2).groupby(["variant", "feature"]).mse.mean().unstack()
            T["total"] = T.sum(axis=1); result[pool] = T.to_dict(orient="index")
            print(pool, T.round(4).to_string(), flush=True)
    (ROOT / f"experiments/{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--tasks", choices=["both", "impute", "energy"], default="both")
    main(parser.parse_args())
