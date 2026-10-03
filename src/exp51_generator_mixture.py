"""Paired test/train feature holdouts for an explicit generator hypothesis."""
import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from common import ROOT, load, test_mask_patterns, apply_masks
from features import NUMERIC
from generative_imputer import GeneratorImputer
from proxy import SL


def main(args):
    threadpool_limits(4)
    tr, te = load(); XTR = tr.drop(columns="energy_usage"); XTE = te[XTR.columns]
    kt = list(KFold(5, shuffle=True, random_state=1).split(XTR))
    ke = list(KFold(5, shuffle=True, random_state=0).split(XTE)); P = test_mask_patterns(te)
    saved = []
    for k, ((ta, tb), (ea, eb)) in enumerate(zip(kt[:args.folds], ke[:args.folds])):
        rows = pd.concat([XTR.iloc[ta], XTE.iloc[ea]], ignore_index=True)
        dom = np.r_[np.zeros(len(ta)), np.ones(len(ea))]
        for nc in [1, 8, 16]:
            t0 = time.time(); model = GeneratorImputer(nc).fit(rows, dom)
            for col in NUMERIC:
                for pool, hold, ids, domain in [("test", XTE.iloc[eb].reset_index(drop=True), eb, 1),
                                               ("train", XTR.iloc[tb].reset_index(drop=True), tb, 0)]:
                    valid = hold[col].notna().to_numpy()
                    H = apply_masks(hold[valid], P, seed=100+k); H[col] = np.nan
                    p = model.predict(H, domain=domain)[:, NUMERIC.index(col)]
                    true = hold.loc[valid, col].to_numpy(); slope = H.building_id.map(SL[col]).to_numpy()
                    nat = hold.loc[valid, [c for c in NUMERIC if c != col]].isna().any(axis=1).to_numpy()
                    for i, e, s, natural in zip(np.asarray(ids)[valid], p-true, slope, nat):
                        saved.append(dict(fold=k, pool=pool, feature=col, row=int(i), variant=f"gmm{nc}",
                                          error=float(e), energy_error=float(e*s), naturally_incomplete=bool(natural)))
            print(f"fold {k+1} components {nc}: {time.time()-t0:.1f}s", flush=True)
    E = pd.DataFrame(saved); E.to_csv(ROOT/f"experiments/generator_mixture_{args.folds}fold_errors.csv.gz", index=False)
    W = pd.read_csv(ROOT/"experiments/energy_weighted_imputers_errors.csv.gz")
    R = W[((W.feature == "occupancy") & (W.variant == "blend_weighted")) |
          ((W.feature != "occupancy") & (W.variant == "c11_baseline"))]
    R = R[R.fold < args.folds].assign(variant="c19")
    idx = ["fold", "pool", "feature", "row"]
    blends = []
    for v in E.variant.unique():
        A = E[E.variant == v].set_index(idx).reindex(pd.MultiIndex.from_frame(R[idx]))
        Z = R.copy().assign(variant=f"blend_{v}")
        Z["error"] = .5*(R.error.to_numpy()+A.error.to_numpy())
        Z["energy_error"] = .5*(R.energy_error.to_numpy()+A.energy_error.to_numpy()); blends.append(Z)
    E = pd.concat([E, R, *blends], ignore_index=True); result = {}
    for pool, sel in [("test", E.pool == "test"), ("natmiss", (E.pool == "test") & E.naturally_incomplete),
                      ("train", E.pool == "train")]:
        T = E[sel].assign(mse=lambda d: d.energy_error**2).groupby(["variant", "feature"]).mse.mean().unstack()
        T["total"] = T.sum(axis=1); result[pool] = T.to_dict(orient="index")
        print(pool, T.round(4).to_string(), flush=True)
    (ROOT/f"experiments/generator_mixture_{args.folds}fold.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--folds", type=int, default=1)
    main(parser.parse_args())
