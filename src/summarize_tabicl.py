"""Fixed blends and paired uncertainty for the saved foundation-model holdout."""
import json
import numpy as np
import pandas as pd
from common import ROOT


def main():
    E = pd.read_csv(ROOT/"experiments/tabicl_5fold_errors.csv.gz")
    W = pd.read_csv(ROOT/"experiments/energy_weighted_imputers_errors.csv.gz")
    R = W[((W.feature == "occupancy") & (W.variant == "blend_weighted")) |
          ((W.feature != "occupancy") & (W.variant == "c11_baseline"))].assign(variant="c19")
    idx = ["fold", "pool", "feature", "row"]
    A = E.set_index(idx).reindex(pd.MultiIndex.from_frame(R[idx]))
    variants = [R, E]
    for name, features in [("blend_all", list(R.feature.unique())),
                           ("blend_occ_prev", ["occupancy", "previous_usage"])]:
        Z = R.copy().assign(variant=name); use = Z.feature.isin(features)
        for c in ["error", "energy_error"]:
            Z.loc[use, c] = .5*(R.loc[use, c].to_numpy()+A.loc[use.to_numpy(), c].to_numpy())
        variants.append(Z)
    E = pd.concat(variants, ignore_index=True); result = {}
    for pool, sel in [("test", E.pool == "test"), ("natmiss", (E.pool == "test") & E.naturally_incomplete),
                      ("train", E.pool == "train")]:
        S = E[sel]
        T = S.assign(mse=lambda d: d.energy_error**2).groupby(["variant", "feature"]).mse.mean().unstack()
        T["total"] = T.sum(axis=1); result[pool] = T.to_dict(orient="index")
        print(pool, T.round(4).to_string())
        table = S.pivot(index=idx, columns="variant", values="energy_error")
        counts = table.groupby(level="feature").size(); nrows = len(table.index.droplevel("feature").unique())
        scale = nrows/table.index.get_level_values("feature").map(counts).to_numpy()
        uncertainty = {}
        for v in ["tabicl", "blend_all", "blend_occ_prev"]:
            dr = ((table[v]**2-table.c19**2)*scale).groupby(level=["fold", "row"]).sum()
            rng = np.random.default_rng(20261003); values = dr.to_numpy()
            boot = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(2000)])
            uncertainty[v] = dict(delta=float(dr.mean()), se=float(dr.std(ddof=1)/np.sqrt(len(dr))),
                                  ci95=np.quantile(boot, [.025, .975]).tolist(),
                                  fold_means=dr.groupby(level="fold").mean().to_dict())
        result[pool+"_uncertainty"] = uncertainty
        print(json.dumps(uncertainty, indent=2))
    (ROOT/"experiments/tabicl_blend_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    E.to_csv(ROOT/"experiments/tabicl_blends_errors.csv.gz", index=False)


if __name__ == "__main__":
    main()
