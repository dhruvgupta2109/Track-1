"""Paired row uncertainty for the two new target-free imputer comparisons.

Energy-error proxy is a sum of per-feature MSEs with different observed counts;
per-row contributions below rescale to preserve that exact estimand. Bootstrap
rows together across features, rather than treating four errors as independent.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT


def summarize(path):
    E = pd.read_csv(path)
    output = {}
    for pool in ("test", "train", "natmiss"):
        S = E[E.pool == ("test" if pool == "natmiss" else pool)]
        if pool == "natmiss":
            S = S[S.naturally_incomplete]
        table = S.pivot(index=["fold", "row", "feature"], columns="variant", values="energy_error")
        baseline = table.c11_baseline ** 2
        counts = table.groupby(level="feature").size()
        nrows = len(table.index.droplevel("feature").unique())
        scale = nrows / table.index.get_level_values("feature").map(counts).to_numpy()
        for variant in table.columns:
            if variant == "c11_baseline":
                continue
            delta = (table[variant] ** 2 - baseline) * scale
            dr = delta.groupby(level=["fold", "row"]).sum()
            rng = np.random.default_rng(20261003)
            values = dr.to_numpy()
            boot = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(2000)])
            output[f"{pool}:{variant}"] = {
                "proxy_mse_delta": float(dr.mean()),
                "row_paired_se": float(dr.std(ddof=1) / np.sqrt(len(dr))),
                "row_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
                "fold_contribution_means": dr.groupby(level="fold").mean().to_dict(),
            }
    return output


if __name__ == "__main__":
    names = ["extrapolating_imputers_5fold_trio_errors.csv.gz", "energy_weighted_imputers_errors.csv.gz"]
    result = {name: summarize(ROOT / "experiments" / name) for name in names}
    (ROOT / "experiments/imputer_trial_uncertainty.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, results in result.items():
        for key, r in results.items():
            if key.startswith("test:") or key.startswith("natmiss:blend"):
                print(name, key, json.dumps(r))
