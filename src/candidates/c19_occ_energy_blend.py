"""Experimental c18/c17 pipeline with a fixed energy-weighted occupancy blend.

The occupancy blend improved all three saved feature-holdout slices. Temperature
weighting hurt, so retain that imputer and every other statistical component.
The feature choice follows the experiment-49 results; no public score is known.
"""
import hashlib

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from candidates._c11 import CONFIGS
from candidates.c18_private_safe import PREDICT_CODE
from common import ROOT, NUM, test_mask_patterns
from devimpute import add_cyclic, add_dev
from features import NUMERIC, make_features
from imputed import _mask_aug
from proxy import SL


def fit_artifact(train):
    source = ROOT / "package/c18_private_safe/model.pkl"
    art = dict(joblib.load(source))
    art["imputers"] = {c: list(models) for c, models in art["imputers"].items()}
    test = pd.read_csv(ROOT / "test.csv")
    X = train.drop(columns="energy_usage")
    rows = pd.concat([X, test[X.columns]], ignore_index=True)
    domain = np.r_[np.zeros(len(X)), np.ones(len(test))]
    F0 = add_cyclic(make_features(rows).assign(dom=domain))
    patterns = test_mask_patterns(test)
    patterns = patterns[patterns.any(axis=1)][:, [NUM.index(c) for c in NUMERIC]]
    weights = np.where(domain == 1, 2., 1.)
    slope_weights = rows.building_id.map(SL["occupancy"]).to_numpy() ** 2
    slope_weights /= np.average(slope_weights, weights=weights)
    # Three original + three weighted members => a fixed 50/50 ensemble blend.
    for params, seed in CONFIGS:
        F = _mask_aug(F0, "occupancy", patterns, 5, seed=seed)
        w = np.tile(weights * slope_weights, 6)
        observed = F.occupancy.notna().to_numpy()
        A = add_dev(F, art["dev_tables"], art["dev_levels"], "occupancy").drop(columns="occupancy")
        model = lgb.LGBMRegressor(**params).fit(
            A[observed], F.loc[observed, "occupancy"], sample_weight=w[observed])
        art["imputers"]["occupancy"].append(model.booster_.model_to_string())
        print(f"weighted occupancy member seed {seed}: fitted", flush=True)
    art["source_candidate"] = "c18_private_safe"
    art["source_artifact_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    art["experimental_change"] = {"feature": "occupancy", "weighted_blend": .5,
                                   "energy_slopes_source": "c04_linear_imputed",
                                   "selection": "feature selected from experiment 49; blend fixed at 0.5"}
    return art
