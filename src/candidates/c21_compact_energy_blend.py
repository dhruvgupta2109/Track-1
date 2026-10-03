"""Platform-compatible c19 with a fixed 50/50 parsimonious energy blend.

Compact formula uses simple daily waves, a library evening bump and residential
day/night steps. It is a generator hypothesis selected on train CV, not known
source code. Project the fit into the existing V5 design so inference and all
missing-pattern failure paths retain their established implementation.
"""
import hashlib
import itertools
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from common import ROOT
from candidates.c18_private_safe import PREDICT_CODE
from exp52_compact_formula import compact
from features import BUILDINGS, BUILDING_TYPE, clean_input, linear_design


def fit_artifact(train):
    source = ROOT/"package/c19_occ_energy_blend/model.pkl"; art = dict(joblib.load(source))
    lin = dict(art["linear"]); lin["coefs"] = dict(lin["coefs"])
    data = train.dropna().reset_index(drop=True); clean = clean_input(data)
    model = Ridge(alpha=3).fit(compact(clean), data.energy_usage.to_numpy())
    basis = [dict(building_id=b, building_type=BUILDING_TYPE[b], hour=h, day_of_week=d,
                  month=6, occupancy=50., previous_usage=50., temperature=28., humidity=78.)
             for b,h,d in itertools.product(BUILDINGS, range(24), ["Tuesday", "Saturday"])]
    for b in BUILDINGS:
        r = dict(building_id=b, building_type=BUILDING_TYPE[b], hour=12, day_of_week="Tuesday",
                 month=6, occupancy=50., previous_usage=50., temperature=28., humidity=78.)
        for col in ["occupancy", "previous_usage", "temperature", "humidity"]:
            for delta in [-2., 2.]:
                basis.append({**r, col:r[col]+delta})
    basis.extend([{**basis[0], "month":m} for m in range(1,13)])
    C = clean_input(pd.DataFrame(basis)); old = linear_design(C,spec=lin["spec"])
    coef = np.linalg.lstsq(np.c_[old,np.ones(len(old))], model.predict(compact(C)), rcond=1e-11)[0]
    D = linear_design(clean,spec=lin["spec"])
    error = float(np.max(np.abs(D@coef[:-1]+coef[-1]-model.predict(compact(clean)))))
    assert error < 1e-8, error
    original, intercept = lin["coefs"][()]
    lin["coefs"][()] = (.5*(original+coef[:-1]), .5*(intercept+coef[-1]))
    art["linear"] = lin; art["source_candidate"] = "c19_occ_energy_blend"
    art["source_artifact_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    art["experimental_change"] = dict(component="energy", compact_blend=.5, compact_columns=77,
                                       projection_max_train_error=error, rounding=False)
    return art
