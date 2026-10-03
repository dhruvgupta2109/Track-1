"""c17 statistical model with input recovery and isolated imputer fallback.

Normal supplied-test predictions are unchanged. Known building IDs recover the
type when it is absent or malformed; a failed imputer leaves just that feature
missing for the matching linear sub-model. MNAR shifts apply only to successfully
filled features, because their estimated slopes assume the full linear model.
"""
import hashlib
from pathlib import Path

import joblib

from common import ROOT
from devimpute import CYC_CODE, DEV_CODE


def fit_artifact(train):
    """Reuse the frozen, train-estimated c17 parameters; this is not a retrain."""
    source = ROOT / "package/c17_mnar_eb/model.pkl"
    art = dict(joblib.load(source))
    art["source_candidate"] = "c17_mnar_eb"
    art["source_artifact_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return art


PREDICT_CODE = DEV_CODE + CYC_CODE + '''
import lightgbm as lgb

# Known building IDs determine their type. All inputs remain row-local.
_raw = test.copy()
_clean = clean_input(_raw)
_type_names = {t.lower(): t for t in TYPES}
_types = _clean["building_type"].str.lower().map(_type_names)
_known = _clean["building_id"].map(BUILDING_TYPE)
_raw["building_type"] = _known.fillna(_types).to_numpy()

F = make_features(_raw)
F["dom"] = artifact["domain"]
if artifact["cyclic"]:
    F = add_cyclic(F)
X = _raw.copy()
_filled = {}
for col in artifact["impute_cols"]:
    _miss = F[col].isna().to_numpy()
    _filled[col] = np.zeros(len(F), dtype=bool)
    X[col] = F[col].to_numpy()
    if not _miss.any():
        continue
    try:
        _xm = add_dev(F, artifact["dev_tables"], artifact["dev_levels"], col).drop(columns=[col])[_miss]
        _values = np.mean([lgb.Booster(model_str=s).predict(_xm)
                           for s in artifact["imputers"][col]], axis=0)
        # The same physical bounds as clean_input; unusable fills stay missing.
        _lo, _hi = VALID[col]
        _ok = np.isfinite(_values) & (_values >= _lo) & (_values <= _hi)
        _positions = np.flatnonzero(_miss)
        X.iloc[_positions[_ok], X.columns.get_loc(col)] = _values[_ok]
        _filled[col][_positions[_ok]] = True
    except Exception as _error:
        print('WARNING: imputer fallback for', col, repr(_error))

try:
    preds = np.asarray(predict_linear(artifact["linear"], X), dtype=float).copy()
    _bid = clean_input(_raw)["building_id"].to_numpy()
    _complete = clean_input(X)[NUMERIC].notna().all(axis=1).to_numpy()
    for _col, _k in artifact["mnar_kappa"].items():
        _m = _filled[_col] & _complete
        if _k == 0 or not _m.any():
            continue
        _h = artifact["mnar_step"][_col]
        _up = X.loc[_m].copy(); _dn = X.loc[_m].copy()
        _up[_col] = _up[_col] + _h; _dn[_col] = _dn[_col] - _h
        _slope = (predict_linear(artifact["linear"], _up) - predict_linear(artifact["linear"], _dn)) / (2 * _h)
        _sd = np.array([artifact["mnar_sd"][_col].get(b, artifact["mnar_sd_default"][_col]) for b in _bid[_m]])
        preds[_m] = preds[_m] + np.nan_to_num(_k * _slope * _sd)
except Exception as _error:
    print('WARNING: energy model fallback:', repr(_error))
    try:
        preds = predict_linear(artifact["linear"], _raw)
    except Exception as _fallback_error:
        print('WARNING: mean fallback:', repr(_fallback_error))
        preds = np.full(len(_raw), artifact["fallback_mean"], dtype=float)

preds = np.asarray(preds, dtype=float).ravel()
preds = np.where(np.isfinite(preds), preds, artifact["fallback_mean"])
'''
