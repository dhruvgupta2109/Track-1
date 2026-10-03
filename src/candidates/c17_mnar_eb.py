"""c17: c11 + missing-not-at-random correction estimated on TRAIN ONLY (no leaderboard-tuned numbers).

When feature c is missing, the true value is on average kappa_c conditional SDs above its imputation (self-masking;
NeuMiss / pattern-mixture delta adjustment). kappa_c is fitted on naturally-missing train rows (src/exp43_mnar_trainfit.py,
OOF energy residuals) and shrunk by empirical Bayes with a skeptical zero-centred prior (mode "zero_ml"):
prev 0.30, occupancy 0.10, temperature -0.11, humidity 0.03. Energy shift per missing feature = kappa_c x model slope
(finite difference at the imputed value) x conditional SD of c for the building (test-domain imputation errors,
features only). Inference is wrapped so the notebook can never fail: any exception -> plain linear model,
non-finite -> train mean.
"""
import json, textwrap
import joblib, numpy as np, pandas as pd
from common import ROOT
from candidates._c11 import PREDICT_CODE as _C11
from features import NUMERIC

STEP = {"occupancy": 5.0, "previous_usage": 1.0, "temperature": 0.5, "humidity": 1.0}


def fit_artifact(train):
    art = dict(joblib.load(ROOT / "package" / "c11_v5_cyclic" / "model.pkl"))
    art["prev_missing_offset"] = 0.0
    kappa = json.load(open(ROOT / "experiments" / "mnar_trainfit.json"))["kappa"]["zero_ml"]
    err = pd.read_csv(ROOT / "experiments" / "imputation_oof_errors.csv")
    sd = err[err.domain == 1].groupby("building_id")[NUMERIC].agg(lambda s: np.sqrt(np.nanmean(s ** 2)))
    art["mnar_kappa"] = {c: float(kappa[c]) for c in NUMERIC}
    art["mnar_sd"] = {c: {b: float(v) for b, v in sd[c].items()} for c in NUMERIC}
    art["mnar_sd_default"] = {c: float(sd[c].mean()) for c in NUMERIC}
    art["mnar_step"] = dict(STEP)
    art["fallback_mean"] = float(train["energy_usage"].mean())
    return art


_MNAR = '''
# 4) missing-not-at-random correction (kappa estimated on naturally-missing TRAIN rows, empirical-Bayes shrunk):
#    when feature c is missing, shift energy by kappa_c x slope_c (at the imputed value) x conditional SD_c(building)
_bid = clean_input(test)["building_id"].to_numpy()
for _col, _k in artifact["mnar_kappa"].items():
    _m = F[_col].isna().to_numpy()
    if _k == 0 or not _m.any():
        continue
    _h = artifact["mnar_step"][_col]
    _up = X.loc[_m].copy(); _dn = X.loc[_m].copy()
    _up[_col] = _up[_col] + _h; _dn[_col] = _dn[_col] - _h
    _slope = (predict_linear(artifact["linear"], _up) - predict_linear(artifact["linear"], _dn)) / (2 * _h)
    _sd = np.array([artifact["mnar_sd"][_col].get(b, artifact["mnar_sd_default"][_col]) for b in _bid[_m]])
    preds[_m] = preds[_m] + np.nan_to_num(_k * _slope * _sd)
'''

PREDICT_CODE = (
    "try:\n" + textwrap.indent(_C11 + "\npreds = np.asarray(preds, dtype=float).copy()\n" + _MNAR, "    ") +
    "\nexcept Exception as _e:  # never fail on unexpected private-test input: fall back to the plain linear model\n"
    "    print('WARNING: full pipeline failed, linear fallback:', repr(_e))\n"
    "    preds = predict_linear(artifact['linear'], test)\n"
    "preds = np.asarray(preds, dtype=float).ravel()\n"
    "preds = np.where(np.isfinite(preds), preds, artifact['fallback_mean'])"
)
