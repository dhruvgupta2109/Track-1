"""MNAR (self-masking) offsets on top of the frozen c11 artifact. Offsets are per (feature, building) energy shifts
added when that feature is missing: a delta-adjustment / pattern-mixture correction (cf. NeuMiss, Le Morvan 2020)."""
import joblib, numpy as np, pandas as pd
from common import ROOT
from candidates._c11 import PREDICT_CODE as _C11
from features import BUILDINGS

def _sd_test():
    err = pd.read_csv(ROOT / "experiments" / "imputation_oof_errors.csv")
    return err[err.domain == 1].groupby("building_id")[["previous_usage", "occupancy", "temperature", "humidity"]].agg(lambda s: np.sqrt(np.nanmean(s ** 2)))

def _slopes(lin):
    from interpret import effects
    e = effects({"linear": lin})
    return {"occupancy": (e["per_100_occupants"] / 100).to_dict(), "previous_usage": (e["per_10_prev"] / 10).to_dict()}

def build(offsets):
    art = joblib.load(ROOT / "package" / "c11_v5_cyclic" / "model.pkl")
    art = dict(art); art["prev_missing_offset"] = 0.0
    art["missing_offsets"] = {c: {b: float(v) for b, v in d.items()} for c, d in offsets.items()}
    return art

PREDICT_CODE = _C11 + '''
# 4) missing-not-at-random correction: per (feature, building) energy shift when that feature is missing
_bid = clean_input(test)["building_id"].to_numpy()
for _col, _off in artifact.get("missing_offsets", {}).items():
    _m = F[_col].isna().to_numpy()
    _shift = np.array([_off.get(b, np.mean(list(_off.values()))) for b in _bid])
    preds = preds + _shift * _m'''
