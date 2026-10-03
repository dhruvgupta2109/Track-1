"""c16: c12 (prev +0.6) + occupancy self-masking offset: kappa 0.20 (train single-missing OOF, se 0.20) x per-building
occupancy slope x TEST conditional SD of occupancy."""
import joblib
from common import ROOT
from candidates._mnar import build, PREDICT_CODE, _sd_test, _slopes
from features import BUILDINGS
def fit_artifact(train):
    sd = _sd_test()["occupancy"]; sl = _slopes(joblib.load(ROOT / "package" / "c11_v5_cyclic" / "model.pkl")["linear"])["occupancy"]
    return build({"previous_usage": {b: 0.6 for b in BUILDINGS}, "occupancy": {b: 0.20 * sl[b] * sd[b] for b in BUILDINGS}})
