"""c14: c11 + flat +0.85 energy offset when previous_usage is missing (precision-weighted combination of the
train OOF estimate 1.02 +- 0.35 and the c12 leaderboard-implied ~0.67 +- 0.35)."""
from candidates._mnar import build, PREDICT_CODE
from features import BUILDINGS
def fit_artifact(train): return build({"previous_usage": {b: 0.85 for b in BUILDINGS}})
