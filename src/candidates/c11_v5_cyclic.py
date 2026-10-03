"""c11: c09 imputers with mask-order bug fixed + cyclic hour/month features, and the V5 energy model
(residential day/night steps, hour-deviation 0.1; complete-row CV -0.0073 vs V4, 4/4 repeats).
Proxy: test 20.81 / NATMISS 26.86 (c09 20.87 / 27.63)."""
from candidates._c11 import build, PREDICT_CODE
def fit_artifact(train): return build(train, cyclic=True, prev_missing_offset=0.0)
