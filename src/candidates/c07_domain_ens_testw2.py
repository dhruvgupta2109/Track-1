"""c07: c06 with test rows weighted x2 in imputer fitting. Held-out test-row imputation energy-MSE 23.57
(c06 23.65); train-like rows 13.54 (c06 13.45). Most test-specific variant."""
from candidates._domain_ens import build, PREDICT_CODE
def fit_artifact(train): return build(train, test_weight=2.0)
