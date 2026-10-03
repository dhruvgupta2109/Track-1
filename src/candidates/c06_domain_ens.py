"""c06: c05 with an averaged trio of domain-aware imputers (default, deep, slow). Held-out test-row
imputation energy-MSE 24.06 -> 23.65; train-like rows 13.68 -> 13.45."""
from candidates._domain_ens import build, PREDICT_CODE
def fit_artifact(train): return build(train, test_weight=1.0)
