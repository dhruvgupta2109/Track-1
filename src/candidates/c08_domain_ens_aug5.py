"""c08: c07 (domain-aware imputer trio, test rows x2) with 5 test-like mask-augmentation copies.
Test-row proxy 22.21 (c07 22.61, LB 3.23149)."""
from candidates._domain_ens import build, PREDICT_CODE
def fit_artifact(train): return build(train, test_weight=2.0, copies=5)
