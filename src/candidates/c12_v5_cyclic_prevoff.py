"""c12: c11 + offset +0.6 on rows whose previous_usage is missing (shrunk from the +1.02 (se 0.35) OOF bias on
naturally-missing train rows). Diagnostic for missing-not-at-random previous_usage in test."""
from candidates._c11 import build, PREDICT_CODE
def fit_artifact(train): return build(train, cyclic=True, prev_missing_offset=0.6)
