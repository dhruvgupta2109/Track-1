import sys; sys.path.insert(0, "src")
from common import load, cross_validate, test_mask_patterns
from imputed import ImputeLinear
from specs import V4
tr, te = load(); P = test_mask_patterns(te); P = P[P.any(1)]
ALL = ["occupancy", "previous_usage", "temperature", "humidity"]
cross_validate("c05a_trainonly_imputers_5x4", lambda: ImputeLinear(V4, cols=ALL, extra_rows=None, aug_patterns=P, aug_copies=1), tr, te, repeats=4)
