import sys; sys.path.insert(0, "src")
from common import load, cross_validate, test_mask_patterns
from imputed import ImputeLinear, IMP_PARAMS
from specs import V4
tr, te = load(); P = test_mask_patterns(te); P = P[P.any(1)]
K = dict(extra_rows=te, aug_patterns=P, aug_copies=1)
ALL = ["occupancy", "previous_usage", "temperature", "humidity"]
cross_validate("i10_imp_all4", lambda: ImputeLinear(V4, cols=ALL, **K), tr, te)
cross_validate("i11_imp_small", lambda: ImputeLinear(V4, params=dict(IMP_PARAMS, n_estimators=400, num_leaves=15, min_child_samples=20), **K), tr, te)
cross_validate("i12_imp_big", lambda: ImputeLinear(V4, params=dict(IMP_PARAMS, n_estimators=1500, learning_rate=0.02, num_leaves=63, min_child_samples=5), **K), tr, te)
cross_validate("i13_imp_slow", lambda: ImputeLinear(V4, params=dict(IMP_PARAMS, n_estimators=1500, learning_rate=0.01), **K), tr, te)
