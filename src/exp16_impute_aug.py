import sys; sys.path.insert(0, "src")
from common import load, cross_validate, test_mask_patterns
from imputed import ImputeLinear
from specs import V4
tr, te = load(); P = test_mask_patterns(te); P = P[P.any(1)]
cross_validate("i06_V4_imp_fallback", lambda: ImputeLinear(V4, extra_rows=te, both_missing_fallback=True), tr, te)
cross_validate("i07_V4_imp_aug1", lambda: ImputeLinear(V4, extra_rows=te, aug_patterns=P, aug_copies=1), tr, te)
cross_validate("i08_V4_imp_aug2", lambda: ImputeLinear(V4, extra_rows=te, aug_patterns=P, aug_copies=2), tr, te)
cross_validate("i09_V4_imp_aug1_fallback", lambda: ImputeLinear(V4, extra_rows=te, aug_patterns=P, aug_copies=1, both_missing_fallback=True), tr, te)
