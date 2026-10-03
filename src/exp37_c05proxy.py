import sys; sys.path.insert(0, "src")
from proxy import evaluate, lgb_member, IMP_PARAMS
evaluate([lgb_member(IMP_PARAMS, 11, 1)], 1.0, label="c05 (LB 3.25549)")
