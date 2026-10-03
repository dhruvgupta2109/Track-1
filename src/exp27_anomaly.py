import sys; sys.path.insert(0, "src")
import numpy as np
from common import load, cross_validate, test_mask_patterns
from anomaly import AnomalyLinear
from specs import V4
tr, te = load(); P = test_mask_patterns(te); P = P[P.any(1)]
m = cross_validate("a01_anomaly_split", lambda: AnomalyLinear(V4, te, P, mode="split"), tr, te, repeats=2)
m = cross_validate("a02_anomaly_single", lambda: AnomalyLinear(V4, te, P, mode="single"), tr, te, repeats=2)
