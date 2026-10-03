import sys; sys.path.insert(0, "src")
import numpy as np
from common import load, cross_validate, test_mask_patterns
from imputed import ImputeLinear
from specs import V4
tr, te = load(); P = test_mask_patterns(te); P = P[P.any(1)]
ALL = ["occupancy", "previous_usage", "temperature", "humidity"]
WD = np.load("experiments/density_w.npy"); WS = np.load("experiments/shift_w.npy")
K = dict(cols=ALL, extra_rows=te, aug_patterns=P, aug_copies=1)
run = lambda n, **kw: cross_validate(n, lambda: ImputeLinear(V4, **K, **kw), tr, te, repeats=2)
run("f01_densityW_fit", fit_weights=WD)
run("f02_densityW_sqrt_fit", fit_weights=np.sqrt(WD))
run("f03_spikeW_sqrt_fit", fit_weights=np.sqrt(WS))
run("f06_alpha1", alpha=1.0)
run("f07_alpha10", alpha=10.0)
