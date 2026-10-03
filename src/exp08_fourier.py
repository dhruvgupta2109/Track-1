import sys, copy; sys.path.insert(0, "src")
from common import load, cross_validate
from linmodels import PatternLinear
from features import DEFAULT_SPEC
tr, te = load()
for K in [3, 4, 6, 8]:
    s = copy.deepcopy(DEFAULT_SPEC); s["hour_fourier"] = K
    cross_validate(f"l12_fourier_K{K}", lambda: PatternLinear(3.0, s), tr, te)
s = copy.deepcopy(DEFAULT_SPEC); s["hour_fourier"] = 6; s["hours"] = "building"
cross_validate("l13_fourier6_building", lambda: PatternLinear(3.0, s), tr, te)
