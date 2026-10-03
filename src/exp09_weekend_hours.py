import sys, copy; sys.path.insert(0, "src")
from common import load, cross_validate
from linmodels import PatternLinear
from features import DEFAULT_SPEC
tr, te = load()
def S(**kw):
    s = copy.deepcopy(DEFAULT_SPEC); s.update(kw); return s
cross_validate("l14_f3_wkhours_shared", lambda: PatternLinear(3.0, S(hour_fourier=3, weekend_hours="shared")), tr, te)
cross_validate("l15_f3_wkhours_type", lambda: PatternLinear(3.0, S(hour_fourier=3, weekend_hours="type")), tr, te)
cross_validate("l16_f3_occ_type", lambda: PatternLinear(3.0, S(hour_fourier=3, slopes={**DEFAULT_SPEC["slopes"], "occupancy": "type"})), tr, te)
cross_validate("l17_f3_a10", lambda: PatternLinear(10.0, S(hour_fourier=3)), tr, te)
cross_validate("l18_f3_a1", lambda: PatternLinear(1.0, S(hour_fourier=3)), tr, te)
