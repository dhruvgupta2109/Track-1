import sys, copy; sys.path.insert(0, "src")
from common import load, cross_validate
from linmodels import PatternLinear
from features import DEFAULT_SPEC
tr, te = load()
def S(**kw):
    s = copy.deepcopy(DEFAULT_SPEC); sl = kw.pop("slopes", {}); s["slopes"].update(sl); s.update(kw); return s
ALLB = S(hours="building", slopes={c: "building" for c in DEFAULT_SPEC["slopes"]})
cross_validate("l02_v2_allbuilding(=l01)", lambda: PatternLinear(3.0, ALLB), tr, te)
cross_validate("l03_v2_default", lambda: PatternLinear(3.0), tr, te)
cross_validate("l04_v2_default_a1", lambda: PatternLinear(1.0), tr, te)
cross_validate("l05_v2_occ_temp_type", lambda: PatternLinear(3.0, S(slopes={"occupancy": "type", "temperature": "type"})), tr, te)
cross_validate("l06_v2_weekend_type", lambda: PatternLinear(3.0, S(weekend="type")), tr, te)
cross_validate("l07_v2_no_t2", lambda: PatternLinear(3.0, S(temp2=False)), tr, te)
