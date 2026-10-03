import sys; sys.path.insert(0, "src")
from common import load, cross_validate
from linmodels import PatternLinear
from specs import V2, V3
tr, te = load()
cross_validate("final_cmp_V2_5x4", lambda: PatternLinear(3.0, V2), tr, te, repeats=4)
cross_validate("final_cmp_V3_5x4", lambda: PatternLinear(3.0, V3), tr, te, repeats=4)
