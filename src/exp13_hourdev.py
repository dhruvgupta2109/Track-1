import sys, copy; sys.path.insert(0, "src")
from common import load, cross_validate
from linmodels import PatternLinear
from specs import V3
tr, te = load()
for sc in [0.1, 0.2, 0.35, 0.6, 1.0]:
    s = copy.deepcopy(V3); s["hour_dev_scale"] = sc
    cross_validate(f"l20_hourdev_s{sc}", lambda: PatternLinear(3.0, s), tr, te)
