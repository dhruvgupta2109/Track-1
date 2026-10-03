import sys, copy; sys.path.insert(0, "src")
from common import load, cross_validate
from linmodels import PatternLinear
from features import DEFAULT_SPEC
tr, te = load()
def S(**kw):
    s = copy.deepcopy(DEFAULT_SPEC); s.update(kw); return s
base = dict(hour_fourier=3, weekend_hours="shared")
for nm, ms in {"fourier6_type_wk": dict(hour_fourier=6, weekend_hours="type"),
               "onehot_type_wktype": dict(hour_fourier=0, hours="type", weekend_hours="type"),
               "onehot_bldg_wktype": dict(hour_fourier=0, hours="building", weekend_hours="type"),
               "fourier6_bldg_wkbldg": dict(hour_fourier=6, hours="building", weekend_hours="building")}.items():
    cross_validate(f"l19_rich_{nm}", lambda: PatternLinear(3.0, S(**base, missing_spec=ms)), tr, te)
