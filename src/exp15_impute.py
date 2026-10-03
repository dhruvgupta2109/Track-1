import sys; sys.path.insert(0, "src")
from common import load, cross_validate
from imputed import ImputeLinear
from linmodels import PatternLinear
from specs import V3, V4
tr, te = load()
TE = te
cross_validate("i02_V3_impute_occ_prev_train+test", lambda: ImputeLinear(V3, extra_rows=TE), tr, te)
cross_validate("i03_V4_impute_occ_prev_train+test", lambda: ImputeLinear(V4, extra_rows=TE), tr, te)
cross_validate("i04_V4_impute_occ_only", lambda: ImputeLinear(V4, extra_rows=TE, cols=["occupancy"]), tr, te)
cross_validate("i05_V4_impute_blend0.5", lambda: ImputeLinear(V4, extra_rows=TE, blend=0.5), tr, te)
