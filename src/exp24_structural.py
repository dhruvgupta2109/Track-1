import sys, copy; sys.path.insert(0, "src")
from common import load, cross_validate, test_mask_patterns
from imputed import ImputeLinear
from specs import V4
tr, te = load(); P = test_mask_patterns(te); P = P[P.any(1)]
ALL = ["occupancy", "previous_usage", "temperature", "humidity"]
def run(name, inter=None, **kw):
    s = copy.deepcopy(V4)
    if inter: s["interactions"] = inter
    s.update(kw)
    cross_validate(name, lambda: ImputeLinear(s, cols=ALL, extra_rows=te, aug_patterns=P, aug_copies=1), tr, te, repeats=2)
run("s00_c04_ref")
run("s01_prev_x_type", [("previous_usage", "type", 0.5)])
run("s02_prev_x_building", [("previous_usage", "building", 0.5)])
run("s03_occ_x_weekend", [("occupancy", "weekend", 0.5)])
run("s04_occ_x_hourband", [("occupancy", "hourband", 0.5)])
run("s05_prev_x_hourband", [("previous_usage", "hourband", 0.5)])
run("s06_prev_x_weekend", [("previous_usage", "weekend", 0.5)])
run("s07_temp_hinge_type", [("temp_hinge", "type", 0.5)])
run("s08_hum_x_type", [("humidity", "type", 0.5)])
run("s09_temp_x_hourband", [("temperature", "hourband", 0.5)])
run("s10_occ_x_type_hourband", [("occupancy", "hourband", 0.5), ("occupancy", "type", 0.5)])
