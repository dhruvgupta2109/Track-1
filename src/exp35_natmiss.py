import sys; sys.path.insert(0, "src")
from proxy import evaluate, lgb_member, IMP_PARAMS
from exp34_devfeat import multi_member, DEEP, SLOW
L = [["building", "hour"], ["building"]]
g = sys.argv[1]
if g == "c07": evaluate([lgb_member(IMP_PARAMS, 11), lgb_member(DEEP, 12), lgb_member(SLOW, 13)], 2.0, label="c07 (LB 3.23149)")
if g == "c08": evaluate([lgb_member(IMP_PARAMS, 11, 5), lgb_member(DEEP, 12, 5), lgb_member(SLOW, 13, 5)], 2.0, label="c08 (LB 3.20937)")
if g == "c09": evaluate([multi_member(IMP_PARAMS, L, 11, 5), multi_member(DEEP, L, 12, 5), multi_member(SLOW, L, 13, 5)], 2.0, label="c09 (LB 3.20392)")
if g == "c06": evaluate([lgb_member(IMP_PARAMS, 11), lgb_member(DEEP, 12), lgb_member(SLOW, 13)], 1.0, label="c06 (LB 3.23804)")
