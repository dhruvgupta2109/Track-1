import sys; sys.path.insert(0, "src")
from common import load, cross_validate
from linmodels import PatternRidge
tr, te = load()
for a in [1.0, 3.0, 10.0]:
    cross_validate(f"l01_pattern_ridge_a{a}", lambda: PatternRidge(a), tr, te)
