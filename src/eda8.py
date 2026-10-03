import sys; sys.path.insert(0,"src")
import numpy as np, pandas as pd, lightgbm as lgb
from common import load, NUM
from linmodels import PatternRidge
tr, te = load()
r = PatternRidge(3.0).fit(tr.drop(columns="energy_usage"), tr.energy_usage.values)
pr = r.predict(te); pl = pd.read_csv("submissions/c00_lgb_baseline.csv").prediction.values
d = pr - pl
q = tr.groupby("building_id")[NUM].quantile([.005,.995]).unstack()
out = np.zeros(len(te),bool)
for c in NUM:
    lo = te.building_id.map(q[(c,.005)]); hi = te.building_id.map(q[(c,.995)])
    out |= ((te[c]<lo)|(te[c]>hi)).to_numpy()
print("test rows out-of-range any numeric:", out.sum())
print("rms diff ridge-lgb: in-range %.3f, out-of-range %.3f" % (np.sqrt(np.mean(d[~out]**2)), np.sqrt(np.mean(d[out]**2))))
print("pred range ridge", pr.min().round(1), pr.max().round(1), "lgb", pl.min().round(1), pl.max().round(1), "train y", tr.energy_usage.min(), tr.energy_usage.max())
print("ridge prev slope per building", {b: round(m,3) for b,m in zip(sorted(tr.building_id.unique()), r.models[()].coef_[12+288+12:12+288+24])})
# mean shift: test pred mean vs train target mean per building
print(pd.DataFrame({"b":te.building_id,"pr":pr,"pl":pl}).groupby("b").mean().round(1).T)
