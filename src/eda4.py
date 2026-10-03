import pandas as pd, numpy as np
tr=pd.read_csv("train.csv"); te=pd.read_csv("test.csv")
g=lambda d,c: d.groupby("building_id")[c].agg(["mean","std"]).round(1)
for c in ["previous_usage","temperature","occupancy","humidity"]:
    print(c); print(pd.concat([g(tr,c),g(te,c)],axis=1,keys=["train","test"]))
# tails: fraction beyond train 1/99 pct per building
for c in ["previous_usage","temperature"]:
    q=tr.groupby("building_id")[c].quantile([.005,.995]).unstack()
    t=te.join(q,on="building_id"); out=((t[c]<t[0.005])|(t[c]>t[0.995])).mean()
    print(c,"test frac outside train per-bldg 0.5-99.5pct", round(out,3))
# temperature vs hour/month in train & test
print(pd.concat([tr.groupby("month").temperature.mean(),te.groupby("month").temperature.mean()],axis=1).round(2).T)
print(pd.concat([tr.groupby("hour").temperature.std(),te.groupby("hour").temperature.std()],axis=1).round(2).T)
