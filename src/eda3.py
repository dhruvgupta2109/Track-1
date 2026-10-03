import pandas as pd, numpy as np
tr=pd.read_csv("train.csv"); te=pd.read_csv("test.csv")
C=["temperature","humidity","occupancy","previous_usage"]
pat=te[C].isna().astype(int).astype(str).agg("".join,axis=1); print(pat.value_counts())
patt=tr[C].isna().astype(int).astype(str).agg("".join,axis=1); print(patt.value_counts())
te["anym"]=te[C].isna().any(axis=1)
print(te.groupby("building_id").anym.mean().round(3).to_dict())
print(te.groupby("hour").anym.mean().round(2).to_dict())
# do numeric distributions differ train vs test (non-missing)?
for c in C+["hour","month"]: print(c, tr[c].describe()[["mean","std","min","max"]].round(2).to_dict(), te[c].describe()[["mean","std","min","max"]].round(2).to_dict())
# missing in train: does target differ?
for c in C: print(c, "target mean when missing", tr.loc[tr[c].isna(),"energy_usage"].mean().round(2))
