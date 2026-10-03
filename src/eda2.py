import pandas as pd, numpy as np
tr=pd.read_csv("train.csv"); te=pd.read_csv("test.csv")
tr["n"]=tr.id.str[2:].astype(int); te["n"]=te.id.str[2:].astype(int)
print("corr id vs target/month/hour", tr[["n","energy_usage","month","hour","temperature"]].corr().n.round(3).to_dict())
# missing pattern co-occurrence
for df,nm in [(tr,"train"),(te,"test")]:
    m=df[["temperature","humidity","occupancy","previous_usage"]].isna()
    print(nm, "rows with any missing", m.any(axis=1).mean().round(4), "n missing dist", m.sum(1).value_counts().to_dict())
    print(nm,"temp&hum both missing", (m.temperature&m.humidity).sum())
# chaining: does previous_usage equal energy_usage of some other row in same building?
s=set(zip(tr.building_id, tr.energy_usage.round(1)))
hit=np.mean([ (b,p) in s for b,p in zip(tr.building_id, tr.previous_usage.round(1)) if not np.isnan(p)])
print("prev matches same-building target value frac", hit)
# baseline expected hit rate by chance: shuffle building
rb=tr.building_id.sample(frac=1,random_state=0).values
print("chance", np.mean([(b,p) in s for b,p in zip(rb, tr.previous_usage.round(1)) if not np.isnan(p)]))
# value granularity
print("decimals", (tr.energy_usage*10%1).abs().max(), tr.occupancy.dropna().apply(float.is_integer).mean())
# relationship: residual target-prev by building
tr["d"]=tr.energy_usage-tr.previous_usage
print(tr.groupby("building_id").d.agg(["mean","std"]).round(2))
print(tr.groupby("hour").energy_usage.mean().round(1).to_dict())
print(tr.groupby("day_of_week").energy_usage.mean().round(1).to_dict())
print(tr.groupby("month").energy_usage.mean().round(1).to_dict())
# per building, combos of month/day/hour: unique timestamps?
print("dup building,month,dow,hour", tr.duplicated(["building_id","month","day_of_week","hour"]).sum())
print("test dup within test", te.duplicated(["building_id","month","day_of_week","hour"]).sum())
k=["building_id","month","day_of_week","hour"]
print("test keys in train", te.merge(tr[k].drop_duplicates(),on=k).shape[0])
print("hum==100 frac", (tr.humidity==100).mean(), "occ==0 frac", (tr.occupancy==0).mean())
print(tr.groupby("building_type").occupancy.describe().round(1))
