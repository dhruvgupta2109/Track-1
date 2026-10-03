import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, RepeatedKFold
tr=pd.read_csv("train.csv").dropna().reset_index(drop=True); y=tr.energy_usage.values
B=pd.get_dummies(tr.building_id,dtype=float); T=pd.get_dummies(tr.building_type,dtype=float)
def inter(cols,G): return pd.concat([(G[b]*c.values).rename(f"{b}:{c.name}") for c in cols for b in G],axis=1)
H=pd.get_dummies(tr.hour,prefix="h",dtype=float); hc=[H[c] for c in H]
we=tr.day_of_week.isin(["Saturday","Sunday"]).astype(float).rename("we")
D=pd.get_dummies(tr.day_of_week,dtype=float)
ms=pd.DataFrame({"s":np.sin(2*np.pi*tr.month/12),"c":np.cos(2*np.pi*tr.month/12)}); t2=((tr.temperature-28)**2).rename("t2")
num={c:tr[c] for c in ["previous_usage","occupancy","temperature","humidity"]}
kf=RepeatedKFold(n_splits=5,n_repeats=3,random_state=0)
def cv(X,a=3.0):
    s=[]
    for i,(a_,b_) in enumerate(kf.split(X)):
        m=Ridge(alpha=a).fit(X.iloc[a_],y[a_]); s.append(np.mean((y[b_]-m.predict(X.iloc[b_]))**2))
    return round(np.sqrt(np.mean(s)),4)
def build(slopes, hours, wk="B", extra=()):
    G={"B":B,"T":T}
    parts=[B, inter(hc,G[hours]), inter([we],G[wk]), ms, t2]
    for c,v in num.items():
        g=slopes.get(c,"B")
        parts.append(v.rename(c).to_frame() if g=="S" else inter([v],G[g]))
    return pd.concat(parts+list(extra),axis=1)
print("current (all per-bldg)", cv(build({},"B")))
print("hours per type", cv(build({},"T")))
print("prev shared", cv(build({"previous_usage":"S"},"B")))
print("prev+hum shared", cv(build({"previous_usage":"S","humidity":"S"},"B")))
print("all slopes per type", cv(build({c:"T" for c in num},"B")))
print("prev,hum shared; occ,temp per type; hours bldg", cv(build({"previous_usage":"S","humidity":"S","occupancy":"T","temperature":"T"},"B")))
print("prev,hum shared; hours type", cv(build({"previous_usage":"S","humidity":"S"},"T")))
print("prev shared + dow dummies", cv(build({"previous_usage":"S"},"B",extra=[D])))
print("prev shared, weekend per type", cv(build({"previous_usage":"S"},"B","T")))
