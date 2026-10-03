import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold
tr=pd.read_csv("train.csv").dropna().reset_index(drop=True); y=tr.energy_usage.values
B=pd.get_dummies(tr.building_id,dtype=float); T=pd.get_dummies(tr.building_type,dtype=float)
def inter(cols, G=B):
    return pd.concat([(G[b]*c.values).rename(f"{b}:{c.name}") for c in cols for b in G],axis=1)
H=pd.get_dummies(tr.hour,prefix="h",dtype=float); M=pd.get_dummies(tr.month,prefix="m",dtype=float)
we=tr.day_of_week.isin(["Saturday","Sunday"]).astype(float).rename("we")
t=tr.temperature; o=tr.occupancy
num=[tr[c] for c in ["previous_usage","occupancy","temperature","humidity"]]
kf=KFold(5,shuffle=True,random_state=0)
def cv(X,a=3.0):
    p=cross_val_predict(Ridge(alpha=a),X,y,cv=kf); return round(np.sqrt(np.mean((y-p)**2)),4), p
base=pd.concat([B]+[inter([c]) for c in num]+[inter([H[c] for c in H]), inter([we]), M],axis=1)
s0,p0=cv(base); print("base",s0)
for nm,extra in {"t2 shared":[((t-28)**2).rename("t2")],
                 "t2 bldg":[inter([((t-28)**2).rename("t2")])],
                 "o2 bldg":[inter([(o**2/100).rename("o2")])],
                 "sqrt o bldg":[inter([np.sqrt(o).rename("so")])],
                 "occ x we bldg":[inter([(o*we).rename("ow")])],
                 "hour x we bldg":[inter([(H[c]*we).rename(c+"we") for c in H])],
                 "temp x hour shared":[pd.concat([(H[c]*t).rename(c+"t") for c in H],axis=1)],
                 "prev x hour shared":[pd.concat([(H[c]*tr.previous_usage).rename(c+"p") for c in H],axis=1)],
                 "temp x occ bldg":[inter([(t*o/100).rename("to")])],
                 "hum x temp":[ (t*tr.humidity/100).rename("th")],
                 "cdd hinge 26-30 bldg":[inter([np.maximum(t-k,0).rename(f"cdd{k}") for k in [26,28,30]])],
                }.items():
    s,_=cv(pd.concat([base]+extra,axis=1)); print(nm, s, round(s-s0,4))
# month shape
from sklearn.linear_model import Ridge as R
m=R(alpha=3).fit(base,y); print("month coefs", dict(zip(M.columns, m.coef_[-12:].round(2))))
