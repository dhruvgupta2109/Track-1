import pandas as pd, numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold
tr=pd.read_csv("train.csv").dropna().reset_index(drop=True); y=tr.energy_usage.values
B=pd.get_dummies(tr.building_id,dtype=float)
def inter(cols):
    parts=[]
    for c in cols:
        for b in B: parts.append((B[b]*c.values).rename(f"{b}:{c.name}"))
    return pd.concat(parts,axis=1)
H=pd.get_dummies(tr.hour,prefix="h",dtype=float); M=pd.get_dummies(tr.month,prefix="m",dtype=float); D=pd.get_dummies(tr.day_of_week,dtype=float)
we=tr.day_of_week.isin(["Saturday","Sunday"]).astype(float).rename("we")
num=[tr[c] for c in ["previous_usage","occupancy","temperature","humidity"]]
kf=KFold(5,shuffle=True,random_state=0)
def cv(X,a=1.0):
    p=cross_val_predict(Ridge(alpha=a),X,y,cv=kf); return np.sqrt(np.mean((y-p)**2))
base=pd.concat([B]+[inter([c]) for c in num],axis=1)
print("bldg x num", cv(base))
print("+hour shared", cv(pd.concat([base,H],axis=1)))
X=pd.concat([base,inter([H[c] for c in H])],axis=1); print("+bldg x hour", cv(X))
X2=pd.concat([X,inter([we])],axis=1); print("+bldg x weekend", cv(X2))
X3=pd.concat([X2,M],axis=1); print("+month shared", cv(X3))
X4=pd.concat([X2,inter([M[c] for c in M])],axis=1); print("+bldg x month", cv(X4))
X5=pd.concat([X3,D],axis=1); print("+dow shared", cv(X5))
for a in [0.1,0.3,3,10]: print("alpha",a, cv(X3,a))
# nonlinearity check: residual vs temp/occ bins
from sklearn.linear_model import Ridge as R
p=cross_val_predict(R(alpha=1),X3,y,cv=kf); r=y-p
for c in ["temperature","occupancy","humidity","previous_usage","hour"]:
    print(c, pd.Series(r).groupby(pd.qcut(tr[c],10,duplicates="drop")).mean().round(2).values)
print("resid by bldg", pd.Series(r).groupby(tr.building_id).std().round(2).to_dict())
print("resid kurtosis", pd.Series(r).kurt(), "quantiles", np.quantile(r,[.001,.01,.5,.99,.999]).round(2))
