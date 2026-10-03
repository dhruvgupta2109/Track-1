import sys; sys.path.insert(0,"src")
import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_predict, KFold
from features import make_features
tr=pd.read_csv("train.csv"); te=pd.read_csv("test.csv")
Ftr=make_features(tr); Fte=make_features(te)
for c in ["previous_usage","temperature","humidity","occupancy"]:
    X=Ftr.drop(columns=[c]); ok=Ftr[c].notna()
    m=lgb.LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=15,verbose=-1,n_jobs=4)
    r_tr=Ftr[c][ok]-cross_val_predict(m,X[ok],Ftr[c][ok],cv=KFold(5,shuffle=True,random_state=0))
    m.fit(X[ok],Ftr[c][ok]); okt=Fte[c].notna(); r_te=Fte[c][okt]-m.predict(Fte.drop(columns=[c])[okt])
    s=r_tr.std(); q=lambda r:(np.abs(r)>3*s).mean()
    print(f"{c:15s} resid std train {r_tr.std():.2f} test {r_te.std():.2f} | MAD train {np.median(np.abs(r_tr)):.2f} test {np.median(np.abs(r_te)):.2f} | frac >3sd train {q(r_tr):.4f} test {q(r_te):.4f}")
