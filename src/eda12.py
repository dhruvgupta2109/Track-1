import sys; sys.path.insert(0,"src")
import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from features import make_features, NUMERIC
tr=pd.read_csv("train.csv"); te=pd.read_csv("test.csv"); F=make_features(tr); Ft=make_features(te)
for c in ["previous_usage","occupancy","temperature"]:
    X=F.drop(columns=[c]); ok=F[c].notna()
    m=lgb.LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=15,verbose=-1,n_jobs=4).fit(X[ok],F[c][ok])
    rt=(F[c]-m.predict(X)); rte=(Ft[c]-m.predict(Ft.drop(columns=[c]))); s=rt.std()
    zt, zte = rt/s, rte/s
    print(c, "train z quantiles", np.nanquantile(zt,[.001,.01,.99,.999]).round(1), "test", np.nanquantile(zte,[.001,.01,.99,.999]).round(1),
          "| test frac z>3:", round((zte>3).mean(),4), "z<-3:", round((zte<-3).mean(),4), "train z>3:", round((zt>3).mean(),4), "z<-3:", round((zt<-3).mean(),4))
