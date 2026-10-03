import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
tr=pd.read_csv("train.csv").dropna()
for b,g in tr.groupby("building_id"):
    X=g[["previous_usage"]]; y=g.energy_usage
    lr=LinearRegression().fit(X,y); r=y-lr.predict(X)
    # full linear with occupancy, temp, hour dummies
    X2=pd.concat([g[["previous_usage","occupancy","temperature","humidity"]], pd.get_dummies(g.hour,prefix="h",dtype=float), (g.day_of_week.isin(["Saturday","Sunday"])).astype(float).rename("we")],axis=1)
    lr2=LinearRegression().fit(X2,y); r2=y-lr2.predict(X2)
    print(b, "slope prev %.3f int %.2f rmse %.2f | full lin: coef prev %.3f occ %.3f temp %.3f hum %.3f rmse %.2f" % (lr.coef_[0], lr.intercept_, r.std(), *lr2.coef_[:4], r2.std()))
