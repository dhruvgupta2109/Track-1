import sys; sys.path.insert(0,"src")
import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_predict, KFold
from features import make_features, clean_input, predict_linear, NUMERIC
from linmodels import fit_linear_artifact
tr=pd.read_csv("train.csv"); F=make_features(tr); y=tr.energy_usage.values
# outlier flags from feature-consistency model (no target)
flag={}; z={}
for c in NUMERIC:
    X=F.drop(columns=[c]); ok=F[c].notna()
    m=lgb.LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=15,verbose=-1,n_jobs=4)
    r=pd.Series(np.nan,index=tr.index); r[ok]=F[c][ok]-cross_val_predict(m,X[ok],F[c][ok],cv=KFold(5,shuffle=True,random_state=0))
    z[c]=r/r.std(); flag[c]=(z[c].abs()>3).fillna(False).values
# OOF: full model vs model ignoring column c on the flagged rows
kf=KFold(5,shuffle=True,random_state=0); X=tr.drop(columns="energy_usage")
full=np.zeros(len(tr)); drop={c:np.zeros(len(tr)) for c in NUMERIC}
for a,b in kf.split(X):
    art=fit_linear_artifact(X.iloc[a],y[a],3.0)
    full[b]=predict_linear(art,X.iloc[b])
    for c in NUMERIC:
        Xb=X.iloc[b].copy(); Xb[c]=np.nan; drop[c][b]=predict_linear(art,Xb)
rm=lambda e: np.sqrt(np.mean(e**2))
for c in NUMERIC:
    f=flag[c]; print(f"{c:15s} n_flag {f.sum():4d} | RMSE using feature {rm(y[f]-full[f]):.2f} | ignoring it {rm(y[f]-drop[c][f]):.2f} | unflagged rows using {rm(y[~f]-full[~f]):.2f}")
    # direction: does residual of 'drop' model correlate with feature outlier z?
    e=(y-drop[c])[f]; print("   corr(target-minus-dropmodel, feature z) on flagged:", np.corrcoef(e, z[c][f])[0,1].round(3))
pd.DataFrame({c:z[c] for c in NUMERIC}).to_csv("experiments/train_feature_z.csv",index=False)
print("---- linearity at extremes: residual of full model vs feature z on flagged rows")
for c in NUMERIC:
    f=flag[c]; e=(y-full)[f]; zz=z[c][f].values
    b=np.polyfit(zz,e,1); print(f"{c:15s} resid mean {e.mean():+.2f}, slope of resid on z {b[0]:+.3f}; pos-z mean resid {e[zz>0].mean():+.2f} (n={int((zz>0).sum())}), neg-z {e[zz<0].mean() if (zz<0).any() else float('nan'):+.2f} (n={int((zz<0).sum())})")
