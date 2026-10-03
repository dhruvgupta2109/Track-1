"""Explicit sinusoidal weather generator with a two-variable noise mixture."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from common import ROOT, load, apply_masks, test_mask_patterns
from generative_imputer import moments
from proxy import SL


def clock(X):
    h = X.hour.to_numpy(); m = X.month.to_numpy()
    return np.c_[np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                 np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)]


def group(X, dom):
    return np.asarray(dom).astype(int)*6+np.digitize(X.hour.to_numpy(), [4,7,10,19,22])


def fingerprint():
    tr,te=load();result={}
    for pool,X in [("train",tr),("test",te)]:
        result[pool]={}
        for col in ["temperature","humidity"]:
            ok=X[col].notna();A=clock(X[ok]);m=HuberRegressor(max_iter=1000).fit(A,X.loc[ok,col])
            r=X.loc[ok,col].to_numpy()-m.predict(A)
            result[pool][col]=dict(intercept=float(m.intercept_),coefficients=m.coef_.tolist(),
                                  central_residual_sd=float(np.std(r[np.abs(r)<3*np.std(r)])))
        complete=X.dropna(subset=["temperature","humidity"])
        m=HuberRegressor(max_iter=1000).fit(np.c_[clock(complete),complete.temperature],complete.humidity)
        result[pool]["humidity_on_temperature_and_clock_coefficient"]=float(m.coef_[-1])
    (ROOT/"experiments/weather_generator_fingerprint.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


class Weather:
    def __init__(self, components):
        self.components = components

    def fit(self, X, dom):
        A = clock(X); Y = X[["temperature", "humidity"]].to_numpy(); w = np.where(dom == 1, 2., 1.)
        self.models = [HuberRegressor(max_iter=1000).fit(A[np.isfinite(Y[:,j])], Y[np.isfinite(Y[:,j]),j],
                                                      sample_weight=w[np.isfinite(Y[:,j])]) for j in range(2)]
        mu = np.stack([m.predict(A) for m in self.models], axis=1)
        self.scale = np.array([.8,4.]); R = (Y-mu)/self.scale
        complete = R[np.isfinite(R).all(axis=1)]
        init = GaussianMixture(self.components, random_state=54, reg_covar=.01, n_init=2).fit(complete)
        means, covs = init.means_, init.covariances_; probs = np.tile(init.weights_,(12,1)); g = group(X,dom)
        for it in range(50):
            resp, ex, vc, _ = moments(R,means,covs,probs,g)
            rw = resp*w[:,None]; nk = rw.sum(axis=0)
            means = np.einsum("nk,nkd->kd",rw,ex)/nk[:,None]
            covs = (np.einsum("nk,nkde->kde",rw,vc)+np.einsum("nk,nkd,nke->kde",rw,ex,ex))/nk[:,None,None]
            covs -= np.einsum("kd,ke->kde",means,means); covs += np.eye(2)[None]*.01
            for gg in range(12):
                count = rw[g == gg].sum(axis=0)+40*nk/nk.sum(); probs[gg] = count/count.sum()
        self.means,self.covs,self.probs = means,covs,probs
        return self

    def predict(self,X,dom):
        A = clock(X); mu = np.stack([m.predict(A) for m in self.models],axis=1)
        Y = X[["temperature","humidity"]].to_numpy(); R = (Y-mu)/self.scale
        resp,ex,_,_ = moments(R,self.means,self.covs,self.probs,group(X,np.full(len(X),dom)),False)
        return np.einsum("nk,nkd->nd",resp,ex)*self.scale+mu


def main():
    threadpool_limits(4);fingerprint();tr,te=load();XTR=tr.drop(columns="energy_usage");XTE=te[XTR.columns];P=test_mask_patterns(te)
    kt=list(KFold(5,shuffle=True,random_state=1).split(XTR));ke=list(KFold(5,shuffle=True,random_state=0).split(XTE));saved=[]
    for k,((ta,tb),(ea,eb)) in enumerate(zip(kt,ke)):
        X=pd.concat([XTR.iloc[ta],XTE.iloc[ea]],ignore_index=True);dom=np.r_[np.zeros(len(ta)),np.ones(len(ea))]
        for nc in [1,4,8]:
            model=Weather(nc).fit(X,dom)
            for col in ["temperature","humidity"]:
                for pool,H0,ids,d in [("test",XTE.iloc[eb].reset_index(drop=True),eb,1),("train",XTR.iloc[tb].reset_index(drop=True),tb,0)]:
                    ok=H0[col].notna().to_numpy();H=apply_masks(H0[ok],P,seed=100+k);H[col]=np.nan
                    p=model.predict(H,d)[:,["temperature","humidity"].index(col)];truth=H0.loc[ok,col].to_numpy()
                    s=H.building_id.map(SL[col]).to_numpy();nat=H0.loc[ok,[c for c in SL if c != col]].isna().any(axis=1).to_numpy()
                    for i,e,ss,natural in zip(np.asarray(ids)[ok],p-truth,s,nat):
                        saved.append(dict(fold=k,pool=pool,feature=col,row=int(i),variant=f"weather{nc}",error=float(e),
                                          energy_error=float(e*ss),naturally_incomplete=bool(natural)))
        print(f"weather fold {k+1}/5",flush=True)
    E=pd.DataFrame(saved);E.to_csv(ROOT/"experiments/weather_generator_errors.csv.gz",index=False)
    W=pd.read_csv(ROOT/"experiments/energy_weighted_imputers_errors.csv.gz")
    R=W[((W.feature=="occupancy")&(W.variant=="blend_weighted"))|((W.feature!="occupancy")&(W.variant=="c11_baseline"))].assign(variant="c19")
    idx=["fold","pool","feature","row"];variants=[R]
    for v in E.variant.unique():
        A=E[E.variant==v].set_index(idx).reindex(pd.MultiIndex.from_frame(R[idx]));S=R.copy().assign(variant=v)
        use=S.feature.isin(["temperature","humidity"])
        for c in ["error","energy_error"]:S.loc[use,c]=A.loc[use.to_numpy(),c].to_numpy()
        variants.append(S);B=S.copy().assign(variant=f"blend_{v}")
        for c in ["error","energy_error"]:B[c]=.5*(S[c].to_numpy()+R[c].to_numpy())
        variants.append(B)
    E=pd.concat(variants,ignore_index=True);out={}
    for pool,sel in [("test",E.pool=="test"),("natmiss",(E.pool=="test")&E.naturally_incomplete),("train",E.pool=="train")]:
        T=E[sel].assign(mse=lambda d:d.energy_error**2).groupby(["variant","feature"]).mse.mean().unstack()
        T["total"]=T.sum(axis=1);out[pool]=T.to_dict(orient="index");print(pool,T.round(4).to_string(),flush=True)
    (ROOT/"experiments/weather_generator.json").write_text(json.dumps(out,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
