"""Paired row bootstrap of full student holdout and compact-energy blend."""
import json
import numpy as np
import pandas as pd
from common import ROOT


def uncertainty(dr):
    rng=np.random.default_rng(20261003);v=dr.to_numpy() if hasattr(dr,"to_numpy") else dr
    boot=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(2000)])
    return dict(delta_mse=float(np.mean(v)),se=float(np.std(v,ddof=1)/np.sqrt(len(v))),
                ci95=np.quantile(boot,[.025,.975]).tolist())


def main():
    S=pd.read_csv(ROOT/"experiments/tabicl_distill_synthetic_5fold_aug5_trio_errors.csv.gz")
    W=pd.read_csv(ROOT/"experiments/energy_weighted_imputers_errors.csv.gz")
    R=W[((W.feature=="occupancy")&(W.variant=="blend_weighted"))|((W.feature!="occupancy")&(W.variant=="c11_baseline"))].assign(variant="c19")
    idx=["fold","pool","feature","row"];A=S.set_index(idx).reindex(pd.MultiIndex.from_frame(R[idx]))
    blend=R.copy().assign(variant="blend_student");use=R.feature.isin(S.feature.unique())
    for c in ["error","energy_error"]:blend.loc[use,c]=.5*(R.loc[use,c].to_numpy()+A.loc[use.to_numpy(),c].to_numpy())
    E=pd.concat([R,blend],ignore_index=True);result={}
    for pool,sel in [("test",E.pool=="test"),("natmiss",(E.pool=="test")&E.naturally_incomplete),("train",E.pool=="train")]:
        T=E[sel].pivot(index=idx,columns="variant",values="energy_error")
        count=T.groupby(level="feature").size();nr=len(T.index.droplevel("feature").unique())
        factor=nr/T.index.get_level_values("feature").map(count).to_numpy()
        dr=((T.blend_student**2-T.c19**2)*factor).groupby(level=["fold","row"]).sum()
        result[pool]=uncertainty(dr);result[pool]["fold_contribution_means"]=dr.groupby(level="fold").mean().to_dict()
    z=np.load(ROOT/"experiments/compact_formula_oof.npz");a=z["v5"];b=z["compact"];y=z["y"];p=.5*(a+b)
    dr=np.mean((y-p)**2-(y-a)**2,axis=0)
    result["compact_energy_blend"]=uncertainty(dr)
    result["compact_energy_blend"].update(baseline_rmse=float(np.sqrt(np.mean((y-a)**2))),
                                           blend_rmse=float(np.sqrt(np.mean((y-p)**2))))
    (ROOT/"experiments/research_improvement_uncertainty.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
