"""TabICL soft labels -> platform-compatible LightGBM student.

Outer feature holdouts are never used to create training queries or labels.
Default: inner three-fold teachers exclude each query row's feature target.
Synthetic mode: the full outer-fold teacher labels jittered unlabeled queries;
no held-out features enter context, queries or targets. Counterfactual query
targets are not ground truth, so training RMS differences are diagnostic only.
Student sees the same fixed deviation features and test-like mask augmentation
as the frozen pipeline, and fits soft labels rather than noisy observations.
"""
import argparse
import gc
import json
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from candidates._c11 import CONFIGS
from common import ROOT, load, test_mask_patterns, apply_masks, NUM
from devimpute import LEVELS, make_tables, add_dev, add_cyclic
from features import make_features, NUMERIC
from imputed import _mask_aug
from proxy import SL
from exp50_tabicl import regressor, numerical


def fit_student(rows, domain, patterns, copies=5, trio=True, synthetic=False):
    F0 = add_cyclic(make_features(rows).assign(dom=domain)); tabs = make_tables(F0, LEVELS)
    PN = patterns[patterns.any(axis=1)][:, [NUM.index(c) for c in NUMERIC]]
    n = len(F0); configs = CONFIGS if trio else CONFIGS[:1]; students = {}; debug = {}
    for col in ["occupancy", "previous_usage"]:
        t0 = time.time()
        F = _mask_aug(F0, col, PN, copies, seed=11)
        if synthetic:
            # New unlabeled queries approximate the full teacher's function.
            # Jitter prevents copying targets through an exact context-row match.
            rng = np.random.default_rng(713)
            for other in NUMERIC:
                if other == col:
                    continue
                sd = F0.groupby("building", observed=True)[other].std().to_dict()
                noise_scale = F.building.astype(object).map(sd).to_numpy(dtype=float)*.15
                values = F[other].to_numpy()+rng.normal(size=len(F))*noise_scale
                from features import VALID
                F[other] = np.clip(values, *VALID[other])
        pseudo = np.zeros(len(F))
        inner_splits = [(np.arange(n), np.arange(n))] if synthetic else KFold(3, shuffle=True, random_state=517).split(F0)
        for inner, (a, b) in enumerate(inner_splits):
            tabs_inner = make_tables(F0.iloc[a], LEVELS)
            A = numerical(add_dev(F0.iloc[a], tabs_inner, LEVELS, col).drop(columns=col))
            ok = F0.iloc[a][col].notna().to_numpy()
            model = regressor().set_params(kv_cache="repr").fit(A[ok], F0.iloc[a].loc[ok, col].to_numpy())
            query_rows = np.concatenate([b+i*n for i in range(copies+1)])
            Q = numerical(add_dev(F.iloc[query_rows], tabs_inner, LEVELS, col).drop(columns=col))
            # Fixed chunks bound GPU memory; cache avoids recomputing context.
            pseudo[query_rows] = np.concatenate([model.predict(Q.iloc[i:i+2048]) for i in range(0, len(Q), 2048)])
            print(f"  {col} teacher query batch {inner+1}, queries {len(Q)}, elapsed {time.time()-t0:.1f}s", flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()
        sw = np.where(np.asarray(domain) == 1, 2., 1.)
        slope = rows.building_id.map(SL[col]).to_numpy()**2
        slope /= np.average(slope, weights=sw); weights = np.tile(sw*slope, copies+1)
        A = add_dev(F, tabs, LEVELS, col).drop(columns=col)
        # Same queries/soft labels across ensemble members: only model capacity differs.
        students[col] = []
        for params, seed in configs:
            model = lgb.LGBMRegressor(**params).fit(A, pseudo, sample_weight=weights)
            students[col].append(model.booster_.model_to_string())
        observed = F[col].notna().to_numpy()
        debug[col] = dict(teacher_target_rms_difference=float(np.sqrt(np.mean((pseudo[observed]-F.loc[observed,col])**2))),
                          queries=len(F), seconds=time.time()-t0)
        print(f"{col} student fitted {debug[col]}", flush=True)
    return students, tabs, debug


def main(args):
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.72)
    tr, te = load(); XTR = tr.drop(columns="energy_usage"); XTE = te[XTR.columns]
    kt = list(KFold(5, shuffle=True, random_state=1).split(XTR))
    ke = list(KFold(5, shuffle=True, random_state=0).split(XTE)); P = test_mask_patterns(te); saved = []
    mode = "synthetic" if args.synthetic else "crossfit"
    tag = f"tabicl_distill_{mode}_{args.folds}fold_aug{args.copies}_{'trio' if args.trio else 'single'}"
    done = set()
    if args.resume_from:
        cache = pd.read_csv(ROOT/args.resume_from)
        assert set(cache.variant) == {"student"}
        for k, fold in cache.groupby("fold"):
            assert set(zip(fold.pool, fold.feature)) == {("test", "occupancy"), ("train", "occupancy"),
                                                       ("test", "previous_usage"), ("train", "previous_usage")}
            done.add(int(k))
        saved = cache.to_dict(orient="records")
        print(f"Resume completed paired folds {sorted(done)} from {args.resume_from}", flush=True)
    for k, ((ta,tb),(ea,eb)) in enumerate(zip(kt[:args.folds],ke[:args.folds])):
        if k in done:
            continue
        rows = pd.concat([XTR.iloc[ta], XTE.iloc[ea]], ignore_index=True)
        dom = np.r_[np.zeros(len(ta)), np.ones(len(ea))]
        students, tabs, _ = fit_student(rows, dom, P, copies=args.copies, trio=args.trio, synthetic=args.synthetic)
        for col, strings in students.items():
            models = [lgb.Booster(model_str=s) for s in strings]
            for pool, hold, ids, domain in [("test", XTE.iloc[eb].reset_index(drop=True), eb, 1),
                                           ("train", XTR.iloc[tb].reset_index(drop=True), tb, 0)]:
                valid = hold[col].notna().to_numpy(); H = apply_masks(hold[valid], P, seed=100+k); H[col] = np.nan
                G = add_cyclic(make_features(H).assign(dom=domain)); A = add_dev(G, tabs, LEVELS, col).drop(columns=col)
                pred = np.mean([m.predict(A) for m in models], axis=0)
                slope = H.building_id.map(SL[col]).to_numpy(); true = hold.loc[valid,col].to_numpy()
                nat = hold.loc[valid,[c for c in NUMERIC if c != col]].isna().any(axis=1).to_numpy()
                for i,e,s,natural in zip(np.asarray(ids)[valid], pred-true, slope, nat):
                    saved.append(dict(fold=k,pool=pool,feature=col,row=int(i),variant="student",
                                      error=float(e),energy_error=float(e*s),naturally_incomplete=bool(natural)))
        pd.DataFrame(saved).to_csv(ROOT/f"experiments/{tag}_errors.csv.gz", index=False)
        print(f"fold {k+1}/{args.folds} completed",flush=True)
    E = pd.DataFrame(saved)
    W = pd.read_csv(ROOT/"experiments/energy_weighted_imputers_errors.csv.gz")
    R = W[((W.feature == "occupancy") & (W.variant == "blend_weighted")) |
          ((W.feature != "occupancy") & (W.variant == "c11_baseline"))]
    R = R[R.fold < args.folds].assign(variant="c19")
    shared = R[~R.feature.isin(E.feature.unique())].assign(variant="student")
    E = pd.concat([E,shared],ignore_index=True); idx=["fold","pool","feature","row"]
    A = E.set_index(idx).reindex(pd.MultiIndex.from_frame(R[idx])); Z=R.copy().assign(variant="blend_student")
    for c in ["error","energy_error"]: Z[c]=.5*(R[c].to_numpy()+A[c].to_numpy())
    E=pd.concat([E,R,Z],ignore_index=True); result={}
    for pool,sel in [("test",E.pool=="test"),("natmiss",(E.pool=="test")&E.naturally_incomplete),("train",E.pool=="train")]:
        T=E[sel].assign(mse=lambda d:d.energy_error**2).groupby(["variant","feature"]).mse.mean().unstack()
        T["total"]=T.sum(axis=1);result[pool]=T.to_dict(orient="index");print(pool,T.round(4).to_string(),flush=True)
    (ROOT/f"experiments/{tag}.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--folds",type=int,default=2)
    parser.add_argument("--copies",type=int,default=5);parser.add_argument("--trio",action="store_true")
    parser.add_argument("--synthetic",action="store_true")
    parser.add_argument("--resume-from")
    main(parser.parse_args())
