"""Research CSV: fixed 50/50 c19/TabICL occupancy and previous-usage fills.

Requires the isolated TabICL environment and public checkpoint. This is NOT a
platform-compatible inference package: the supplied platform image lacks Torch
and TabICL. c19 remains the usable private package while distillation is tested.
"""
import hashlib
import argparse
import json
import time
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import torch
import features
from build_package import predict_inprocess
from candidates.c18_private_safe import PREDICT_CODE
from common import ROOT, load
from devimpute import add_cyclic, add_dev
from exp50_tabicl import regressor, numerical
from features import make_features

HOOK = '''
# Fixed research blend; only the two validation-supported imputers change.
for _col, _model in foundation.items():
    _m = _filled[_col]
    if not _m.any():
        continue
    _a = numerical(add_dev(F, artifact["dev_tables"], artifact["dev_levels"], _col).drop(columns=_col))
    _p = _model.predict(_a[_m])
    X.iloc[np.flatnonzero(_m), X.columns.get_loc(_col)] = .5*(X.loc[_m, _col].to_numpy()+_p)
'''
CODE = PREDICT_CODE.replace('try:\n    preds = np.asarray(predict_linear', HOOK+'\ntry:\n    preds = np.asarray(predict_linear')
assert HOOK in CODE


def predict(art, models, test):
    ns = {**vars(features), "np": np, "pd": pd, "artifact": art, "test": test,
          "foundation": models, "numerical": numerical}
    exec(CODE, ns)
    return np.asarray(ns["preds"])


def main(args):
    name = args.name; out = ROOT/f"submissions/{name}.csv"
    if out.exists():
        raise SystemExit("Refusing to overwrite existing candidate")
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.72)
    tr, te = load(); artpath = ROOT/f"package/{args.source}/model.pkl"; art = joblib.load(artpath)
    X = tr.drop(columns="energy_usage"); rows = pd.concat([X, te[X.columns]], ignore_index=True)
    domain = np.r_[np.zeros(len(X)), np.ones(len(te))]
    F0 = add_cyclic(make_features(rows).assign(dom=domain)); models = {}; checkpoint = None
    for col in ["occupancy", "previous_usage"]:
        A = numerical(add_dev(F0, art["dev_tables"], art["dev_levels"], col).drop(columns=col))
        ok = F0[col].notna().to_numpy(); t0 = time.time()
        models[col] = regressor().fit(A[ok], F0.loc[ok, col].to_numpy())
        checkpoint = models[col].model_path_
        print(f"{col} context: {ok.sum()} observed feature rows, {time.time()-t0:.2f}s", flush=True)
    p = predict(art, models, te); base = predict_inprocess(art, PREDICT_CODE, te)
    perm = np.random.default_rng(123).permutation(len(te))
    shuffled = predict(art, models, te.iloc[perm].drop(columns="id").reset_index(drop=True))
    order_delta = float(np.max(np.abs(p[perm]-shuffled)))
    chunks = np.concatenate([predict(art, models, t) for t in np.array_split(te, 4)])
    chunk_delta = float(np.max(np.abs(p-chunks)))
    assert np.isfinite(p).all() and len(p) == 3000
    assert order_delta < .002 and chunk_delta < .002, (order_delta, chunk_delta)
    affected = te[["occupancy", "previous_usage"]].isna().any(axis=1).to_numpy()
    assert np.max(np.abs(p[~affected]-base[~affected])) < 1e-10
    sample = pd.read_csv(ROOT/"sample_submission.csv")
    assert np.array_equal(sample.id.to_numpy(), te.id.to_numpy())
    pd.DataFrame({"id": te.id, "prediction": p}).to_csv(out, index=False)
    meta = dict(candidate=name, source_candidate=args.source, blend=.5,
                features=list(models), rows=len(p), changed_rows=int(affected.sum()),
                source_artifact_sha256=hashlib.sha256(artpath.read_bytes()).hexdigest(),
                csv_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                tabicl=version("tabicl"), torch=torch.__version__,
                checkpoint_sha256=hashlib.sha256(open(checkpoint, "rb").read()).hexdigest(),
                shuffled_idless_max_delta=order_delta, chunked_max_delta=chunk_delta,
                rms_change=float(np.sqrt(np.mean((p-base)**2))),
                changed_rms=float(np.sqrt(np.mean((p[affected]-base[affected])**2))),
                max_change=float(np.max(np.abs(p-base))),
                validation="5-fold feature holdout; feature selection exploratory; fixed 0.5 blend",
                private_package_compatible=False,
                package_limitation="Platform image does not include Torch or TabICL; CSV is public experimental only")
    (ROOT/f"experiments/{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--name",default="c20_tabicl_gap_blend")
    parser.add_argument("--source",default="c19_occ_energy_blend")
    main(parser.parse_args())
