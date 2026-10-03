"""Finish verification after a builder tooling failure without retraining.

Never changes model, notebook or submission. A missing manifest is written
only after disk-artifact predictions match the CSV and fresh kernels pass.
"""
import hashlib
import json
import sys
import joblib
import nbformat
import numpy as np
import pandas as pd
from common import ROOT
from build_package import predict_inprocess, verify


def main(name):
    pkg=ROOT/"package"/name;sub=ROOT/"submissions"/f"{name}.csv"
    nb=nbformat.read(pkg/"prediction_notebook.ipynb",4)
    source=nb.cells[3].source
    prefix='artifact = joblib.load("model.pkl")  # uploaded next to this notebook\ntest = pd.read_csv(TEST_PATH)\n'
    assert source.startswith(prefix)
    code=source[len(prefix):];art=joblib.load(pkg/"model.pkl");test=pd.read_csv(ROOT/"test.csv")
    p=predict_inprocess(art,code,test);csv=pd.read_csv(sub)
    assert np.array_equal(test.id.to_numpy(),csv.id.to_numpy())
    np.testing.assert_allclose(p,csv.prediction.to_numpy(),rtol=0,atol=1e-10)
    verify(pkg,test,p,name)
    manifest=dict(candidate=name,source_candidate=art.get("source_candidate"),
                  source_artifact_sha256=art.get("source_artifact_sha256"),
                  files={f:hashlib.sha256((pkg/f).read_bytes()).hexdigest() for f in ["model.pkl","prediction_notebook.ipynb","requirements.txt"]},
                  submission_sha256=hashlib.sha256(sub.read_bytes()).hexdigest(),prediction_rows=len(p),
                  fresh_kernel_scenarios=["platform","template_default","stress","dropcol","onerow"])
    dest=pkg/"manifest.json"
    if dest.exists():
        assert json.loads(dest.read_text(encoding="utf-8"))==manifest
    else:
        dest.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":main(sys.argv[1])
