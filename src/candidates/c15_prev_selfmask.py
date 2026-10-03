"""c15: c11 + self-masking prev offset scaled by each building's TEST conditional SD of previous_usage
(shift proportional to slope x sd), normalised to the same row-weighted mean as c12 (0.6)."""
import numpy as np, pandas as pd
from common import ROOT
from candidates._mnar import build, PREDICT_CODE, _sd_test
def fit_artifact(train):
    sd = _sd_test()["previous_usage"]; te = pd.read_csv(ROOT / "test.csv")
    w = te.loc[te.previous_usage.isna(), "building_id"].map(sd)
    return build({"previous_usage": (0.6 * sd / w.mean()).to_dict()})
