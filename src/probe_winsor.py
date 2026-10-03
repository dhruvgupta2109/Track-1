"""Diagnostic probe CSV: c05 with previous_usage / occupancy / temperature clipped to each building's
TRAIN range before prediction (no extrapolation). Tells us via public LB whether test spikes continue linearly."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, joblib, importlib
from build_package import predict_inprocess
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
art = joblib.load("package/c05_domain_imputed/model.pkl"); code = importlib.import_module("candidates.c05_domain_imputed").PREDICT_CODE
T = te.copy(); n = 0
for c in ["previous_usage", "occupancy", "temperature"]:
    lo = te.building_id.map(tr.groupby("building_id")[c].min()); hi = te.building_id.map(tr.groupby("building_id")[c].max())
    out = (T[c] < lo) | (T[c] > hi); n += out.sum(); T[c] = T[c].clip(lo, hi)
p = predict_inprocess(art, code, T); base = pd.read_csv("submissions/c05_domain_imputed.csv").prediction.values
d = p - base; ch = np.abs(d) > 1e-9
print(f"values clipped: {n}; rows changed: {ch.sum()}; mean change on those {d[ch].mean():+.2f}, rms {np.sqrt(np.mean(d[ch]**2)):.2f}, max {np.abs(d).max():.1f}")
print(f"overall rms change {np.sqrt(np.mean(d**2)):.3f}")
pd.DataFrame({"id": te.id, "prediction": p}).to_csv("submissions/probe_c05_winsor_trainrange.csv", index=False)
