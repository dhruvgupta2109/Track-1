"""Per-building linear model; one Ridge per missingness pattern (CV l01 a=3: 3.111 clean / 3.340 masked)."""
import numpy as np
from linmodels import fit_pattern_ridges

def fit_artifact(train):
    models = fit_pattern_ridges(train.drop(columns="energy_usage"), train["energy_usage"].to_numpy(), alpha=3.0)
    return {"ridges": models, "features": ["hour"]}

PREDICT_CODE = '''# one ridge per missingness pattern: each row uses the model fitted without its missing columns
miss = test[NUMERIC].isna().to_numpy()
keys = [tuple(sorted(c for c, mm in zip(NUMERIC, row) if mm)) for row in miss]
preds = np.empty(len(test))
for k in set(keys):
    idx = np.array([kk == k for kk in keys])
    preds[idx] = artifact["ridges"][k].predict(linear_design(test[idx], drop=set(k)))'''
