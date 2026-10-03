"""Pooled per-building linear model (type-level hour profiles, shared prev/humidity slopes),
one ridge per missingness pattern, stored as plain numpy. CV l03: 3.087 clean / 3.314 masked."""
from linmodels import fit_linear_artifact

def fit_artifact(train):
    return fit_linear_artifact(train.drop(columns="energy_usage"), train["energy_usage"].to_numpy(), alpha=3.0)

PREDICT_CODE = "preds = predict_linear(artifact, test)"
