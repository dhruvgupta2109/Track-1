"""Linear v3: smooth (Fourier K=3) hour profile per building type + shared weekend shape; sub-models
without occupancy/previous_usage get richer time design (K=6 per type, weekend shape per type).
One ridge per missingness pattern; plain-numpy artifact. CV 5x4: 3.077 clean / 3.297 masked / 3.544 shiftW."""
from linmodels import fit_linear_artifact
from specs import V3

def fit_artifact(train):
    return fit_linear_artifact(train.drop(columns="energy_usage"), train["energy_usage"].to_numpy(), alpha=3.0, spec=V3)

PREDICT_CODE = "preds = predict_linear(artifact, test)"
