"""c10: average of the c08 and c09 pipelines (same linear energy model; imputers with and without deviation
features). Equivalent to averaging the two imputer families. Built from the frozen c08/c09 artifacts."""
import joblib, importlib
from common import ROOT

def fit_artifact(train):
    return {"a": joblib.load(ROOT / "package" / "c08_domain_ens_aug5" / "model.pkl"),
            "b": joblib.load(ROOT / "package" / "c09_dev_imputed" / "model.pkl")}

_c08 = importlib.import_module("candidates.c08_domain_ens_aug5").PREDICT_CODE
_c09 = importlib.import_module("candidates.c09_dev_imputed").PREDICT_CODE
PREDICT_CODE = ("_both = artifact\n"
                "artifact = _both['a']\n" + _c08 + "\n_pa = np.asarray(preds, dtype=float)\n"
                "artifact = _both['b']\n" + _c09 + "\n_pb = np.asarray(preds, dtype=float)\n"
                "artifact = _both\n"
                "preds = 0.5 * _pa + 0.5 * _pb  # average of c08 and c09 pipelines")
