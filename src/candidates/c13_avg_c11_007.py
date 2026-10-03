"""c13: average of c11 (ours) and submission 007 (teammate's model, built on c09). Both share an almost identical
energy model on complete rows (rms diff 0.035) but use different imputers (rms diff 0.40 on rows with missing values).
007's inference code runs in an isolated namespace, exactly as in its own notebook."""
import json, joblib, nbformat
from common import ROOT
from candidates._c11 import PREDICT_CODE as _C11_CODE

_nb = nbformat.read(ROOT / "external" / "submission_007" / "prediction_notebook.ipynb", 4)
_F007_SRC = _nb.cells[1].source + "\n" + _nb.cells[2].source

def fit_artifact(train):
    return {"c11": joblib.load(ROOT / "package" / "c11_v5_cyclic" / "model.pkl"),
            "f007": joblib.load(ROOT / "external" / "submission_007" / "model.pkl")}

PREDICT_CODE = ("# --- component 1: c11 (our pipeline)\n_both = artifact\nartifact = _both['c11']\n" + _C11_CODE +
                "\n_p11 = np.asarray(preds, dtype=float)\n"
                "# --- component 2: submission 007, run in its own namespace with its original inference code\n"
                "_F007_SRC = " + repr(_F007_SRC) + "\n"
                "_ns = {}\nexec(_F007_SRC, _ns)\n_p007 = np.asarray(_ns['predict_model'](_both['f007'], test), dtype=float)\n"
                "artifact = _both\n"
                "preds = 0.5 * _p11 + 0.5 * _p007  # equal-weight average of the two pipelines")
