"""Single source for the reviewed submission choice and measured provenance."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from common import ROOT

FINAL_CANDIDATE = "c18_private_safe"
STATISTICAL_CANDIDATE = "c17_mnar_eb"
BEST_PUBLIC_CANDIDATE = "c19_occ_energy_blend"


def final_summary():
    """Assert equal supplied-test predictions before inheriting c17's score."""
    package = ROOT / "package" / FINAL_CANDIDATE
    sub = ROOT / "submissions" / f"{FINAL_CANDIDATE}.csv"
    reference = ROOT / "submissions" / f"{STATISTICAL_CANDIDATE}.csv"
    if sub.read_bytes() != reference.read_bytes():
        raise ValueError("Final predictions differ from c17; its public score cannot be inherited.")
    scores = {}
    for line in (ROOT / "experiments/log.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["name"].startswith("LB_"):
            scores[r["name"][3:]] = r["public_rmse"]
    predictions = pd.read_csv(sub)
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    assert list(predictions.columns) == ["id", "prediction"]
    assert predictions.id.equals(sample.id)
    assert np.isfinite(predictions.prediction).all()
    art = joblib.load(package / "model.pkl")
    return {
        "candidate": FINAL_CANDIDATE,
        "statistical_candidate": STATISTICAL_CANDIDATE,
        "source_public_rmse": scores[STATISTICAL_CANDIDATE],
        "best_public_candidate": BEST_PUBLIC_CANDIDATE,
        "best_public_rmse": scores[BEST_PUBLIC_CANDIDATE],
        "byte_identical_source_csv": True,
        "separate_public_submission": False,
        "submission_sha256": hashlib.sha256(sub.read_bytes()).hexdigest(),
        "rows": len(predictions),
        "prediction_min": float(predictions.prediction.min()),
        "prediction_max": float(predictions.prediction.max()),
        "model_bytes": (package / "model.pkl").stat().st_size,
        "mnar_kappa": art["mnar_kappa"],
        "package": f"package/{FINAL_CANDIDATE}",
    }
