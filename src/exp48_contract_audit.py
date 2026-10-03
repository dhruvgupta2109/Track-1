"""Audit prediction invariants for the frozen c17 package; no retraining."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_package import predict_inprocess
from candidates.c17_mnar_eb import PREDICT_CODE
from common import ROOT
from features import NUMERIC


def main():
    test = pd.read_csv(ROOT / "test.csv")
    art = joblib.load(ROOT / "package/c17_mnar_eb/model.pkl")
    p = predict_inprocess(art, PREDICT_CODE, test)
    cases = {
        "empty": test.head(0),
        "no_columns": pd.DataFrame(index=range(7)),
        "all_missing": test.head(16).assign(**{c: np.nan for c in test.columns}),
        "duplicate_index": test.head(100).set_axis(np.zeros(100, dtype=int)),
        "offset_index": test.head(100).set_axis(np.arange(100) + 500),
        "missing_type": test.head(400).drop(columns="building_type"),
        "lowercase_type": test.head(400).assign(building_type=lambda d: d.building_type.str.lower()),
    }
    out = {}
    for name, rows in cases.items():
        try:
            q = predict_inprocess(art, PREDICT_CODE, rows)
            r = dict(rows=len(q), finite=bool(np.isfinite(q).all()))
            if name in ("duplicate_index", "offset_index", "missing_type", "lowercase_type"):
                r["max_change"] = float(np.max(np.abs(q - p[:len(q)])))
            out[name] = r
        except Exception as e:
            out[name] = {"exception": repr(e)}
    # Inject one faulty imputer: full-model fallback discards other working imputers.
    bad = dict(art, imputers=dict(art["imputers"]))
    bad["imputers"]["humidity"] = ["invalid booster"]
    rows = test[test.humidity.isna()].head(20)
    q = predict_inprocess(bad, PREDICT_CODE, rows)
    out["faulty_humidity_imputer"] = {"rows": len(q), "finite": bool(np.isfinite(q).all())}
    (ROOT / "experiments/contract_audit_c17.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
