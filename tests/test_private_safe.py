"""Meaningful invariants for the private-test wrapper around frozen c17."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features
from build_package import predict_inprocess
from candidates.c18_private_safe import PREDICT_CODE
from common import ROOT


class PrivateSafeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.art = joblib.load(ROOT / "package/c18_private_safe/model.pkl")
        cls.test = pd.read_csv(ROOT / "test.csv")
        cls.reference = pd.read_csv(ROOT / "submissions/c17_mnar_eb.csv").prediction.to_numpy()

    def predict(self, rows, art=None):
        return predict_inprocess(self.art if art is None else art, PREDICT_CODE, rows)

    def test_frozen_test_predictions_and_csv_are_identical(self):
        np.testing.assert_allclose(self.predict(self.test), self.reference, rtol=0, atol=1e-12)
        a = ROOT / "submissions/c18_private_safe.csv"
        b = ROOT / "submissions/c17_mnar_eb.csv"
        self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_recover_type_from_building_id(self):
        rows = self.test.iloc[:400]
        variants = [rows.drop(columns="building_type"),
                    rows.assign(building_type=rows.building_type.str.lower()),
                    rows.assign(building_type="incorrect")]
        for variant in variants:
            np.testing.assert_allclose(self.predict(variant), self.reference[:400], rtol=0, atol=1e-12)

    def test_row_order_chunking_and_duplicate_indices(self):
        perm = np.random.default_rng(91).permutation(400)
        rows = self.test.iloc[perm].drop(columns="id")
        rows.index = np.zeros(len(rows), dtype=int)
        full = self.predict(rows)
        chunks = np.r_[self.predict(rows.iloc[:137]), self.predict(rows.iloc[137:])]
        np.testing.assert_allclose(full, self.reference[perm], rtol=0, atol=1e-12)
        np.testing.assert_allclose(chunks, full, rtol=0, atol=1e-12)

    def test_every_building_and_missingness_pattern(self):
        train = pd.read_csv(ROOT / "train.csv")
        bases = train.dropna(subset=features.NUMERIC).groupby("building_id").head(1)
        variants = []
        for pattern in range(16):
            row = bases.copy()
            for j, col in enumerate(features.NUMERIC):
                if pattern & (1 << j):
                    row[col] = np.nan
            variants.append(row)
        rows = pd.concat(variants, ignore_index=True)
        preds = self.predict(rows)
        self.assertEqual(len(preds), 192)
        self.assertTrue(np.isfinite(preds).all())

    def test_empty_absent_and_invalid_inputs(self):
        for rows in [self.test.head(0), pd.DataFrame(index=range(7)),
                     self.test.head(8).assign(**{c: np.nan for c in self.test.columns}),
                     self.test.head(8).assign(temperature=np.inf, occupancy=-1, previous_usage="bad")]:
            preds = self.predict(rows)
            self.assertEqual(len(preds), len(rows))
            self.assertTrue(np.isfinite(preds).all())

    def test_one_broken_imputer_preserves_the_other_fills(self):
        art = copy.deepcopy(self.art)
        art["imputers"]["humidity"] = ["FAULT"]
        rows = self.test.head(8).assign(humidity=np.nan, previous_usage=np.nan, occupancy=np.nan)
        original_booster = lgb.Booster

        def load_booster(*args, **kwargs):
            if kwargs.get("model_str") == "FAULT":
                raise RuntimeError("injected humidity failure")
            return original_booster(*args, **kwargs)

        ns = {**vars(features), "np": np, "pd": pd, "artifact": art, "test": rows}
        with patch.object(lgb, "Booster", side_effect=load_booster):
            exec(PREDICT_CODE, ns)
        self.assertTrue(ns["X"][["previous_usage", "occupancy"]].notna().all().all())
        self.assertTrue(ns["X"].humidity.isna().all())
        # Failed rows use the sub-model for that feature, without invalid MNAR shifts.
        np.testing.assert_array_equal(ns["preds"], features.predict_linear(art["linear"], ns["X"]))

    def test_last_resort_mean_if_linear_models_fail(self):
        art = dict(self.art, linear={"coefs": {}, "spec": self.art["linear"]["spec"], "clip": (0, 400)})
        preds = self.predict(self.test.head(3), art)
        np.testing.assert_array_equal(preds, np.full(3, art["fallback_mean"]))


if __name__ == "__main__":
    unittest.main()
