"""Retain the complete input contract with new c22 coefficients and imputers."""
import hashlib
import json
import unittest
import joblib
import numpy as np
import pandas as pd
import test_private_safe as base
from common import ROOT


class DistilledCompactTests(base.PrivateSafeTests):
    @classmethod
    def setUpClass(cls):
        cls.art=joblib.load(ROOT/"package/c22_distilled_compact/model.pkl")
        cls.test=pd.read_csv(ROOT/"test.csv")
        cls.reference=pd.read_csv(ROOT/"submissions/c22_distilled_compact.csv").prediction.to_numpy()

    def test_frozen_test_predictions_and_csv_are_identical(self):
        np.testing.assert_allclose(self.predict(self.test),self.reference,rtol=0,atol=1e-12)
        manifest=json.loads((ROOT/"package/c22_distilled_compact/manifest.json").read_text())
        data=(ROOT/"submissions/c22_distilled_compact.csv").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),manifest["submission_sha256"])


if __name__=="__main__":unittest.main()
