# MLDA Datathon 2026 - Track 1: Smart Campus energy forecasting

## Final package: c18_private_safe

**c18 preserves c17's statistical model and supplied-test predictions, and improves private-input handling.**
Its CSV is byte-identical to c17, whose recorded public RMSE is **3.18775**. c18 has no separate public or private score.
The best observed public result remains **c16: 3.18618**. We retain c17's train-estimated correction rather than choosing
offset sizes from public results; the 0.00157 public difference does not establish a private-test advantage.

The pipeline:

1. Clean inputs and recover a known building's type when absent or malformed.
2. Fill missing numeric inputs with three target-free LightGBM models per feature, fitted on train plus unlabelled
   public-test features. They use a domain flag, test rows weighted x2, five copies of correctly ordered mask
   augmentation, deviations from building/hour means, and cyclic time features.
3. Apply the structured ridge energy model: per-building intercept and occupancy/temperature slopes; shared
   previous-usage/humidity slopes; daily profiles, residential schedule steps, weekend shape, season and temperature squared.
4. Add empirical-Bayes-shrunk missing-value corrections estimated on naturally missing labelled training rows.
5. If an imputer fails, retain the other fills and use the matching linear sub-model. Linear and finite-mean fallbacks
   cover further prediction failures. These fallbacks cannot cover file-loading, dependency or platform failures.

The model extrapolates in observed occupancy and previous usage, whose extremes track energy in labelled training
rows. The complete-row energy-model CV RMSE is **2.998** (V5, paired 5-fold x4), versus 3.005 for V4. This is **not**
a full c17 private-like CV score. Missing-value correction has uncertain cross-fitted benefit (-0.078 MSE, SE 0.126).
Transferring the public feature distribution and train-estimated missingness mechanism to private data remains an assumption.

## Latest improvement checks

Two new five-fold checks held out feature values without using test energy labels:

| Imputer | Test-feature proxy MSE | Naturally incomplete slice | Decision |
|---|---:|---:|---|
| Existing c11 trio | 20.809 | 26.857 | Retain |
| Linear plus residual trees, fixed 50/50 blend | 21.257 | 27.208 | Worse |
| Energy-weighted training, fixed 50/50 blend | 20.734 | 26.704 | Uncertain gain; retain existing model |

The energy-weighted blend's paired test-proxy MSE change has a row-bootstrap 95% interval of **[-0.177, +0.027]**.
It does not justify extra model size and complexity. These proxies are not leaderboard RMSE or measurements on genuinely
missing values. No new leaderboard submissions were made. Exact errors, folds and uncertainty are retained under `experiments/`.

c18 fixes a measured input issue: removing or lowercasing `building_type` changed c17 predictions by up to 2.304 units
on the first 400 test rows. Known building IDs now recover the same predictions. All frozen c00-c17 packages are preserved.

## Deliverables

| Path | Purpose |
|---|---|
| `package/c18_private_safe/` | Final inference notebook, model, image-pinned requirements and hash manifest |
| `submissions/c18_private_safe.csv` | 3000 `id,prediction` rows; byte-identical to c17 |
| `package/c17_mnar_eb/` | Statistical-model fallback package, public 3.18775 |
| `package/c16_prev06_occ_selfmask/` | Best observed public variant; public-selected offsets |
| `notebooks/track1_analysis_presentation.ipynb` | Executed analysis, final model explanation and full raw-data rebuild |
| `reports/track1_technical_report.pdf` | Visually checked, one-page A4 report describing c18/c17 |
| `docs/track1_qa.md` | Judge Q&A with assumptions and evidence boundaries |
| `FINAL_SUBMISSION_CHECKLIST.md` | Verified local checks and remaining portal actions |
| `experiments/NOTES.md`, `leaderboard.md`, `log.jsonl` | Historical experiment and public-score record |
| `src/train_final.py` | Raw-data rebuild of energy model, imputers, conditional variances and MNAR calibration |
| `tests/test_private_safe.py` | Prediction equivalence, type recovery, row invariance and failure isolation |

## Reproduce and verify

Use Python 3.12 and the platform library versions. A working local environment is `.venv`.

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements-image.txt nbconvert ipykernel reportlab pypdf pypdfium2
.venv/Scripts/python.exe -X utf8 -B -m unittest discover -s tests -v
.venv/Scripts/python.exe -X utf8 src/build_presentation.py
.venv/Scripts/python.exe -X utf8 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/track1_analysis_presentation.ipynb
.venv/Scripts/python.exe -X utf8 src/build_report.py
```

The executed presentation rebuilt the full statistical pipeline in **40 s**, matching frozen predictions to **2.84e-14**.
The inference notebook passed five fresh-kernel scenarios and a fresh environment with `requirements-image.txt` plus
Jupyter tooling (7.7 s including kernel startup; exact CSV prediction match). The actual portal/private run remains untested.

To package a new candidate, create `src/candidates/<new_name>.py` and run `src/build_package.py <new_name>`.
The builder refuses to overwrite an existing package or prediction CSV. c18's package builder intentionally reuses c17's
frozen parameters; `train_final.py`, exercised by the presentation, provides the independent full rebuild.

Submission deadline in the supplied booklet: **4 October 2026, 11:30 AM Singapore time**. Follow the track page's
inference-only notebook instructions; it conflicts with the booklet's generic full-pipeline wording. Uploading artifacts
and confirming sandbox execution are still team actions.
