# Track 1 - Final submission checklist

Deadline in the supplied booklet: **4 October 2026, 11:30 AM Singapore time**. The track page allows one package
upload per 2 hours; a failed run consumes the window. No portal uploads were made in this branch.

## Final selection and CSV

- [x] Final inference package: **c18_private_safe**, preserving c17's train-estimated statistical model.
- [x] CSV: `submissions/c18_private_safe.csv`, exactly `id,prediction`, 3000 rows in sample-submission ID order.
- [x] Finite predictions; range **14.0696-131.6644**; full precision retained.
- [x] Byte-identical to c17; SHA-256 prefix **05cb6f912c**. c17's recorded public RMSE is **3.18775**.
- [x] Best recorded public remains c16 **3.18618**. c18 has no separate score; its changes affect unexpected inputs.
- [x] No c00-c17 packages or prediction CSVs modified.
- [ ] Confirm with the teammate that c18 is the package the team will upload.

## Package cards

- [ ] Card 1: upload `package/c18_private_safe/prediction_notebook.ipynb` (inference only).
- [ ] Card 2: upload `package/c18_private_safe/model.pkl` (**19.70 MB**, plain numeric parameters and LightGBM text).
- [ ] Card 3: optional requirements; all inference dependencies are in the platform image. If required, upload
  `package/c18_private_safe/requirements.txt`.
- [ ] Confirm the portal sandbox run succeeds and records the intended package.
- [x] Reads `DATATHON_INPUT_PATH` (default `test.csv`); writes `DATATHON_OUTPUT_PATH` (default `predictions.csv`).
- [x] Five fresh-kernel scenarios: shuffled input without IDs, defaults, 400-row stress, dropped column and one row.
- [x] Fresh environment: platform-pinned libraries plus Jupyter tooling; exact match, **7.7 s including startup**.
- [x] Seven regression tests: all 16 numeric missingness patterns across 12 buildings, type recovery, shuffled/chunked
  rows and duplicate indices, empty/all-missing/malformed input, isolated imputer failure and final mean fallback.
- [x] Hashes and source-artifact provenance: `package/c18_private_safe/manifest.json`.

Fallback if needed: **`package/c17_mnar_eb/`**, same supplied-test predictions. Its less capable input recovery and
coarser fallback are documented; it has not been changed.

## Report and presentation

- [x] `reports/track1_technical_report.pdf`: exactly one A4 page, rendered and visually inspected; describes c18/c17.
- [x] Team name **Cookie Squad** matches the supplied track-page PDF.
- [x] Report distinguishes complete-row CV, imputation proxies, public scores and unobserved private performance.
- [x] `notebooks/track1_analysis_presentation.ipynb`: 15 code cells executed, zero errors.
- [x] Full raw-data rebuild includes energy/imputers/conditional variances/MNAR calibration: **40 s**, maximum
  prediction difference **2.84e-14** against frozen final CSV.
- [x] `docs/track1_qa.md` and README explain the final rule, two new rejected/uncertain imputer trials and limitations.
- [ ] Upload the one-page PDF and verify the final-entry page reflects the intended files.
- [ ] Review the notebook/Q&A as a team; bring laptop, charger and HDMI adaptor if shortlisted.

## Remaining evidence limits

The actual portal run and private score are untested. Transductive imputation assumes private relationships resemble
the public feature distribution. Train-domain proxy success alone does not prove private robustness. MNAR correction
benefit is uncertain; tiny public-score gaps do not determine the private winner. Prediction fallbacks cannot prevent
missing dependencies, unreadable artifacts/input files or platform failures.
