# MLDA Datathon 2026 — Track 1: Smart Campus (energy forecasting)

## Final solution: c09_dev_imputed
**Public RMSE 3.20392 (rank 1)** · energy-model CV RMSE 3.048 clean / 3.228 test-like / 3.363 shift-weighted (5-fold × 4 repeats)

1. **Test-aware feature imputation:** each missing numeric input (occupancy, previous_usage, temperature, humidity)
   is filled by the average of 3 LightGBM imputers. They never use the target. They are trained on train plus
   unlabelled test features, with:
   - a train/test domain flag, predicting "as test";
   - test rows weighted ×2;
   - 5 copies of test-like mask augmentation;
   - deviation features: each observed input minus its building×hour and building mean.
2. **Energy model:** a structured linear ridge model:
   - per-building intercept, occupancy and temperature slopes;
   - shared previous_usage and humidity slopes;
   - smooth daily profile per building type with shrunk hour corrections;
   - weekend shape, annual season and temperature².

Why: the test set holds about 4× more extreme (but genuine) occupancy / previous_usage values than train, and energy
responds linearly to them, so linear extrapolation beats tree saturation. Imputing first avoids the slope distortion of
per-missing-pattern sub-models (c03). The test-aware imputers fix c04's under-filling of test values (more spikes). They
were steered by a label-free proxy (imputation error on hidden test-row values, slope-weighted) that ranked every
submission in public-score order: 3.337 (c04) → 3.255 → 3.238 → 3.231 → 3.209 → 3.204 (c09).
Full story: `notebooks/track1_analysis_presentation.ipynb`.

## Where things are
| Path | What |
|---|---|
| `package/c09_dev_imputed/` | **Final submission package**: `prediction_notebook.ipynb` + `model.pkl` (+ optional `requirements.txt`) |
| `submissions/c09_dev_imputed.csv` | Final prediction CSV (`id,prediction`) |
| `package/c08_domain_ens_aug5/` | Fallback package (public 3.20937) |
| `notebooks/track1_analysis_presentation.ipynb` | Analysis / presentation notebook (EDA → validation → model → test-aware imputation → error analysis → insights) |
| `reports/track1_technical_report.pdf` | One-page technical report (source: `src/build_report.py`) |
| `docs/track1_qa.md` | Judge Q&A preparation |
| `FINAL_SUBMISSION_CHECKLIST.md` | Submission checklist |
| `experiments/NOTES.md`, `log.jsonl`, `leaderboard.md`, `proxy_results.csv`, `oof/` | Experiment ledger, proxy scores, OOF predictions, public scores |
| `src/` | Features (`features.py`), energy model (`linmodels.py`), imputers (`imputed.py`, `devimpute.py`), validation (`common.py`, `evalkit.py`, `proxy.py`), packaging (`build_package.py`), experiments (`exp*.py`) |

## Reproduce
```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/Scripts/python.exe -r requirements-image.txt nbconvert ipykernel catboost reportlab pypdf pypdfium2
.venv/Scripts/python.exe src/build_package.py <new_candidate_name>   # retrain, package, run 3 fresh-kernel checks
.venv/Scripts/python.exe src/build_presentation.py                   # regenerate the analysis notebook (then execute it)
.venv/Scripts/python.exe src/build_report.py                         # regenerate the one-page PDF
```
Frozen package directories are read-only; to rebuild, copy a candidate module under a new name. Set `PYTHONUTF8=1` on Windows.
