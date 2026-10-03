# MLDA Datathon 2026 — Track 1: Smart Campus (energy forecasting)

## Final solution: c04_linear_imputed
**Public RMSE 3.33681** · CV RMSE 3.048 clean / 3.228 test-like / 3.363 shift-weighted (5-fold × 4 repeats)

1. **Feature imputation:** each missing numeric input (occupancy, previous_usage, temperature, humidity) is
   filled by a LightGBM model that predicts it from the other features. It never uses the target and is fitted on
   train plus unlabelled test features.
2. **Energy model:** a structured linear ridge model:
   - per-building intercept, occupancy and temperature slopes;
   - shared previous_usage and humidity slopes;
   - smooth daily profile per building type with shrunk hour corrections;
   - weekend shape, annual season and temperature².

Why: the test set holds about 4× more extreme (but genuine) occupancy / previous_usage values than train, and energy
responds linearly to them. Linear extrapolation beats tree saturation. Imputing first avoids the slope distortion of
per-missing-pattern sub-models (c03). Full story: `notebooks/track1_analysis_presentation.ipynb`.

## Where things are
| Path | What |
|---|---|
| `package/c04_linear_imputed/` | **Frozen submission package**: `prediction_notebook.ipynb` + `model.pkl` (+ optional `requirements.txt`) |
| `submissions/c04_linear_imputed.csv` | Final prediction CSV (`id,prediction`) |
| `notebooks/track1_analysis_presentation.ipynb` | Analysis / presentation notebook (EDA → validation → model → error analysis → insights) |
| `reports/track1_technical_report.pdf` | One-page technical report (source: `src/build_report.py`) |
| `docs/track1_qa.md` | Judge Q&A preparation |
| `FINAL_SUBMISSION_CHECKLIST.md` | Submission checklist |
| `experiments/NOTES.md`, `log.jsonl`, `leaderboard.md`, `oof/` | Experiment ledger, OOF predictions, public scores |
| `src/` | Features (`features.py`), models (`linmodels.py`, `imputed.py`), validation (`common.py`, `evalkit.py`), packaging (`build_package.py`), experiments (`exp*.py`) |

## Reproduce
```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/Scripts/python.exe -r requirements-image.txt nbconvert ipykernel catboost reportlab pypdf
.venv/Scripts/python.exe src/build_package.py c04_linear_imputed     # retrain, package, run 3 fresh-kernel checks
.venv/Scripts/python.exe src/build_presentation.py                    # regenerate the analysis notebook
.venv/Scripts/python.exe src/build_report.py                          # regenerate the one-page PDF
```
(Rebuilding c04 writes to the frozen, read-only package directory; build under a new candidate name instead.)
