# Submission 7 - private RMSE 3.19799

This folder contains the scored submission 7 model and its supporting files.
The user reported **3.19799 on the private leaderboard** on 2026-10-03
(Asia/Singapore), the best reported result so far. The public score remains
unknown. See [score.json](score.json) for the exact recorded precision.

## Files to submit

| File | Purpose |
|---|---|
| [submission_007.csv](submission_007.csv) | Prediction CSV, 3,000 rows in sample-submission ID order |
| [prediction_notebook.ipynb](prediction_notebook.ipynb) | Inference-only notebook |
| [model.pkl](model.pkl) | Matching trained model; keep beside the notebook |
| [requirements-image.txt](requirements-image.txt) | Pinned environment reference; all inference dependencies are in the organizer image |

The model, notebook and CSV are byte-for-byte identical to the scored files.
No model was retrained during this cleanup. The notebook needs only its model
and the platform input file; training data and source files are not needed for
inference. Locally, set `DATATHON_INPUT_PATH=data/test.csv` when running the
notebook from this folder, or put the desired input beside it as `test.csv`.

## Supporting files

| Folder | Contents |
|---|---|
| `source/` | Model, sensor reconstruction, training, validation, selection and experiment code; `reference007/` contains the reused Yash primitives |
| `data/` | Exact copies of train, test and sample-submission data |
| `validation/results/` | Cross-validation tables, experiment screens, integrity checks and original logs |
| `validation/v007/` | Frozen configuration, held-out predictions, fold splits and confirmation evidence |
| `validation/shared/` | The original development/confirmation split and shift weights |
| `docs/` | Model explanation and experiment notes, retained for later presentation preparation |
| `reference/` | The Yash c09 model, notebook and predictions used as the starting point |
| `scripts/` | Working verification and reproduction commands for this organized folder |

The original experiment and workflow code is preserved for provenance. Those
scripts retain the old workspace-relative path assumptions; use the commands
below for this folder's verification and rebuilds.

## Verify without retraining

Run from the **parent project directory**, using its existing environment:

```bash
DYLD_LIBRARY_PATH="$PWD/.runtime/lib" .venv/bin/python submission_007/scripts/verify_package.py
```

This checks the submitted predictions and executes the unchanged notebook in
an isolated process with shuffled, ID-free input. The result is saved to
`validation/packaging_verification.json`. The earlier fresh-kernel checks are
preserved in `validation/verification_summary.json`.

## Reproduce later

```bash
DYLD_LIBRARY_PATH="$PWD/.runtime/lib" .venv/bin/python submission_007/scripts/reproduce_model.py
```

Rebuilding writes to `rebuilds/<next-number>/`, including a new consecutively
numbered CSV. It preserves the scored submission 7 files. A rebuilt file has no
known leaderboard score until it is submitted and evaluated. Dependencies are
listed in `requirements-image.txt`; the environment and Mac OpenMP runtime
remain shared in the parent project's `.venv/` and `.runtime/` folders.

The canonical numbered CSVs remain in the parent project directory, and the
original frozen snapshots remain in `.submissions/`. Public and private scores
are recorded separately in the parent `submission_history.json`. No slides or
presentation deck have been created.
