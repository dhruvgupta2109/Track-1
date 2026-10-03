"""Build a sandbox package: model.pkl + prediction notebook + requirements.txt, then verify it.

Usage: python src/build_package.py <candidate_name>
The candidate module src/candidates/<name>.py must define fit_artifact(train_df) -> artifact,
and PREDICT_CODE (str) - the inference cell body using `artifact` and `X_test` -> `preds`.

Verification (simulating the platform): copy ONLY the package files to a fresh temp dir,
feed a shuffled test.csv with the id column dropped via DATATHON_INPUT_PATH, execute the
notebook top-to-bottom in a new kernel, and check row count / order / NaNs against the
predictions produced in-process.
"""
import importlib, inspect, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import joblib, numpy as np, pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features  # noqa: E402

REQS = """numpy==2.1.3
scipy==1.14.1
pandas==2.2.3
scikit-learn==1.5.2
joblib==1.4.2
xgboost==2.1.3
lightgbm==4.5.0
"""


def notebook(predict_code, title):
    feat_src = inspect.getsource(features)
    feat_src = feat_src.split('"""', 2)[2].strip()  # drop module docstring
    cells = [
        nbformat.v4.new_markdown_cell(
            f"# Track 1 Smart Campus - prediction notebook ({title})\n\n"
            "Inference only: loads the trained artifact `model.pkl` (trained locally, see the training "
            "notebook) and predicts `energy_usage` for every row of the input file, in input order. "
            "All features are row-local, so shuffled rows / a missing `id` column are fine."),
        nbformat.v4.new_code_cell(
            "import os\nimport warnings\nimport joblib\nimport numpy as np\nimport pandas as pd\n"
            "warnings.filterwarnings('ignore')\n\n"
            "TEST_PATH = os.environ.get(\"DATATHON_INPUT_PATH\", \"test.csv\")\n"
            "OUTPUT_PATH = os.environ.get(\"DATATHON_OUTPUT_PATH\", \"predictions.csv\")"),
        nbformat.v4.new_code_cell("# Feature engineering - identical to training (src/features.py)\n" + feat_src),
        nbformat.v4.new_code_cell(
            "artifact = joblib.load(\"model.pkl\")  # uploaded next to this notebook\n"
            "test = pd.read_csv(TEST_PATH)\n" + predict_code + "\n"
            "preds = np.asarray(preds, dtype=float).ravel()\n"
            "assert len(preds) == len(test) and np.isfinite(preds).all()"),
        nbformat.v4.new_code_cell(
            "# platform input has no id column (template) -> 'prediction' only; with ids -> id,prediction\n"
            "out = pd.DataFrame({\"prediction\": preds})\n"
            "if \"id\" in test.columns:\n"
            "    out.insert(0, \"id\", test[\"id\"].to_numpy())\n"
            "out.to_csv(OUTPUT_PATH, index=False)\n"
            "print(f\"wrote {len(preds)} predictions to {OUTPUT_PATH}\")"),
    ]
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def predict_inprocess(artifact, predict_code, test):
    ns = {**vars(features), "np": np, "pd": pd, "artifact": artifact, "test": test}
    exec(predict_code, ns)
    return np.asarray(ns["preds"], dtype=float).ravel()


def main(name):
    cand = importlib.import_module(f"candidates.{name}")
    train = pd.read_csv(ROOT / "train.csv"); test = pd.read_csv(ROOT / "test.csv")
    pkg = ROOT / "package" / name
    pkg.mkdir(parents=True, exist_ok=True)
    artifact = cand.fit_artifact(train)
    joblib.dump(artifact, pkg / "model.pkl", compress=3)
    nbformat.write(notebook(cand.PREDICT_CODE, name), pkg / "prediction_notebook.ipynb")
    (pkg / "requirements.txt").write_text(REQS)

    # reload from disk and predict in-process -> leaderboard CSV
    art2 = joblib.load(pkg / "model.pkl")
    preds = predict_inprocess(art2, cand.PREDICT_CODE, test)
    sub = pd.DataFrame({"id": test["id"], "prediction": preds})
    (ROOT / "submissions").mkdir(exist_ok=True)
    sub.to_csv(ROOT / "submissions" / f"{name}.csv", index=False)

    verify(pkg, test, preds, name)


def stress_input(test):
    """Adversarial edge cases the private test might contain."""
    t = test.drop(columns=["id"]).sample(400, random_state=1).reset_index(drop=True)
    t = t[t.columns[::-1]]                       # column order changed
    t["extra_col"] = 1                           # unexpected column
    t.loc[0:19, "building_id"] = "NEW_X"         # unseen building, known type
    t.loc[10:19, "building_type"] = "Hospital"   # unseen building and type
    t.loc[20:39, "hour"] = np.nan
    t.loc[40:49, "hour"] = 24
    t.loc[50:69, "day_of_week"] = "mon"
    t.loc[70:79, "day_of_week"] = np.nan
    t.loc[80:89, "month"] = np.nan
    t.loc[90:99, "month"] = 13
    for k, c in enumerate(["temperature", "humidity", "occupancy", "previous_usage"]):
        t.loc[100 + 10 * k:109 + 10 * k, c] = np.nan
    t.loc[140:179, ["temperature", "humidity", "occupancy", "previous_usage"]] = np.nan  # all missing
    t.loc[180:189, "temperature"] = -999         # sentinel
    t.loc[190:199, "occupancy"] = 1e6
    t.loc[200:209, "previous_usage"] = 400       # far beyond train range (legit extrapolation)
    t.loc[210:219, "building_id"] = np.nan
    t["temperature"] = t["temperature"].astype(object); t.loc[220:229, "temperature"] = "n/a"
    return t


def run_nb(d, env_input, env_output):
    env = dict(os.environ, MPLBACKEND="Agg")
    env.pop("DATATHON_INPUT_PATH", None); env.pop("DATATHON_OUTPUT_PATH", None)
    if env_input: env["DATATHON_INPUT_PATH"] = env_input
    if env_output: env["DATATHON_OUTPUT_PATH"] = env_output
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--output", "executed.ipynb", "prediction_notebook.ipynb"],
                       cwd=d, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:]); raise SystemExit("NOTEBOOK FAILED")


def verify(pkg, test, preds, name):
    """Fresh dir with ONLY the package files; fresh kernel each run."""
    for scenario in ["platform", "template_default", "stress", "dropcol", "onerow"]:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for f in ["model.pkl", "prediction_notebook.ipynb"]:
                shutil.copy(pkg / f, d / f)
            if scenario == "platform":     # shuffled, no id column, paths via env vars
                perm = np.random.default_rng(123).permutation(len(test))
                test.iloc[perm].drop(columns=["id"]).to_csv(d / "input.csv", index=False)
                run_nb(d, str(d / "input.csv"), str(d / "out.csv"))
                out = pd.read_csv(d / "out.csv")
                assert list(out.columns) == ["prediction"] and len(out) == len(test)
                assert np.allclose(out["prediction"].to_numpy(), preds[perm], atol=1e-6), "mismatch"
            elif scenario == "template_default":   # template defaults: ./test.csv -> ./predictions.csv
                test.to_csv(d / "test.csv", index=False)
                run_nb(d, None, None)
                out = pd.read_csv(d / "predictions.csv")
                assert list(out.columns) == ["id", "prediction"] and len(out) == len(test)
                assert (out["id"].to_numpy() == test["id"].to_numpy()).all()
                assert np.allclose(out["prediction"].to_numpy(), preds, atol=1e-6)
            elif scenario in ("dropcol", "onerow"):  # a whole column absent / a single-row file
                st = test.drop(columns=["id", "previous_usage"]).head(200) if scenario == "dropcol" else test.drop(columns=["id"]).head(1)
                st.to_csv(d / "input.csv", index=False)
                run_nb(d, str(d / "input.csv"), str(d / "out.csv"))
                out = pd.read_csv(d / "out.csv")
                assert len(out) == len(st) and np.isfinite(out["prediction"]).all()
                if scenario == "onerow": assert np.allclose(out.prediction.iloc[0], preds[0], atol=1e-6)
            else:
                st = stress_input(test); st.to_csv(d / "input.csv", index=False)
                run_nb(d, str(d / "input.csv"), str(d / "out.csv"))
                out = pd.read_csv(d / "out.csv")
                assert len(out) == len(st) and np.isfinite(out["prediction"]).all()
                print(f"  stress: {len(out)} rows finite, range {out.prediction.min():.1f}..{out.prediction.max():.1f}")
            print(f"  scenario {scenario}: OK")
    print(f"OK package/{name}; submissions/{name}.csv written. pred mean {preds.mean():.3f} std {preds.std():.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
