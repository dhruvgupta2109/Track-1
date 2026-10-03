"""Run a package's notebook in the image-only venv (scratch) and compare with submissions/<name>.csv."""
import sys, shutil, tempfile, time, nbformat, nbclient, pandas as pd, numpy as np
from pathlib import Path
name, root = sys.argv[1], Path(sys.argv[2])
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    for f in ["model.pkl", "prediction_notebook.ipynb"]: shutil.copy(root / "package" / name / f, d / f)
    shutil.copy(root / "test.csv", d / "test.csv")
    nb = nbformat.read(d / "prediction_notebook.ipynb", 4); t = time.time()
    nbclient.NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(d)}}).execute()
    o = pd.read_csv(d / "predictions.csv"); ref = pd.read_csv(root / "submissions" / f"{name}.csv")
    import numpy, pandas, sklearn
    print(f"{name}: env numpy {numpy.__version__} pandas {pandas.__version__} | {time.time()-t:.1f}s | shape {o.shape} cols {list(o.columns)} "
          f"| ids match {(o.id == ref.id).all()} | max diff {np.abs(o.prediction - ref.prediction).max():.2e} | finite {np.isfinite(o.prediction).all()}")
