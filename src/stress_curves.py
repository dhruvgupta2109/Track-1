"""Response curves of packaged models as one feature sweeps beyond train range (others fixed / missing)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, joblib, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import features
from build_package import predict_inprocess
import importlib
names = sys.argv[1:]
arts = {n: (joblib.load(f"package/{n}/model.pkl"), importlib.import_module(f"candidates.{n}").PREDICT_CODE) for n in names}
tr = pd.read_csv("train.csv")
base = dict(building_id="LIB_A", building_type="Library", hour=14, day_of_week="Tuesday", month=4,
            temperature=29.0, humidity=76.0, occupancy=180.0, previous_usage=58.0)
sweeps = {"previous_usage": np.linspace(0, 300, 61), "occupancy": np.linspace(0, 700, 71), "temperature": np.linspace(18, 42, 49)}
scen = {"others observed": {}, "occupancy missing": {"occupancy": np.nan}, "previous_usage missing": {"previous_usage": np.nan},
        "occ+prev missing": {"occupancy": np.nan, "previous_usage": np.nan}}
fig, axes = plt.subplots(len(scen), 3, figsize=(15, 13)); report = []
for i, (sn, over) in enumerate(scen.items()):
    for j, (feat, grid) in enumerate(sweeps.items()):
        ax = axes[i, j]
        if feat in over: ax.set_visible(False); continue
        df = pd.DataFrame([{**base, **over, feat: v} for v in grid])
        for n, (art, code) in arts.items():
            p = predict_inprocess(art, code, df)
            ax.plot(grid, p, label=n)
            dp = np.diff(p); jump = np.abs(np.diff(dp)).max() if len(dp) > 1 else 0
            report.append((sn, feat, n, p.min().round(1), p.max().round(1), "monotone+" if (dp >= -1e-9).all() else ("monotone-" if (dp <= 1e-9).all() else "non-monotone"), round(jump, 3)))
        lo, hi = tr[feat].min(), tr[feat].max(); ax.axvspan(lo, hi, color="grey", alpha=0.12)
        ax.set_title(f"{feat} | {sn}", fontsize=9); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("reports/figures/stress_curves.png", dpi=90)
print(pd.DataFrame(report, columns=["scenario", "feature", "model", "min", "max", "shape", "max_slope_change"]).to_string())
