import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
from features import make_features, NUMERIC
from shift_weights import C as SC
ref, cand = sys.argv[1], sys.argv[2]
te = pd.read_csv("test.csv"); tr = pd.read_csv("train.csv")
a = pd.read_csv(f"submissions/{ref}.csv").prediction.values; b = pd.read_csv(f"submissions/{cand}.csv").prediction.values
d = b - a
print(f"corr {np.corrcoef(a, b)[0,1]:.5f} | mean delta {d.mean():+.3f} | rms delta {np.sqrt(np.mean(d**2)):.3f} | max |d| {np.abs(d).max():.2f}")
print("abs delta quantiles 50/90/95/99/max:", np.quantile(np.abs(d), [.5, .9, .95, .99, 1]).round(2))
# test spike flags (same consistency models as shift_weights)
F = make_features(tr); Ft = make_features(te); zt = pd.DataFrame(index=te.index)
for c in SC:
    X = F.drop(columns=[c]); ok = F[c].notna()
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, verbose=-1, n_jobs=4).fit(X[ok], F[c][ok])
    s = (F[c][ok] - m.predict(X[ok])).std()
    zt[c] = (Ft[c] - m.predict(Ft.drop(columns=[c]))) / s
sp = zt.abs().max(axis=1).fillna(0)
t = te.copy(); t["d"] = d; t["absd"] = np.abs(d)
t["spike"] = np.where(sp > 5, "extreme", np.where(sp > 3, "mild", "none"))
t["missing"] = te[NUMERIC].isna().apply(lambda r: "+".join(c[:4] for c, v in r.items() if v) or "none", axis=1)
t["prev_q"] = pd.qcut(te.previous_usage, 5); t["occ_q"] = pd.qcut(te.occupancy, 5, duplicates="drop")
for by in ["spike", "missing", "building_type", "prev_q", "occ_q"]:
    g = t.groupby(by, observed=True).agg(n=("d", "size"), mean_d=("d", "mean"), rms_d=("d", lambda x: np.sqrt(np.mean(x**2))), max_abs=("absd", "max"))
    print(f"\n== {by}"); print(g.round(3))
print("\nlargest changes:"); print(t.sort_values("absd", ascending=False).head(8)[["building_id", "hour", "day_of_week", "temperature", "humidity", "occupancy", "previous_usage", "spike", "d"]].assign(ref=a[t.sort_values("absd", ascending=False).head(8).index]))
