"""Importance weights making train's feature-spike frequency match test's (no target used)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import cross_val_predict, KFold
from features import make_features
C = ["previous_usage", "occupancy", "temperature"]
def spike_flags():
    tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); F = make_features(tr); Ft = make_features(te)
    ftr = np.zeros(len(tr), bool); fte = np.zeros(len(te), bool)
    for c in C:
        X = F.drop(columns=[c]); ok = F[c].notna()
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, verbose=-1, n_jobs=4)
        r = pd.Series(np.nan, index=tr.index)
        r[ok] = F[c][ok] - cross_val_predict(m, X[ok], F[c][ok], cv=KFold(5, shuffle=True, random_state=0))
        s = r.std(); m.fit(X[ok], F[c][ok])
        rt = Ft[c] - m.predict(Ft.drop(columns=[c]))
        ftr |= (r.abs() > 3 * s).fillna(False).to_numpy(); fte |= (rt.abs() > 3 * s).fillna(False).to_numpy()
    return ftr, fte
if __name__ == "__main__":
    ftr, fte = spike_flags()
    pt, pe = ftr.mean(), fte.mean()
    w = np.where(ftr, pe / pt, (1 - pe) / (1 - pt))
    np.save("experiments/shift_w.npy", w); np.save("experiments/spike_flag.npy", ftr)
    print(f"spike rows: train {pt:.3f}, test {pe:.3f}; weight spike {pe/pt:.2f}, normal {(1-pe)/(1-pt):.3f}")
