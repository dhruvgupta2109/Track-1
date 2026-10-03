"""Train-vs-test classifier on feature VALUES (missingness excluded: it is shifted by design)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features import make_features
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
A = pd.concat([tr.drop(columns="energy_usage"), te], ignore_index=True); y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
F = make_features(A)
p = np.zeros(len(A))
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(F, y):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, verbose=-1, n_jobs=6).fit(F.iloc[a], y[a])
    p[b] = m.predict_proba(F.iloc[b])[:, 1]
print("train-vs-test AUC (values only):", round(roc_auc_score(y, p), 4))
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, verbose=-1, n_jobs=6).fit(F, y)
imp = pd.Series(m.booster_.feature_importance("gain"), index=F.columns).sort_values(ascending=False); print((imp / imp.sum()).round(3).head(8).to_dict())
pt = p[:len(tr)]; w = pt / (1 - pt) * len(tr) / len(te)
w = np.clip(w / w.mean(), 0.2, 5.0); w /= w.mean()
np.save("experiments/density_w.npy", w)
sp = np.load("experiments/spike_flag.npy")
print("density weights: q01/50/99/max", np.quantile(w, [.01, .5, .99, 1]).round(2), "| mean w spike rows", w[sp].mean().round(2), "non-spike", w[~sp].mean().round(2), "| ESS", round(w.sum() ** 2 / (w ** 2).sum()))
