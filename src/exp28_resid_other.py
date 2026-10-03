import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from catboost import CatBoostRegressor
from evalkit import load_oof, Y, TR
oof, _ = load_oof("final_cmp_C04_5x4"); r = Y - oof.mean(0)
X = TR.drop(columns=["id", "energy_usage"]).copy()
num = ["hour", "month", "temperature", "humidity", "occupancy", "previous_usage"]; cat = ["building_id", "building_type", "day_of_week"]
X[num] = X[num].fillna(X[num].median()); X["pred"] = oof.mean(0); num.append("pred")
def cv(mk, name):
    p = np.zeros(len(r))
    for a, b in KFold(5, shuffle=True, random_state=42).split(X):
        p[b] = mk().fit(X.iloc[a], r[a]).predict(X.iloc[b])
    print(f"{name:10s} resid R2 {1-np.mean((r-p)**2)/np.var(r):+.4f}  rmse {np.sqrt(np.mean(r**2)):.4f} -> {np.sqrt(np.mean((r-p)**2)):.4f}")
cv(lambda: CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=10, verbose=0, cat_features=cat, thread_count=6, random_seed=0), "CatBoost")
pre = ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
cv(lambda: make_pipeline(pre, MLPRegressor((64, 32), alpha=1e-2, early_stopping=True, max_iter=500, random_state=0)), "MLP")
