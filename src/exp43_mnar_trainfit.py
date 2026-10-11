"""Round 13: missing-not-at-random (self-masking) correction estimated from TRAIN ONLY - no leaderboard input.

Rows: train rows with a genuinely (naturally) missing feature. Prediction = frozen c11 imputers (target-free, domain
flag 0 = train) + V5 energy model refitted out-of-fold (5-fold x 4 repeats), so the residual r = y - yhat is honest.

Self-masking model (NeuMiss / pattern-mixture delta adjustment): when feature c is missing, the true value sits
kappa_c conditional-SDs above its imputation, so  r_i = kappa_c * g_i + e_i,  g_i = slope_{c,b} * SD^train_{c,b}.
kappa_c by weighted least squares (Var e_i = s2 + g_i^2), then empirical-Bayes shrinkage of the 4 kappas toward 0
(skeptical prior kappa_c ~ N(0, tau^2), tau^2 by method of moments). Validation: cross-fitting over missing rows.
"""
import sys; sys.path.insert(0, "src")
import json, joblib, numpy as np, pandas as pd, lightgbm as lgb
from common import load, folds, ROOT
from features import make_features, predict_linear, NUMERIC, BUILDINGS, BUILDING_TYPE, clean_input, linear_design
from linmodels import fit_linear_artifact
from devimpute import add_dev, add_cyclic
from specs import V5

tr, te = load(); y = tr.energy_usage.to_numpy(); X = tr.drop(columns="energy_usage")
ART = joblib.load(ROOT / "package" / "c11_v5_cyclic" / "model.pkl")

def impute(df, art, dom):
    F = make_features(df); F["dom"] = dom
    if art["cyclic"]: F = add_cyclic(F)
    out = df.copy()
    for col in art["impute_cols"]:
        miss = F[col].isna().to_numpy()
        if miss.any():
            Xm = add_dev(F, art["dev_tables"], art["dev_levels"], col).drop(columns=[col])[miss]
            out[col] = F[col].to_numpy()
            out.loc[out.index[miss], col] = np.mean([lgb.Booster(model_str=s).predict(Xm) for s in art["imputers"][col]], axis=0)
    return out

XI = impute(X, ART, 0.0)
M = X[NUMERIC].isna().to_numpy(); anym = M.any(1)
oof = np.zeros((4, len(X)))
for r, k, a, b in folds(len(X), 5, 4, seed=0):
    lin = fit_linear_artifact(X.iloc[a], y[a], alpha=3.0, spec=V5)
    oof[r, b] = predict_linear(lin, XI.iloc[b])
pred = oof.mean(0); res = y - pred
rm = lambda e: float(np.sqrt(np.mean(e ** 2)))
print(f"OOF RMSE complete rows {rm(res[~anym]):.4f} (n={int((~anym).sum())}) | missing rows {rm(res[anym]):.4f} (n={int(anym.sum())})")

# slopes d energy / d feature per building from the full-data V5 model (finite differences at the imputed value)
lin_full = ART["linear"]
def slope(c, rows):
    up = rows.copy(); dn = rows.copy(); h = {"occupancy": 5.0, "previous_usage": 1.0, "temperature": 0.5, "humidity": 1.0}[c]
    up[c] = up[c] + h; dn[c] = dn[c] - h
    return (predict_linear(lin_full, up) - predict_linear(lin_full, dn)) / (2 * h)
err = pd.read_csv(ROOT / "experiments" / "imputation_oof_errors.csv")
SD = {d: err[err.domain == d].groupby("building_id")[NUMERIC].agg(lambda s: np.sqrt(np.nanmean(s ** 2))) for d in (0, 1)}

rows = []
for i in np.where(anym)[0]:
    for c in np.array(NUMERIC)[M[i]]:
        rows.append(dict(i=i, c=c, n_miss=int(M[i].sum()), b=X.building_id.iloc[i]))
D = pd.DataFrame(rows)
D["slope"] = [slope(c, XI.iloc[[i]])[0] for i, c in zip(D.i, D.c)]
D["g"] = D.slope * np.array([SD[0].loc[b, c] for b, c in zip(D.b, D.c)])
D["r"] = res[D.i]
S = D[D.n_miss == 1].copy()
S2 = float(np.mean(res[~anym] ** 2))  # irreducible noise variance

def fit_kappa(S):
    out = {}
    for c in NUMERIC:
        s = S[S.c == c]; w = 1.0 / (S2 + s.g ** 2)
        k = float((w * s.g * s.r).sum() / (w * s.g ** 2).sum()); se = float(np.sqrt(1.0 / (w * s.g ** 2).sum()))
        out[c] = (k, se, len(s))
    return out

GRID = np.linspace(0, 1.0, 2001)

def shrink(K, mode="exch_reml"):
    """Empirical-Bayes shrinkage of the per-feature kappas. zero_ml: kappa_c ~ N(0, tau2) (skeptical);
    exch_reml: kappa_c ~ N(mu, tau2) - one shared sensor-dropout mechanism (random-effects meta-analysis, REML)."""
    k = np.array([K[c][0] for c in NUMERIC]); v0 = np.array([K[c][1] for c in NUMERIC]) ** 2
    if mode == "none": return dict(zip(NUMERIC, 0 * k)), 0.0, 0.0
    if mode == "raw": return dict(zip(NUMERIC, k)), np.inf, 0.0
    if mode == "prev_raw": return dict(zip(NUMERIC, np.where(np.array(NUMERIC) == "previous_usage", k, 0))), np.inf, 0.0
    best = None
    for t2 in GRID:
        v = v0 + t2; w = 1 / v
        if mode == "zero_ml":
            mu = 0.0; ll = -0.5 * np.sum(np.log(v)) - 0.5 * np.sum(k ** 2 / v)
        else:
            mu = np.sum(w * k) / np.sum(w); ll = -0.5 * np.sum(np.log(v)) - 0.5 * np.log(np.sum(w)) - 0.5 * np.sum(w * (k - mu) ** 2)
        if best is None or ll > best[0]: best = (ll, t2, mu)
    _, t2, mu = best
    post = mu + (k - mu) * t2 / (t2 + v0) if t2 > 0 else mu + 0 * k
    return dict(zip(NUMERIC, post)), t2, mu

K = fit_kappa(S)
print(); print("kappa (shift in conditional-SD units), single-missing natural train rows:")
MODES = ["none", "prev_raw", "raw", "zero_ml", "exch_reml"]
posts = {m: shrink(K, m) for m in MODES}
for c in NUMERIC:
    print(f"  {c:15s} kappa {K[c][0]:+.3f} (se {K[c][1]:.3f}, n={K[c][2]:3d}) | " + "  ".join(f"{m} {posts[m][0][c]:+.3f}" for m in MODES[3:]))
for m in MODES[3:]: print(f"  {m}: tau2 {posts[m][1]:.4f}, mu {posts[m][2]:+.3f}")

# cross-fitted validation on train: estimate on 4/5 of the missing rows, apply to the held-out 1/5 (50 random splits)
print(); print("cross-fitted MSE change vs no correction on", len(S), "single-missing natural train rows:")
CV = {}
for m in MODES[1:]:
    rng = np.random.default_rng(0); dm = []; dd = []
    for rep in range(50):
        fid = rng.integers(0, 5, len(S)); d = np.zeros(len(S))
        for f in range(5):
            pf = shrink(fit_kappa(S[fid != f]), m)[0]; h = fid == f
            adj = np.array([pf[c] for c in S.c[h]]) * S.g[h].to_numpy(); r = S.r[h].to_numpy()
            d[h] = (r - adj) ** 2 - r ** 2
        dm.append(d.mean()); dd.append(d)
    dd = np.mean(dd, 0); CV[m] = float(np.mean(dm))
    print(f"  {m:10s} dMSE {np.mean(dm):+.3f} (paired se {dd.std(ddof=1) / np.sqrt(len(dd)):.3f}; split range {min(dm):+.3f}..{max(dm):+.3f})")

# test-side offsets: kappa x slope(at imputed test value) x TEST conditional SD, summed over missing features
TI = impute(te[X.columns], ART, 1.0); MT = te[NUMERIC].isna().to_numpy()
G = {}
for j, c in enumerate(NUMERIC):
    idx = np.where(MT[:, j])[0]
    G[c] = (idx, slope(c, TI.iloc[idx]) * np.array([SD[1].loc[b, c] for b in te.building_id.iloc[idx]]))
c16 = pd.read_csv(ROOT / "submissions" / "c16_prev06_occ_selfmask.csv").prediction.to_numpy()
c11 = pd.read_csv(ROOT / "submissions" / "c11_v5_cyclic.csv").prediction.to_numpy()
OFF = {}
print(); print("test offsets (mean over rows where the feature is missing):")
for m in MODES[1:]:
    off = np.zeros(len(te))
    for c in NUMERIC:
        idx, g = G[c]; off[idx] += posts[m][0][c] * g
    OFF[m] = off
    print(f"  {m:10s} " + "  ".join(f"{c[:4]} {off[MT[:, j]].mean():+.3f}" for j, c in enumerate(NUMERIC)) + f" | rms vs c16 {rm(c11 + off - c16):.4f}")
print(f"  c16 itself: " + "  ".join(f"{c[:4]} {(c16 - c11)[MT[:, j]].mean():+.3f}" for j, c in enumerate(NUMERIC)))
json.dump({"kappa_raw": {c: K[c][:2] for c in NUMERIC}, "kappa": {m: posts[m][0] for m in MODES},
           "tau2_mu": {m: posts[m][1:] for m in MODES[3:]}, "cv_mse_change": CV}, open(ROOT / "experiments" / "mnar_trainfit.json", "w"), indent=1)
np.savez(ROOT / "experiments" / "mnar_trainfit_test_offsets.npz", **OFF)
