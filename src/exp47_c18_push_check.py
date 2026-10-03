"""Round 15: label-free check of the teammate's c18 'push' (preds = c16 + w * (c16 - MAR baseline) on multi-missing rows).

Where the truth is visible: test rows with exactly ONE naturally missing input A; hide one more observed input B.
For B compare (out-of-fold, so the hidden value was never seen in imputer training):
  x_c16 : c11/c16-style imputer (domain flag = test, deviation + cyclic features, test-pattern mask augmentation),
          predicting with A naturally NaN and B hidden;
  x_mar : missing-at-random pattern imputer (observed inputs only = without A and B, no domain flag, cyclic time).
c18 assumes truth - x_c16 = w * (x_c16 - x_mar) with w ~ 1. Energy-weighted regression slope of
rho = s_B (x_true - x_c16) on delta = s_B (x_c16 - x_mar) estimates w for the 'informative missingness of A' part.
Control: complete test rows with A hidden artificially (MCAR) -> expected slope ~0 if c16 is calibrated.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from common import load, test_mask_patterns, NUM
from features import make_features, NUMERIC
from imputed import _mask_aug, IMP_PARAMS
from devimpute import make_tables, add_dev, add_cyclic, LEVELS

tr, te = load(); P = test_mask_patterns(te); PN = P[P.any(1)][:, [NUM.index(c) for c in NUMERIC]]
X = pd.concat([tr.drop(columns="energy_usage"), te[tr.columns.drop("energy_usage")]], ignore_index=True)
dom = np.r_[np.zeros(len(tr)), np.ones(len(te))]; ntr = len(tr)
nat = X[NUMERIC].isna().to_numpy()
SL = {"previous_usage": 0.339, "occupancy": None, "temperature": None, "humidity": 0.06}
import joblib
from interpret import effects
e = effects(joblib.load("package/c11_v5_cyclic/model.pkl"))
occ_s = (e["per_100_occupants"] / 100).to_dict(); tmp_s = e["per_degC_at_28"].to_dict()
def slope(c, b): return {"previous_usage": 0.339, "humidity": 0.06, "occupancy": occ_s.get(b, 0.08), "temperature": tmp_s.get(b, 1.5)}[c]

PRM = dict(IMP_PARAMS, n_jobs=6)
rows = []
# evaluation rows: test rows with exactly one natural missing (A), and complete test rows (control, A hidden artificially)
ev_idx = np.where((dom == 1) & (nat.sum(1) <= 1))[0]
for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(ev_idx)):
    hold = ev_idx[b]; keep = np.setdiff1d(np.arange(len(X)), hold)
    F0 = add_cyclic(make_features(X.iloc[keep]).assign(dom=dom[keep])); tabs = make_tables(F0, LEVELS)
    w0 = np.where(dom[keep] == 1, 2.0, 1.0)
    H = X.iloc[hold].reset_index(drop=True); Hn = nat[hold]
    G0 = add_cyclic(make_features(H).assign(dom=1.0))
    FM = add_cyclic(make_features(X.iloc[keep]))  # for MAR imputers (no dom)
    c16m = {}
    for Bc in NUMERIC:  # c16-style imputer for B
        F = _mask_aug(F0, Bc, PN, 2, seed=11); w = np.tile(w0, 3); ok = F[Bc].notna().to_numpy()
        c16m[Bc] = lgb.LGBMRegressor(**PRM).fit(add_dev(F, tabs, LEVELS, Bc).drop(columns=[Bc])[ok], F.loc[ok, Bc], sample_weight=w[ok])
    for Ac in NUMERIC:
        for Bc in NUMERIC:
            if Ac == Bc: continue
            # rows: A naturally missing & B observed  (natural)  OR  complete rows (control: A hidden artificially)
            j = np.where((Hn[:, NUMERIC.index(Ac)] & ~Hn[:, NUMERIC.index(Bc)]) | (Hn.sum(1) == 0))[0]
            if len(j) == 0: continue
            G = G0.iloc[j].copy(); G[Ac] = np.nan; G[Bc] = np.nan
            x16 = c16m[Bc].predict(add_dev(G, tabs, LEVELS, Bc).drop(columns=[Bc]))
            cols = [c for c in FM.columns if c not in (Ac, Bc)]
            okm = FM[Bc].notna().to_numpy()
            mm = lgb.LGBMRegressor(**PRM).fit(FM.loc[okm, cols], FM.loc[okm, Bc])
            xm = mm.predict(G[cols])
            for jj, p16, pm in zip(j, x16, xm):
                rows.append(dict(i=hold[jj], A=Ac, B=Bc, natural=bool(Hn[jj].sum() == 1), b=H.building_id.iloc[jj],
                                 hour=H.hour.iloc[jj], x_true=H[Bc].iloc[jj], x16=p16, xmar=pm))
    print("fold", k, flush=True)
D = pd.DataFrame(rows)
D["s"] = [slope(c, b) for c, b in zip(D.B, D.b)]
D["delta"] = D.s * (D.x16 - D.xmar); D["rho"] = D.s * (D.x_true - D.x16)
D.to_csv("experiments/c18_push_check.csv", index=False)
rng = np.random.default_rng(0)
def fit(d):
    w = float(np.sum(d.delta * d.rho) / np.sum(d.delta ** 2))
    bs = [];
    for _ in range(300):
        q = d.iloc[rng.integers(0, len(d), len(d))]; bs.append(np.sum(q.delta * q.rho) / np.sum(q.delta ** 2))
    gain1 = np.mean(d.rho ** 2) - np.mean((d.rho - 1.0 * d.delta) ** 2)
    return w, np.std(bs), gain1
print("\nslope w_hat of (truth - c16) on (c16 - MAR), energy units; gain = MSE reduction from c18's w=1 push")
for lab, d in [("NATURAL: A naturally missing, B hidden", D[D.natural]), ("CONTROL: complete rows, A and B hidden (MCAR)", D[~D.natural])]:
    w, se, g = fit(d); print(f"  {lab:48s} n={len(d):5d}  w_hat {w:+.3f} (se {se:.3f})  w=1 push MSE change {-g:+.3f}  rms delta {np.sqrt(np.mean(d.delta**2)):.3f}")
    for Bc in NUMERIC:
        q = d[d.B == Bc]
        if len(q) > 30:
            w, se, g = fit(q); print(f"      B={Bc:15s} n={len(q):5d} w_hat {w:+.3f} (se {se:.3f})  w=1 MSE change {-g:+.3f}")
