"""Computations + figures for the Track 1 analysis/presentation notebook.

Everything here is computed from the raw data, the frozen packages, or the stored experiment
ledger (experiments/log.jsonl, experiments/oof/*.npz) - no number is typed in by hand.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_predict

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features  # noqa: E402
from features import make_features, NUMERIC, BUILDING_TYPE  # noqa: E402

# reference palette (light mode): categorical slots 1-3, text + surface tokens
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DAYS = features.DAYS


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "figure.dpi": 110, "font.family": "DejaVu Sans"})


def load():
    return pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")


def ledger():
    out = {}
    for line in open(ROOT / "experiments" / "log.jsonl"):
        r = json.loads(line); out[r["name"]] = r
    return out


# ---------------------------------------------------------------- data overview
def overview(tr, te):
    rows = [("rows", len(tr), len(te)),
            ("buildings / types", f"{tr.building_id.nunique()} / {tr.building_type.nunique()}",
             f"{te.building_id.nunique()} / {te.building_type.nunique()}")]
    for c in NUMERIC:
        rows.append((f"missing {c}", f"{tr[c].isna().mean():.1%}", f"{te[c].isna().mean():.1%}"))
    rows.append(("rows with any missing", f"{tr[NUMERIC].isna().any(axis=1).mean():.1%}",
                 f"{te[NUMERIC].isna().any(axis=1).mean():.1%}"))
    rows.append(("rows with >=2 missing", f"{(tr[NUMERIC].isna().sum(1) >= 2).mean():.1%}",
                 f"{(te[NUMERIC].isna().sum(1) >= 2).mean():.1%}"))
    return pd.DataFrame(rows, columns=["", "train", "test"]).set_index("")


def fig_drivers(tr):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    prof = tr.pivot_table(index="building_type", columns="hour", values="energy_usage")
    order = prof.mean(1).sort_values().index
    im = ax[0].imshow(prof.loc[order], aspect="auto", cmap="Blues")
    ax[0].set_yticks(range(len(order)), order); ax[0].set_xticks(range(0, 24, 3)); ax[0].grid(False)
    ax[0].set_xlabel("hour of day"); ax[0].set_title("Mean energy by type and hour")
    fig.colorbar(im, ax=ax[0], fraction=0.04, label="energy_usage")
    for a, (x, lab) in zip(ax[1:], [("previous_usage", "previous_usage"), ("occupancy", "occupancy")]):
        a.scatter(tr[x], tr.energy_usage, s=3, alpha=0.25, color=C1, linewidths=0)
        a.set_xlabel(lab); a.set_ylabel("energy_usage")
        r = tr[[x, "energy_usage"]].corr().iloc[0, 1]
        a.set_title(f"energy vs {lab}  (r = {r:.2f})")
    fig.tight_layout(); return fig


# ---------------------------------------------------------------- previous_usage semantics
def prev_semantics(tr, te):
    A = pd.concat([tr, te])
    rows = []
    for t, g in tr.groupby("building_type"):
        e = g.groupby("hour").energy_usage.mean(); p = A[A.building_type == t].groupby("hour").previous_usage.mean()
        rows.append((t, np.corrcoef(p, e)[0, 1], np.corrcoef(p, np.roll(e.values, 1))[0, 1]))
    tab = pd.DataFrame(rows, columns=["building_type", "corr: same hour", "corr: one hour back"]).set_index("building_type")
    t = "Residential"
    e = tr[tr.building_type == t].groupby("hour").energy_usage.mean()
    p = A[A.building_type == t].groupby("hour").previous_usage.mean()
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(e.index, e.values, color=C1, label="energy_usage (hour h)")
    ax.plot(p.index, p.values, color=C2, label="previous_usage (hour h)")
    ax.plot(e.index, np.roll(e.values, 1), color=INK2, lw=1.2, ls="--", label="energy_usage (hour h-1)")
    ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("hour of day"); ax.set_ylabel("mean value")
    ax.set_title("Residential: previous_usage tracks the SAME hour, not the previous hour")
    ax.legend(loc="upper left"); fig.tight_layout()
    return tab, fig


# ---------------------------------------------------------------- spikes
def feature_z(tr, te, cols=("previous_usage", "occupancy", "temperature")):
    """Target-free consistency z-scores: (x - E[x | other features]) / sd, OOF on train."""
    F, Ft = make_features(tr), make_features(te)
    zt, ze = pd.DataFrame(index=tr.index), pd.DataFrame(index=te.index)
    for c in cols:
        X = F.drop(columns=[c]); ok = F[c].notna()
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, verbose=-1, n_jobs=4)
        r = pd.Series(np.nan, index=tr.index)
        r[ok] = F[c][ok] - cross_val_predict(m, X[ok], F[c][ok], cv=KFold(5, shuffle=True, random_state=0))
        s = r.std(); m.fit(X[ok], F[c][ok])
        zt[c] = r / s; ze[c] = (Ft[c] - m.predict(Ft.drop(columns=[c]))) / s
    return zt, ze


def spike_summary(zt, ze):
    st = (zt.abs() > 3).any(axis=1); se = (ze.abs() > 3).any(axis=1)
    rows = [("any spike (|z|>3)", st.mean(), se.mean())]
    for c in zt.columns:
        rows.append((f"{c} |z|>3", (zt[c].abs() > 3).mean(), (ze[c].abs() > 3).mean()))
    rows.append(("previous_usage z 99.9th pct", np.nanquantile(zt.previous_usage, .999), np.nanquantile(ze.previous_usage, .999)))
    rows.append(("occupancy z 99.9th pct", np.nanquantile(zt.occupancy, .999), np.nanquantile(ze.occupancy, .999)))
    return pd.DataFrame(rows, columns=["", "train", "test"]).set_index("")


def fig_spike_schedule(tr, te, zt, ze):
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.4), sharey=True)
    for a, c, title in [(ax[0], "previous_usage", "previous_usage spikes (z > 3)"), (ax[1], "occupancy", "occupancy spikes (z > 3)")]:
        rt = (zt[c] > 3).groupby(tr.hour).mean(); re_ = (ze[c] > 3).groupby(te.hour).mean()
        a.bar(rt.index - 0.21, rt.values, width=0.4, color=C1, label="train")
        a.bar(re_.index + 0.21, re_.values, width=0.4, color=C2, label="test")
        a.set_xticks(range(0, 24, 2)); a.set_xlabel("hour of day"); a.set_title(title)
        a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax[0].set_ylabel("share of rows in that hour"); ax[0].legend(loc="upper center")
    fig.tight_layout(); return fig


def spikes_are_real(tr, zt):
    """Partial-response check: energy minus an OOF linear prediction made WITHOUT the feature,
    against the feature's z-score. If spikes were corrupt values, energy would not follow them."""
    from linmodels import fit_linear_artifact
    from features import predict_linear
    from specs import V3
    y = tr.energy_usage.to_numpy(); X = tr.drop(columns="energy_usage")
    out = {}
    for c in ["previous_usage", "occupancy"]:
        p = np.zeros(len(tr))
        for a, b in KFold(5, shuffle=True, random_state=0).split(X):
            art = fit_linear_artifact(X.iloc[a], y[a], 3.0, V3)
            Xb = X.iloc[b].copy(); Xb[c] = np.nan
            p[b] = predict_linear(art, Xb)
        out[c] = y - p
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
    stats = []
    for a, c in zip(ax, ["previous_usage", "occupancy"]):
        z = zt[c].to_numpy(); r = out[c]; ok = ~np.isnan(z)
        sp = ok & (np.abs(z) > 3)
        a.scatter(z[ok & ~sp], r[ok & ~sp], s=3, alpha=0.2, color=C1, linewidths=0, label="ordinary rows")
        a.scatter(z[sp], r[sp], s=14, alpha=0.85, color=C2, linewidths=0, label="spike rows (|z|>3)")
        b1, b0 = np.polyfit(z[ok & ~sp], r[ok & ~sp], 1)
        xs = np.linspace(np.nanmin(z), np.nanmax(z), 50)
        a.plot(xs, b0 + b1 * xs, color=INK, lw=1.5, label="line fitted on ordinary rows only")
        a.set_xlabel(f"{c} deviation from expectation (z, target-free)")
        a.set_ylabel("energy not explained without it")
        rr = np.corrcoef(z[sp], r[sp])[0, 1]
        a.set_title(f"{c}: spike rows follow the ordinary-row line (r = {rr:.2f})")
        a.legend(loc="upper left")
        stats.append((c, int(sp.sum()), rr))
    fig.tight_layout()
    return pd.DataFrame(stats, columns=["feature", "train spike rows", "corr(z, unexplained energy) on spike rows"]).set_index("feature"), fig


# ---------------------------------------------------------------- models
def package_predict(name, df):
    from build_package import predict_inprocess
    import importlib
    art = joblib.load(ROOT / "package" / name / "model.pkl")
    if "features" in art and "model" in art:  # c00 tree baseline: model on make_features columns
        return art["model"].predict(make_features(df)[art["features"]])
    return predict_inprocess(art, importlib.import_module(f"candidates.{name}").PREDICT_CODE, df)


def fig_extrapolation(tr):
    base = dict(building_id="LIB_A", building_type="Library", hour=14, day_of_week="Tuesday", month=4,
                temperature=29.0, humidity=76.0, occupancy=180.0, previous_usage=58.0)
    grid = np.linspace(0, 300, 61)
    df = pd.DataFrame([{**base, "previous_usage": v} for v in grid])
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for name, lab, col, ls in [("c00_lgb_baseline", "c00 LightGBM", C2, "-"), ("c04_linear_imputed", "c04 linear + imputation", C1, "-"),
                               ("c03_linear_v3", "c03 linear (identical when all inputs observed)", INK, ":")]:
        ax.plot(grid, package_predict(name, df), color=col, label=lab, lw=2.0, ls=ls)
    ax.axvspan(tr.previous_usage.min(), tr.previous_usage.max(), color=GRID, alpha=0.6, lw=0)
    ax.text(tr.previous_usage.max() - 3, ax.get_ylim()[0] + 3, "train range", ha="right", color=INK2, fontsize=8.5)
    ax.set_xlabel("previous_usage (LIB_A, Tue 14:00, other inputs fixed)"); ax.set_ylabel("predicted energy_usage")
    ax.set_title("Trees saturate beyond the training range; the linear model keeps responding")
    ax.legend(loc="upper left"); fig.tight_layout(); return fig


def results_table():
    from evalkit import paired, summary
    L = ledger()
    rows = []
    for name, lab in [("b03_lgb_slow", "c00 LightGBM (5x2)"), ("final_cmp_V3_5x4", "c03 linear"), ("final_cmp_C04_5x4", "c04 linear + imputation")]:
        s = summary(name)
        rows.append((lab, s["clean"], s["masked"], s["masked_shiftW"], s["masked_spike"], s["masked_nonspike"]))
    t = pd.DataFrame(rows, columns=["model", "clean", "test-like", "shift-weighted", "spike rows", "non-spike rows"]).set_index("model")
    pub = {"c00 LightGBM (5x2)": np.nan, "c03 linear": L["LB_c03_linear_v3"]["public_rmse"], "c04 linear + imputation": L["LB_c04_linear_imputed"]["public_rmse"]}
    t["public LB"] = [pub[i] for i in t.index]
    df, ci = paired("final_cmp_V3_5x4", "final_cmp_C04_5x4")
    return t, df, ci


def c03_failure_example():
    """Slope on previous_usage in c03's full model vs its no-occupancy sub-model, plus a real test row."""
    art = joblib.load(ROOT / "package" / "c03_linear_v3" / "model.pkl")
    from features import clean_input, linear_design
    base = dict(building_id="ENG_A", building_type="Engineering", hour=9, day_of_week="Monday", month=4,
                temperature=28.6, humidity=79.5, occupancy=100.0, previous_usage=50.0)
    def f(pattern, **kw):
        coef, icpt = art["coefs"][pattern]
        return (linear_design(clean_input(pd.DataFrame([{**base, **kw}])), drop=set(pattern), spec=art["spec"]) @ coef + icpt)[0]
    slopes = pd.DataFrame({"effective slope on previous_usage": [
        (f((), previous_usage=60.0) - f((), previous_usage=50.0)) / 10,
        (f(("occupancy",), previous_usage=60.0) - f(("occupancy",), previous_usage=50.0)) / 10]},
        index=["c03 full model (occupancy observed)", "c03 sub-model when occupancy is missing"])
    te = pd.read_csv(ROOT / "test.csv")
    a = pd.read_csv(ROOT / "submissions" / "c03_linear_v3.csv").prediction
    b = pd.read_csv(ROOT / "submissions" / "c04_linear_imputed.csv").prediction
    i = int(np.argmax(np.abs(b - a)))
    row = te.iloc[[i]].drop(columns="id").assign(c03_pred=round(a[i], 1), c04_pred=round(b[i], 1))
    tr = pd.read_csv(ROOT / "train.csv"); r = te.iloc[i]
    g = tr[(tr.building_id == r.building_id) & tr.hour.between(r.hour - 1, r.hour + 1) &
           (tr.day_of_week.isin(["Saturday", "Sunday"]) == (r.day_of_week in ["Saturday", "Sunday"]))]
    ctx = pd.Series({"train rows in that building / hour±1 / day-type": len(g), "mean energy_usage": g.energy_usage.mean(),
                     "mean previous_usage": g.previous_usage.mean(), "mean occupancy": g.occupancy.mean()})
    return slopes, row, ctx


def residual_predictability():
    """Can anything predict c04's out-of-fold residual? (2nd-level CV)"""
    from evalkit import load_oof, Y, TR
    oof, _ = load_oof("final_cmp_C04_5x4"); r = Y - oof.mean(0)
    X = make_features(TR); X["pred"] = oof.mean(0)
    p = np.zeros(len(r))
    for a, b in KFold(5, shuffle=True, random_state=42).split(X):
        p[b] = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=40, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, n_jobs=4).fit(X.iloc[a], r[a]).predict(X.iloc[b])
    shrink = [(s, np.sqrt(np.mean((r - s * p) ** 2))) for s in np.linspace(0, 1, 6)]
    return 1 - np.mean((r - p) ** 2) / np.var(r), pd.DataFrame(shrink, columns=["correction weight", "RMSE of c04 + weight x correction"]).set_index("correction weight")


def forensics(tr, te):
    A = pd.concat([tr, te], ignore_index=True); A["d"] = A.day_of_week.map(DAYS.index)
    key = {}
    for i, (b, m, d, h) in enumerate(zip(A.building_id, A.month, A.d, A.hour)):
        key.setdefault((b, m, d, h), []).append(i)
    prev = A.previous_usage.to_numpy()
    res = []
    for off in [1, 0, -1, 2, 5]:
        has = ex = 0
        for i in range(len(tr)):
            hh = A.hour[i] + off; h2, d2 = hh % 24, (A.d[i] + hh // 24) % 7
            c = [j for j in key.get((A.building_id[i], A.month[i], d2, h2), []) if j != i and not np.isnan(prev[j])]
            if c:
                has += 1; ex += np.min(np.abs(prev[c] - A.energy_usage[i])) < 0.05
        res.append((f"{off:+d} h" + (" (the chaining hypothesis)" if off == 1 else " (control)"), has, ex / has))
    chain = pd.DataFrame(res, columns=["offset", "train rows with a candidate", "exact match rate"]).set_index("offset")
    s = tr.assign(n=tr.id.str[2:].astype(int)).sort_values("n")
    share = s.building_id.value_counts(normalize=True)
    idt = pd.DataFrame([("consecutive ids, same building", (s.building_id.values[1:] == s.building_id.values[:-1]).mean(), (share ** 2).sum()),
                        ("consecutive ids, one hour apart", (((s.hour.values[1:] - s.hour.values[:-1]) % 24) == 1).mean(), 1 / 24)],
                       columns=["check", "observed", "chance"]).set_index("check")
    return chain, idt


# ---------------------------------------------------------------- campus insights
def campus_effects():
    from interpret import effects
    art = joblib.load(ROOT / "package" / "c09_dev_imputed" / "model.pkl")  # final model (same linear part as c04)
    e = effects(art)
    return e.rename(columns={"per_100_occupants": "+100 occupants", "per_degC_at_28": "+1 °C (at 28 °C)",
                             "per_10_prev": "+10 previous_usage", "weekend_noon": "weekend vs Tue (noon)",
                             "empty_baseload_3am": "base load (03:00, empty, prev=0, 26 °C)"})


def fig_campus(eff, tr):
    from interpret import hour_profile
    art = joblib.load(ROOT / "package" / "c09_dev_imputed" / "model.pkl")
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
    t = eff.groupby("type")[["+100 occupants"]].mean().sort_values("+100 occupants")
    ax[0].barh(t.index, t["+100 occupants"], color=C1, height=0.6)
    for i, v in enumerate(t["+100 occupants"]):
        ax[0].text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=8.5, color=INK2)
    ax[0].set_xlabel("modelled change in energy_usage per +100 occupants"); ax[0].set_title("Occupancy sensitivity by building type")
    ax[0].grid(axis="y", visible=False)
    for b, col in [("RES_A", C1), ("LEC_A", C2), ("SCI_A", C3)]:
        ax[1].plot(range(24), hour_profile(art, b), color=col, label=f"{b} ({BUILDING_TYPE[b]})")
    ax[1].set_xticks(range(0, 24, 2)); ax[1].set_xlabel("hour of day (weekday; occupancy=0, prev=0 reference)")
    ax[1].set_ylabel("modelled schedule component"); ax[1].set_title("Learned daily schedule (time-of-day component only)")
    ax[1].legend(loc="upper left"); fig.tight_layout(); return fig


# ---------------------------------------------------------------- test-aware imputation (c05-c09)
PUBLIC_ORDER = [("c03_linear_v3", "c03 linear, per-pattern sub-models"),
                ("c04_linear_imputed", "c04 + LightGBM imputers (train+test pooled)"),
                ("c05_domain_imputed", "c05 + domain flag (impute as test)"),
                ("c06_domain_ens", "c06 + 3-imputer average"),
                ("c07_domain_ens_testw2", "c07 + test rows x2"),
                ("c08_domain_ens_aug5", "c08 + mask augmentation x5"),
                ("c09_dev_imputed", "c09 + deviation features (FINAL)")]


def public_progression():
    L = ledger()
    rows = [(n, lab, L[f"LB_{n}"]["public_rmse"]) for n, lab in PUBLIC_ORDER]
    t = pd.DataFrame(rows, columns=["candidate", "change", "public RMSE"]).set_index("candidate")
    t["Δ vs previous"] = t["public RMSE"].diff()
    return t


def proxy_table():
    p = pd.read_csv(ROOT / "experiments" / "proxy_results.csv")
    pick = {"c04": "c04 pooled single", "c05": "c05 (LB", "c06": "c06 (LB", "c07": "c07 (LB", "c08": "c08 (LB", "c09": "c09 (LB"}
    L = ledger(); rows = []
    for k, pat in pick.items():
        r = p[p.variant.str.startswith(pat)]
        if len(r):
            r = r.iloc[-1]; name = [n for n, _ in PUBLIC_ORDER if n.startswith(k)][0]
            rows.append((k, r.test_proxy, r.natmiss, r.train_proxy, L[f"LB_{name}"]["public_rmse"]))
    return pd.DataFrame(rows, columns=["candidate", "test-row proxy", "proxy on naturally-incomplete rows", "train-row proxy", "public RMSE"]).set_index("candidate")


def fig_proxy_vs_public(t):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(t["test-row proxy"], t["public RMSE"], color=C1, lw=1.2, marker="o", ms=7, markeredgecolor=SURF, markeredgewidth=1.5)
    for k, r in t.iterrows():
        ax.annotate(k, (r["test-row proxy"], r["public RMSE"]), textcoords="offset points", xytext=(6, 4), fontsize=8.5, color=INK2)
    ax.set_xlabel("label-free proxy: slope-weighted imputation MSE on held-out TEST rows (lower = better)")
    ax.set_ylabel("public RMSE"); ax.set_title("The label-free proxy ranked every submission in public-score order")
    fig.tight_layout(); return fig
