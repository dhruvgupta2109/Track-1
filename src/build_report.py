"""One-page technical report (A4 PDF) for Track 1. Numbers come from presentation.py helpers / the ledger."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Image, Table, TableStyle, Spacer, FrameBreak

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import presentation as P

OUT = ROOT / "reports"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
P.style()
tr, te = P.load()
zt, ze = P.feature_z(tr, te)
spk = P.spike_summary(zt, ze)
stats, _ = P.spikes_are_real(tr, zt); plt.close("all")
res, paired, ci = P.results_table()
slopes, row, ctx = P.c03_failure_example()
eff = P.campus_effects()
pub = P.public_progression()
prox = P.proxy_table()

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
# ---- figure 1: spike schedule
fig, ax = plt.subplots(1, 2, figsize=(4.6, 1.5), sharey=True)
for a, c, t in [(ax[0], "previous_usage", "previous_usage spikes"), (ax[1], "occupancy", "occupancy spikes")]:
    rt = (zt[c] > 3).groupby(tr.hour).mean(); re_ = (ze[c] > 3).groupby(te.hour).mean()
    a.bar(rt.index - 0.22, rt.values, 0.42, color=P.C1, label="train"); a.bar(re_.index + 0.22, re_.values, 0.42, color=P.C2, label="test")
    a.set_title(t, fontweight="bold"); a.set_xticks(range(0, 24, 4)); a.set_xlabel("hour")
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax[0].set_ylabel("rows with |z|>3"); ax[0].legend(loc="upper center")
fig.tight_layout(pad=0.3); fig.savefig(FIG / "report_spikes.png", dpi=300); plt.close(fig)

# ---- figure 2: extrapolation
base = dict(building_id="LIB_A", building_type="Library", hour=14, day_of_week="Tuesday", month=4,
            temperature=29.0, humidity=76.0, occupancy=180.0, previous_usage=58.0)
grid = np.linspace(0, 300, 61); df = pd.DataFrame([{**base, "previous_usage": v} for v in grid])
fig, a = plt.subplots(figsize=(4.6, 1.45))
a.axvspan(tr.previous_usage.min(), tr.previous_usage.max(), color=P.GRID, alpha=0.7, lw=0)
a.plot(grid, P.package_predict("c00_lgb_baseline", df), color=P.C2, label="LightGBM (c00)")
a.plot(grid, P.package_predict("c09_dev_imputed", df), color=P.C1, label="linear energy model (c03-c09)")
a.text(tr.previous_usage.max() - 4, a.get_ylim()[0] + 4, "train range", ha="right", fontsize=6.5, color=P.INK2)
a.set_xlabel("previous_usage (other inputs fixed)"); a.set_ylabel("predicted energy"); a.legend(loc="upper left")
fig.tight_layout(pad=0.3); fig.savefig(FIG / "report_extrap.png", dpi=300); plt.close(fig)

# ---- figure 3: label-free proxy vs public score
fig, a = plt.subplots(figsize=(4.6, 1.45))
a.plot(prox["test-row proxy"], prox["public RMSE"], color=P.C1, lw=1.0, marker="o", ms=4.5, markeredgecolor=P.SURF, markeredgewidth=1)
for k, r in prox.iterrows():
    a.annotate(k, (r["test-row proxy"], r["public RMSE"]), textcoords="offset points", xytext=(4, 2), fontsize=6.5, color=P.INK2)
a.set_xlabel("label-free imputation proxy on held-out test rows"); a.set_ylabel("public RMSE")
fig.tight_layout(pad=0.3); fig.savefig(FIG / "report_proxy.png", dpi=300); plt.close(fig)

# ---- document
H = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9.8, leading=11.6, spaceBefore=4, spaceAfter=1.2, textColor=colors.HexColor("#1d4f91"))
B = ParagraphStyle("b", fontName="Helvetica", fontSize=8.6, leading=10.6, spaceAfter=2.2, alignment=4)
T = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=13.5, leading=16)
ST = ParagraphStyle("st", fontName="Helvetica", fontSize=8.4, leading=10.5, textColor=colors.HexColor("#52514e"))

c03, c04, c00 = res.loc["c03 linear"], res.loc["c04 linear + imputation"], res.loc["c00 LightGBM (5x2)"]
p = pub["public RMSE"]
sp_tr, sp_te = spk.loc["any spike (|z|>3)", "train"], spk.loc["any spike (|z|>3)", "test"]
r_prev = stats.loc["previous_usage"].iloc[1]
s_full, s_noocc = slopes.iloc[0, 0], slopes.iloc[1, 0]
occ = eff.groupby("type")["+100 occupants"].mean()
base_sci = eff[eff.type == "Science"].iloc[:, -1]; base_oth = eff[eff.type != "Science"].iloc[:, -1]
fw = lambda x: f"{x:.5f}"

story = [Paragraph("Track 1 Smart Campus · Linear energy model with test-aware feature imputation", T),
         Paragraph(f"Team Cookie Squad · Final model c09 · Public RMSE <b>{fw(p['c09_dev_imputed'])}</b> (rank 1; c04 {fw(p['c04_linear_imputed'])}, "
                   f"c03 {fw(p['c03_linear_v3'])}) · energy-model CV RMSE {c04['clean']:.3f} clean / {c04['test-like']:.3f} test-like", ST), FrameBreak()]

L = []
L += [Paragraph("1 · Problem", H), Paragraph(
    "Predict <i>energy_usage</i> for 12 buildings (8 types) from building, hour / weekday / month, temperature, humidity, "
    "occupancy and <i>previous_usage</i>; metric RMSE. Test rows arrive shuffled without ids, so features are row-local. "
    "<i>previous_usage</i> is not the previous hour: its daily profile matches energy at the <b>same</b> hour "
    "(residential corr. 0.97 vs 0.52 one hour back) and rows never chain, so it is a same-slot reference from an earlier period.", B)]
L += [Paragraph("2 · Critical insight: test is shifted, and the extremes are real", H), Paragraph(
    f"Test has 3× train's missing rate (18.2% vs 5.8% of rows) and, by a target-free consistency model (each feature predicted "
    f"from the others), <b>{sp_te:.1%}</b> spike rows (|z|>3) vs <b>{sp_tr:.1%}</b> in train, on a schedule (prev 22:00–04:00, "
    f"occupancy 07:00–09:00). They are not errors: on train spike rows the energy left unexplained without the feature tracks its "
    f"deviation (r = {r_prev:.2f}) along the ordinary-row line. So the energy equation must <b>extrapolate</b>; trees saturate.", B),
      Image(str(FIG / "report_spikes.png"), width=9.1 * cm, height=9.1 * cm * 1.5 / 4.6),
      Image(str(FIG / "report_extrap.png"), width=9.1 * cm, height=9.1 * cm * 1.45 / 4.6)]
L += [Paragraph("3 · Validation", H), Paragraph(
    "Energy model: paired 5-fold × 4-repeat CV scored <b>clean</b>, <b>test-like</b> (validation rows get missing-value masks "
    "sampled from the test file's joint patterns) and <b>shift-weighted</b> (spike rows ×4.05 to test frequency; a train-vs-test "
    "classifier, AUC 0.70, independently gave them ~3×). Imputers: a <b>label-free proxy</b> hides observed values in held-out "
    "public-test rows, re-imputes them and weights each error by that feature's energy slope, i.e. the energy error caused by "
    "imputation, using no targets.", B)]
L += [Paragraph("4 · Method", H), Paragraph(
    "<b>Energy model (ridge, α=3):</b> per-building intercept, occupancy and temperature slopes; shared <i>previous_usage</i> "
    "(~0.34) and humidity slopes; smooth daily profile per type with shrunk hour corrections; weekend; season; temperature². "
    f"<b>Missing inputs:</b> c03's per-pattern sub-models distort correlated slopes (without occupancy the prev slope inflates "
    f"{s_full:.2f}→{s_noocc:.2f}); <b>c04</b> imputes inputs with LightGBM and keeps the <b>full</b> model. <b>c05–c09</b> make the "
    "imputers test-aware: a train/test domain flag (predict as test), an average of 3 settings, test rows ×2, 5 copies of "
    "test-like mask augmentation, and <b>deviation features</b> (each observed input minus its building×hour and building mean), "
    "because spikes co-occur across features. LightGBM only fills inputs; the energy prediction stays linear.", B)]

R = []
tab = [["RMSE", "Clean CV", "Test-like", "Shift-wtd", "Public"],
       ["c00 LightGBM", f"{c00['clean']:.3f}", f"{c00['test-like']:.3f}", f"{c00['shift-weighted']:.3f}", "–"],
       ["c03 linear", f"{c03['clean']:.3f}", f"{c03['test-like']:.3f}", f"{c03['shift-weighted']:.3f}", fw(p["c03_linear_v3"])],
       ["c04 + imputation", f"{c04['clean']:.3f}", f"{c04['test-like']:.3f}", f"{c04['shift-weighted']:.3f}", fw(p["c04_linear_imputed"])],
       ["c05 domain flag", "", "", "", fw(p["c05_domain_imputed"])],
       ["c08 + mask aug ×5", "", "", "", fw(p["c08_domain_ens_aug5"])],
       ["c09 final", "", "", "", fw(p["c09_dev_imputed"])]]
tb = Table(tab, colWidths=[2.7 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.7 * cm])
tb.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.2), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.2),
                        ("FONT", (0, 6), (-1, 6), "Helvetica-Bold", 8.2), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#52514e")), ("BACKGROUND", (0, 6), (-1, 6), colors.HexColor("#e8f0fb")),
                        ("TOPPADDING", (0, 0), (-1, -1), 1.0), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.0)]))
R += [Paragraph("5 · Results", H), tb, Spacer(1, 2), Paragraph(
    f"c04 beats c03 in 4/4 CV repeats on every view (test-like Δ {paired.loc['masked','mean_delta']:+.3f}, shift-weighted "
    f"{paired.loc['shiftW','mean_delta']:+.3f}; MSE 95% CI [{ci['ci95'][0]:+.2f}, {ci['ci95'][1]:+.2f}]). Test-aware imputation then "
    f"cut public RMSE {fw(p['c04_linear_imputed'])} → {fw(p['c09_dev_imputed'])}. The label-free proxy ranked all six submissions in "
    "public-score order (below); its magnitude is less reliable, so final steps within public noise were not pursued.", B),
      Image(str(FIG / "report_proxy.png"), width=9.1 * cm, height=9.1 * cm * 1.45 / 4.6)]
R += [Paragraph("6 · Error analysis: what is left", H), Paragraph(
    "Rows without missing values (82% of test) are at the noise floor (~3.0): LightGBM, ExtraTrees, CatBoost and an MLP trained "
    "on out-of-fold residuals reach R² below 0, and group slopes, spike-specific slopes, density weighting and blends did not help. "
    "Forensics found no sequence, duplicates or cross-row linkage. The remaining error sits in rows with a missing value, "
    "where a spike hidden in the missing input cannot be recovered. c09 also beats c04 on train-like held-out rows (proxy 12.2 vs 13.0).", B)]
R += [Paragraph("7 · Campus insights (associations, not causal effects)", H), Paragraph(
    f"+100 occupants ~ <b>+{occ['LectureHall']:.1f}</b> energy units in the lecture hall vs <b>+{occ['Residential']:.1f}</b> in residences. "
    f"Science buildings carry the largest modelled base load ({base_sci.min():.0f}–{base_sci.max():.0f} vs {base_oth.min():.0f}–{base_oth.max():.0f}), "
    "consistent with fixed equipment loads (interpretation). Residences step down at 08:00 and up at 21:00. Uses: hourly peak "
    "planning and flagging buildings that exceed expected use.", B)]
R += [Paragraph("8 · Limitations", H), Paragraph(
    "Spikes beyond the labelled range are extrapolated linearly (supported by train spikes, assumed beyond). c05–c09 assume the "
    "private test resembles the public test; if not, c09 is still better than c04 on train-like rows. No dates, so no external "
    "weather or calendar data. Inference: numpy / pandas / LightGBM 4.5.0 from the base image, about 3 s.", B)]

doc = BaseDocTemplate(str(OUT / "track1_technical_report.pdf"), pagesize=A4, leftMargin=1.1 * cm, rightMargin=1.1 * cm,
                      topMargin=0.9 * cm, bottomMargin=0.8 * cm, title="Track 1 Smart Campus - technical report", author="Cookie Squad")
W, Hh = A4; m = 1.1 * cm; gap = 0.55 * cm; colw = (W - 2 * m - gap) / 2; head = 1.6 * cm
frames = [Frame(m, Hh - 0.9 * cm - head, W - 2 * m, head, id="head", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
          Frame(m, 0.8 * cm, colw, Hh - 1.7 * cm - head, id="L", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
          Frame(m + colw + gap, 0.8 * cm, colw, Hh - 1.7 * cm - head, id="R", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]
doc.addPageTemplates([PageTemplate(frames=frames)])
doc.build(story + L + [FrameBreak()] + R)
from pypdf import PdfReader
print("pages:", len(PdfReader(str(OUT / "track1_technical_report.pdf")).pages))
