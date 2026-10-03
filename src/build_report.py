"""One-page technical report (A4 PDF) for Track 1. Numbers come from presentation.py helpers / the ledger."""
import sys
from pathlib import Path
import numpy as np
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

# ---- figure 1: spike schedule (compact)
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
fig, ax = plt.subplots(1, 2, figsize=(4.6, 1.55), sharey=True)
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
import pandas as pd
grid = np.linspace(0, 300, 61); df = pd.DataFrame([{**base, "previous_usage": v} for v in grid])
fig, a = plt.subplots(figsize=(4.6, 1.6))
a.axvspan(tr.previous_usage.min(), tr.previous_usage.max(), color=P.GRID, alpha=0.7, lw=0)
a.plot(grid, P.package_predict("c00_lgb_baseline", df), color=P.C2, label="LightGBM (c00)")
a.plot(grid, P.package_predict("c04_linear_imputed", df), color=P.C1, label="linear (c03/c04)")
a.text(tr.previous_usage.max() - 4, a.get_ylim()[0] + 4, "train range", ha="right", fontsize=6.5, color=P.INK2)
a.set_xlabel("previous_usage (other inputs fixed)"); a.set_ylabel("predicted energy"); a.legend(loc="upper left")
fig.tight_layout(pad=0.3); fig.savefig(FIG / "report_extrap.png", dpi=300); plt.close(fig)

# ---- figure 3: occupancy sensitivity by type
o = eff.groupby("type")["+100 occupants"].mean().sort_values()
fig, a = plt.subplots(figsize=(4.6, 1.45))
a.barh(o.index, o.values, color=P.C1, height=0.62)
for i, v in enumerate(o.values):
    a.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=6.5, color=P.INK2)
a.set_xlabel("modelled energy change per +100 occupants"); a.grid(axis="y", visible=False); a.tick_params(axis="y", labelsize=6.5)
fig.tight_layout(pad=0.3); fig.savefig(FIG / "report_occ.png", dpi=300); plt.close(fig)

# ---- document
H = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=10.2, leading=12.2, spaceBefore=5, spaceAfter=1.5, textColor=colors.HexColor("#1d4f91"))
B = ParagraphStyle("b", fontName="Helvetica", fontSize=9.0, leading=11.2, spaceAfter=2.5, alignment=4)
S = ParagraphStyle("s", parent=B, fontSize=7.2, leading=8.6, textColor=colors.HexColor("#52514e"), alignment=0)
T = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=13.5, leading=16)
ST = ParagraphStyle("st", fontName="Helvetica", fontSize=8.4, leading=10.5, textColor=colors.HexColor("#52514e"))

c03, c04, c00 = res.loc["c03 linear"], res.loc["c04 linear + imputation"], res.loc["c00 LightGBM (5x2)"]
sp_tr, sp_te = spk.loc["any spike (|z|>3)", "train"], spk.loc["any spike (|z|>3)", "test"]
r_prev = stats.loc["previous_usage"].iloc[1]
s_full, s_noocc = slopes.iloc[0, 0], slopes.iloc[1, 0]
occ = eff.groupby("type")["+100 occupants"].mean()
base_sci = eff[eff.type == "Science"].iloc[:, -1]; base_oth = eff[eff.type != "Science"].iloc[:, -1]

story = []
story += [Paragraph("Track 1 · Smart Campus: forecasting hourly building energy with an extrapolating linear model and learned feature imputation", T),
          Paragraph(f"Team Cookie Squad · Final model c04 · Public RMSE <b>{c04['public LB']:.5f}</b> (c03 {c03['public LB']:.5f}) · "
                    f"CV RMSE {c04['clean']:.3f} clean / {c04['test-like']:.3f} test-like · R² {P.ledger()["final_cmp_C04_5x4"]["clean"]["r2"]:.3f}", ST), FrameBreak()]

L = []
L += [Paragraph("1 · Problem", H), Paragraph(
    "Predict <i>energy_usage</i> for 12 buildings (8 types) from building, hour / weekday / month, temperature, humidity, "
    "occupancy and <i>previous_usage</i>; metric RMSE. Test rows arrive shuffled without ids, so all features are row-local. "
    "<i>previous_usage</i> is not the previous hour: its daily profile matches energy at the <b>same</b> hour "
    "(residential corr. 0.97 vs 0.52 one hour back) and rows never chain, so it acts as a same-slot reference from an earlier period.", B)]
L += [Paragraph("2 · Critical insight: test is shifted, and the extremes are real", H), Paragraph(
    f"Test has 3× train's missing rate (18.2% vs 5.8% of rows) and, using a target-free consistency model (each feature "
    f"predicted from the others), <b>{sp_te:.1%}</b> spike rows (|z|>3) vs <b>{sp_tr:.1%}</b> in train. The spikes follow a schedule "
    f"(prev 22:00–04:00, occupancy 07:00–09:00). They are not errors: on train spike rows the energy left unexplained "
    f"without the feature tracks its deviation (r = {r_prev:.2f}) along the same line as ordinary rows. "
    "So the model must <b>extrapolate</b>, and trees, which saturate beyond the training range, are the wrong tool for the energy equation.", B),
      Image(str(FIG / "report_spikes.png"), width=9.1 * cm, height=9.1 * cm * 1.55 / 4.6)]
L += [Paragraph("3 · Validation design", H), Paragraph(
    "Paired 5-fold × 4-repeat CV, scored three ways: <b>clean</b>; <b>test-like</b> (validation rows receive missing-value masks "
    "sampled from the test file's joint missing patterns); <b>shift-weighted</b> (test-like, with spike rows up-weighted ×4.05 to "
    "test frequency). A train-vs-test classifier on features (AUC 0.70) independently gave spike rows ≈3× weight. "
    "Imputers never see the target, and validation rows are excluded from imputer fitting inside each fold.", B)]
L += [Paragraph("4 · Method", H), Paragraph(
    "<b>Energy model (linear ridge, α=3):</b> per-building intercept, occupancy and temperature slopes; shared "
    "<i>previous_usage</i> (≈0.34) and humidity slopes; smooth daily profile per building type (Fourier) with ridge-shrunk "
    "hour corrections; weekend shape; annual season; temperature². <b>Missing inputs:</b> c03 used a reduced sub-model per "
    f"missing pattern. Error analysis showed this distorts correlated slopes: without occupancy, the <i>previous_usage</i> slope "
    f"inflates from {s_full:.2f} to {s_noocc:.2f} and mis-extrapolates on prev spikes. <b>c04</b> instead fills each missing input "
    "with a LightGBM imputer (features only; train + unlabelled test features; test-like mask augmentation) and always applies the "
    "<b>full</b> linear model. LightGBM estimates inputs; the energy prediction stays linear.", B)]

R = []
tab = [["RMSE", "Clean", "Test-like", "Shift-wtd", "Spike", "Public"],
       ["c00 LightGBM", f"{c00['clean']:.3f}", f"{c00['test-like']:.3f}", f"{c00['shift-weighted']:.3f}", f"{c00['spike rows']:.3f}", "–"],
       ["c03 linear", f"{c03['clean']:.3f}", f"{c03['test-like']:.3f}", f"{c03['shift-weighted']:.3f}", f"{c03['spike rows']:.3f}", f"{c03['public LB']:.5f}"],
       ["c04 final", f"{c04['clean']:.3f}", f"{c04['test-like']:.3f}", f"{c04['shift-weighted']:.3f}", f"{c04['spike rows']:.3f}", f"{c04['public LB']:.5f}"]]
tb = Table(tab, colWidths=[2.2 * cm, 1.2 * cm, 1.4 * cm, 1.4 * cm, 1.2 * cm, 1.7 * cm])
tb.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.4), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.4),
                        ("FONT", (0, 3), (-1, 3), "Helvetica-Bold", 8.4), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#52514e")), ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#e8f0fb")),
                        ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
R += [Paragraph("5 · Results", H), tb, Spacer(1, 2), Paragraph(
    f"c04 beats c03 in all 4/4 repeats on every view (test-like Δ {paired.loc['masked','mean_delta']:+.3f}, shift-weighted "
    f"{paired.loc['shiftW','mean_delta']:+.3f}, non-spike {paired.loc['nonspike_masked','mean_delta']:+.3f}; test-like MSE Δ "
    f"{ci['masked_mse_delta']:+.2f}, 95% CI [{ci['ci95'][0]:+.2f}, {ci['ci95'][1]:+.2f}]). The public score confirmed the direction "
    f"({c03['public LB']:.5f} → {c04['public LB']:.5f}); the smaller size is within public-set noise.", B),
      Image(str(FIG / "report_extrap.png"), width=9.1 * cm, height=9.1 * cm * 1.6 / 4.6)]
R += [Paragraph("6 · Robustness and error analysis", H), Paragraph(
    "Our validation indicates the remaining error is noise: LightGBM, ExtraTrees, CatBoost and an MLP trained on c04's "
    "out-of-fold residuals all reach out-of-sample R² below 0 (−0.02 to −0.004), and the best correction weight is zero. "
    "Group-specific slopes, a separate slope for spiked values, density-ratio weighting, other imputers and blends did not "
    "improve. Forensics found no exploitable sequence: <i>previous_usage</i> matches the next hour's energy at chance rate "
    "(0.5% vs 0.3–0.7% controls) and ids are randomly ordered. Imputers fitted with test features are no worse than "
    "train-only imputers on held-out train-like rows and clearly better on held-out test-like rows.", B)]
R += [Paragraph("7 · Campus insights (associations, not causal effects)", H), Paragraph(
    f"+100 occupants ≈ <b>+{occ['LectureHall']:.1f}</b> energy units in the lecture hall vs <b>+{occ['Residential']:.1f}</b> in residences, "
    f"so lecture halls are the most occupancy-driven. Science buildings carry the largest modelled base load "
    f"({base_sci.min():.0f}–{base_sci.max():.0f} vs {base_oth.min():.0f}–{base_oth.max():.0f}), consistent with fixed equipment loads "
    "(interpretation). Residences step down at 08:00 and up at 21:00. Use cases: hourly peak planning, and flagging buildings "
    "that exceed expected use.", B), Image(str(FIG / "report_occ.png"), width=9.1 * cm, height=9.1 * cm * 1.45 / 4.6)]
R += [Paragraph("8 · Limitations", H), Paragraph(
    "Spikes beyond the labelled range are extrapolated linearly; that is supported by train spikes but is an assumption "
    "beyond them. Spikes in a <i>missing</i> feature cannot be recovered by imputation. No dates exist, so no external "
    "weather or calendar join was possible. Inference: numpy / pandas / LightGBM 4.5.0 from the base image, about 3 s.", B)]

doc = BaseDocTemplate(str(OUT / "track1_technical_report.pdf"), pagesize=A4, leftMargin=1.1 * cm, rightMargin=1.1 * cm,
                      topMargin=0.9 * cm, bottomMargin=0.8 * cm, title="Track 1 Smart Campus - technical report", author="Cookie Squad")
W, Hh = A4; m = 1.1 * cm; gap = 0.55 * cm; colw = (W - 2 * m - gap) / 2; head = 1.85 * cm
frames = [Frame(m, Hh - 0.9 * cm - head, W - 2 * m, head, id="head", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
          Frame(m, 0.8 * cm, colw, Hh - 1.7 * cm - head, id="L", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
          Frame(m + colw + gap, 0.8 * cm, colw, Hh - 1.7 * cm - head, id="R", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]
doc.addPageTemplates([PageTemplate(frames=frames)])
doc.build(story + L + [FrameBreak()] + R)
from pypdf import PdfReader
print("pages:", len(PdfReader(str(OUT / "track1_technical_report.pdf")).pages))
