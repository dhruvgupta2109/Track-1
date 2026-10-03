"""Generate notebooks/track1_analysis_presentation.ipynb (then execute it with nbconvert)."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbformat.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbformat.v4.new_code_cell(s.strip()))

md("""
# Track 1 · Smart Campus Analytics — forecasting hourly building energy use

**Objective.** Predict `energy_usage` for 12 campus buildings (8 types) from building, time, weather, occupancy and
`previous_usage`. Official metric: **RMSE** (lower is better).

**Final model (c04).** A structured **linear energy model**; any missing input is first filled by a
**feature-level LightGBM imputer** that never sees the target. **Public RMSE 3.33681** (c03: 3.36875).

**Executive summary.** The test set contains about four times as many extreme (but genuine) occupancy /
`previous_usage` values as the training set. On labelled training rows, energy keeps responding *linearly* to such
extremes, so a model that **extrapolates** beats tree models that **saturate** outside the training range.
Error analysis then showed that dropping a missing feature distorts the slopes of the remaining correlated features;
imputing the missing input first lets the *same full* linear model be used on every row. Out-of-fold residuals of the
final model could not be predicted by LightGBM, ExtraTrees, CatBoost or an MLP — the remaining error appears to be noise.

*All numbers below are computed in this notebook from the data, the frozen model packages, or the stored experiment
ledger (`experiments/log.jsonl`, `experiments/oof/`). The deployed inference notebook is separate:
`package/c04_linear_imputed/prediction_notebook.ipynb`.*
""")
code("""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "../src")
import numpy as np, pandas as pd
import presentation as P
P.style(); pd.set_option("display.precision", 3)
tr, te = P.load()
""")

md("""
## 1 · Problem and data
Features: **building** (`building_id`, `building_type`), **time** (`hour`, `day_of_week`, `month` — no date or year),
**weather** (`temperature`, `humidity`), **occupancy**, **`previous_usage`**. Four numeric features have missing values,
and the test set has **≈3× the missing rate** of train, with far more rows missing two features at once.
""")
code("P.overview(tr, te)")
md("""
**What is `previous_usage`?** It is *not* the previous hour. Its daily profile lines up with energy at the **same hour**
(e.g. the residential step at 21:00 appears at 21:00 in both), not one hour earlier. It behaves like a reference reading
for the same slot in an earlier period — a noisy, slightly lower copy of current usage. (Section 8 also shows that rows
do not chain into a sequence.)
""")
code("tab, fig = P.prev_semantics(tr, te); tab")

md("""
## 2 · What drives energy?
Strong daily schedules that differ by building type, plus close relationships with `previous_usage` and occupancy.
""")
code("fig = P.fig_drivers(tr)")

md("""
## 3 · The key discovery: train/test distribution shift
A **target-free** consistency model predicts each feature from the others; a row is a *spike* when a feature deviates by
|z| > 3. Test has four times train's spike rate, and larger spikes. The spikes follow a **schedule**: `previous_usage`
spikes at 22:00–23:00 and 00:00–04:00, occupancy spikes at 07:00–09:00 and never at night. *(Observation; we do not
claim a physical cause.)*
""")
code("""
zt, ze = P.feature_z(tr, te)
display(P.spike_summary(zt, ze))
fig = P.fig_spike_schedule(tr, te, zt, ze)
""")

md("""
## 4 · Are the spikes errors? No — energy follows them
For each feature we predict energy **without** it (out-of-fold), then plot the unexplained energy against the feature's
deviation. If spikes were corrupt values, the orange points would sit near zero. Instead they continue the line fitted on
**ordinary rows only**. *An extreme value is not a corrupted value.*
""")
code("stats, fig = P.spikes_are_real(tr, zt); stats")

md("""
## 5 · Why extrapolation matters
Tree models split the observed feature range, so beyond the training maximum they reuse the last leaf value.
A linear model keeps its slope. Because test holds more extreme values than train, this difference shows up directly
in the error.
""")
code("fig = P.fig_extrapolation(tr)")

md("""
## 6 · Validation design
Ordinary CV underestimates test difficulty, because test has more missing values and more spikes. So every model is
scored three ways on the **same 5-fold × 4-repeat splits** (paired comparisons):

| View | What it does |
|---|---|
| **Clean** | standard repeated K-fold |
| **Test-like** | each validation row gets a missingness mask drawn from the *test file's* joint missing-pattern distribution |
| **Shift-weighted** | test-like, with spike rows up-weighted (×4.05) to test's spike frequency |

A separate train-vs-test classifier on feature values (AUC ≈ 0.70, features only) independently gave train spike rows
about 3× the weight of ordinary rows, which supports the shift-weighting.
""")

md("""
## 7 · Model evolution and results
- **c00 — LightGBM:** strong in-distribution, but collapses on missing values and spikes.
- **c03 — structured linear model:** per-building intercept, occupancy and temperature slopes; shared `previous_usage`
  and humidity slopes; smooth daily profile per building type; weekend shape; annual season; *one reduced sub-model per
  missing-value pattern*.
- **c04:** error analysis of c03 (below) → impute missing inputs with target-free LightGBM imputers, then always use the
  **full** linear model; plus restrained hour-by-hour corrections for schedule steps (residential 08:00 / 21:00).
""")
code("""
t, paired, ci = P.results_table()
display(t.style.format("{:.3f}", na_rep="—").set_caption("RMSE (mean over CV repeats); public LB where submitted"))
display(paired[["ref", "cand", "mean_delta", "wins"]].rename(columns={"ref": "c03", "cand": "c04", "mean_delta": "Δ (c04 − c03)", "wins": "c04 wins"}).style.format({"c03": "{:.3f}", "c04": "{:.3f}", "Δ (c04 − c03)": "{:+.3f}"}))
print(f"Test-like MSE change c04 vs c03: {ci['masked_mse_delta']:+.3f}, bootstrap 95% CI [{ci['ci95'][0]:+.3f}, {ci['ci95'][1]:+.3f}]")
""")
md("""
**External confirmation.** Public RMSE improved from **3.36875 (c03)** to **3.33681 (c04)**. CV predicted the same
direction; the size (−0.032 vs −0.068 predicted) lies within about one standard error of the public-set noise, and
imputation is harder on test rows than on train rows, so a smaller transfer is expected.
""")

md("""
### c03's failure mode — and the c04 fix
When occupancy is missing, c03 falls back to a sub-model without it. `previous_usage` is correlated with occupancy, so
in that sub-model it **absorbs part of the occupancy effect** and its slope inflates. On a `previous_usage` spike the
inflated slope then extrapolates wrongly. c04 imputes occupancy instead, so `previous_usage` keeps its own slope.
""")
code("""
slopes, row, ctx = P.c03_failure_example()
display(slopes)
display(row)
ctx.round(1)
""")
md("""
Largest c03→c04 change on the test set: occupancy is missing and `previous_usage` is unusually low. c03's inflated slope
drags the prediction down to 50.0; c04 predicts 62.0, close to the 62.6 typical of that building / hour / day-type.

*Why imputation is sound here:* the energy model is linear in each input, so plugging in an estimate of the missing
input's conditional mean approximates the conditional-mean energy prediction (exact only if the conditional mean were
known; the imputer estimates it). The imputers use features only (train + unlabelled test features; validation rows
are excluded inside CV). LightGBM is used **only** to estimate missing inputs — the energy prediction itself stays linear.
""")

md("""
### Reproducibility: retrain c04 from scratch and match the frozen package
""")
code("""
import importlib, time
from build_package import predict_inprocess
c04 = importlib.import_module("candidates.c04_linear_imputed")
t0 = time.time(); art = c04.fit_artifact(tr)
fresh = predict_inprocess(art, c04.PREDICT_CODE, te)
frozen = pd.read_csv("../submissions/c04_linear_imputed.csv").prediction.to_numpy()
print(f"retrained in {time.time()-t0:.1f}s; max |fresh - frozen submission| = {np.abs(fresh - frozen).max():.2e}")
""")

md("""
## 8 · Error analysis — is anything left?
We took c04's **out-of-fold** residuals and asked whether any model can predict them out of sample.
Result: R² < 0 — every correction makes things worse, and the best correction weight is **0**.
""")
code("""
r2, shrink = P.residual_predictability()
print(f"LightGBM on c04 OOF residuals: out-of-sample R² = {r2:+.4f}")
shrink.T
""")
md("""
| Hypothesis tested (paired vs c04, test-like Δ RMSE) | Result | Conclusion |
|---|---|---|
| Residual model: LightGBM / ExtraTrees / CatBoost / MLP | R² −0.019 / −0.011 / −0.005 / −0.004 | no learnable residual signal |
| Group-specific slopes (prev, occupancy × type / building / hour band / weekend; temperature hinges) | +0.0005 to +0.004 | slopes are shared |
| Separate slope for spiked values (deviation features) | +0.0005 (CI spans 0) | spikes respond like ordinary values |
| Train/test density-ratio or spike weighting in the fit | +0.002 to +0.018 | worse |
| Alternative imputers (ExtraTrees, LGB+ET, chained) | +0.004 to +0.068 | LightGBM imputers best |
| Train-only imputers (no test features) | +0.005 | worse; no safer on train-like rows |
| c03/c04 blend, c04 + LightGBM energy blend | ≥ −0.004 test-like, worse shift-weighted & spikes | rejected |
| Log target / WLS / ridge α 1 or 10 | worse | additive, homoscedastic-enough, α=3 |

*(Values from `experiments/NOTES.md` / `log.jsonl`.)*

### Data forensics — no hidden sequence
If rows were consecutive hours, a row's target would reappear as the next hour's `previous_usage`. It does not: the
exact-match rate at +1 h equals the control offsets, and ID order carries no structure. Rows behave as independent
samples, so sequence or time-series methods have nothing to exploit.
""")
code("""
chain, idt = P.forensics(tr, te)
display(chain.style.format({"exact match rate": "{:.2%}"}))
idt.style.format("{:.1%}")
""")

md("""
## 9 · Campus insights (from the fitted c04 model)
Coefficients describe **associations** in this dataset, not proven causal effects.
""")
code("""
eff = P.campus_effects()
display(eff.drop(columns=["+10 previous_usage"]).round(1))
fig = P.fig_campus(eff, tr)
""")
md("""
- **Occupancy sensitivity varies about 3×:** each additional 100 occupants is associated with **+13.4** energy units
  in the lecture hall but only **+4.5–4.7** in residences. Lecture-hall energy is the most occupancy-driven.
- **Base load:** science buildings have the largest modelled base load (**≈39–43** at the model's empty, low-history
  03:00 reference, vs **≈16–27** elsewhere). This is *consistent with* energy-intensive fixed loads (interpretation).
- **Schedules:** residences step **down at 08:00** and **up at 21:00**. These steps were sharp enough that hour-level
  corrections improved the model.
- **Temperature:** +1 °C is associated with +1.4 to +2.4 units (largest in science buildings), a cooling-type response.
- **Spike scheduling:** extreme `previous_usage` values cluster late at night and extreme occupancy in the morning;
  their marginal relationship with energy matches ordinary values, so c04 applies the same linear response.

**How this could support operations:** hourly forecasts for peak-demand planning; flagging buildings whose metered use
exceeds the model's expectation (possible faults or waste); prioritising occupancy-driven buildings for demand-based
HVAC/lighting control; monitoring base load in science buildings.
""")

md("""
## 10 · Limitations
- Private/test spikes may exceed the labelled training range. The linear response is verified only up to the largest
  train spikes; beyond that it is a **modelling assumption** (the most defensible one given train evidence).
- Imputers cannot reconstruct an unpredictable spike in a *missing* feature, so those rows keep a higher error.
- With no date or year, no external weather or calendar data could be joined.
- The remaining residual appears unlearnable from the given features; coefficients are associations, not causal effects.

## Why c04?
✓ best repeated validation on clean, test-like, shift-weighted, spike and non-spike rows (4/4 repeat wins)
✓ robust to missing values · ✓ extrapolates instead of saturating · ✓ public RMSE 3.36875 → **3.33681**
✓ out-of-fold residuals contain no learnable signal · ✓ inference in seconds with base-image libraries

**Final model = structured linear energy model + feature-level LightGBM imputation.**
""")

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "track1_analysis_presentation.ipynb"
nbformat.write(nb, out)
print("wrote", out)
