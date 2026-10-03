# Track 1 — research notes (evidence for report)

Full per-run metrics: `experiments/log.jsonl`; OOF predictions: `experiments/oof/*.npz`.
CV = 5-fold x 2 repeats, shuffled. "masked" = same OOF, but each validation row gets a missingness
mask sampled from the *test file's* empirical joint missing pattern (test has 3x train's missing rate).

## Rules (verified from files)
- Metric RMSE (leaderboard); round-1 judging 60% RMSE/MAE/R², 40% understanding. Deadline 4 Oct 11:30.
- Platform test input: **no id column, rows shuffled** (template_notebook) -> features must be row-local.
- Conflict: booklet says notebook = full pipeline; track page says prediction notebook = inference only.
  Following the track page (platform executes it). Full training/EDA kept in repo for presentation.

## Data findings
- 8000 train / 3000 test, 12 buildings / 8 types; id uncorrelated with anything -> random split, not temporal.
- Rows are independent samples (1195 duplicate building/month/dow/hour keys); previous_usage is a row
  attribute, not linkable to other rows. Copying it as the forecast: RMSE 6.72.
- Missingness: train 5.8% rows, test 18.2% rows, test has strong co-missingness (e.g. 49 rows temp+hum).
  Roughly uniform over building/hour -> MCAR-like. Target mean ~unchanged when a feature is missing.
- Covariate shift: test is wider within every building (temp std 2.38 vs 1.88; prev std 23.4 vs 18.9;
  occupancy higher). 393 test rows (13%) fall outside train's per-building 0.5–99.5% range.
- Data-generating structure is **additive-linear**: per building, y ~ 0.35*prev + b_occ*occ + b_temp*temp
  + hour/weekend profile + season. prev slope 0.33–0.37 in every building -> shared. Month effect is a
  clean annual sinusoid peaking April. Residuals near-Gaussian (kurtosis 0.16), no structure left by
  hour/dow/month; residual std grows mildly with level (2.6 ADM .. 3.75 SCI_A).

## Experiments (clean / masked RMSE)
| id | model | clean | masked | note |
|---|---|---|---|---|
| b00 | previous_usage copy | 6.716 | 8.128 | naive |
| b01 | HGB default (template) | 3.519 | 4.215 | |
| b03 | LightGBM slow lr | 3.341 | 3.964 | **c00 package (fallback)** |
| b04 | LightGBM linear_tree | 3.558 | 4.877 | collapses with missing values |
| m02 | LGB + test-like mask augmentation | 3.379 | 3.594 | augmentation: -0.37 masked |
| l01 | per-building ridge, pattern sub-models | 3.111 | 3.340 | linear beats GBM by 0.23 |
| h01-3 | ridge + LGB on residuals | 3.109 | 3.329–3.360 | no gain: residual is noise |
| l03 | **pooled design v2** (type hour-profiles, shared prev/hum slope) | **3.087** | **3.314** | **c02 candidate** |
| l05-07 | design variants | 3.087–3.092 | 3.313–3.320 | within noise |
| l08 | log target | 3.473 | 3.628 | additive, not multiplicative |
| l09 | WLS by level | 3.098 | 3.344 | no gain |
| blend | l03 + 10% m02 | 3.085 | 3.306 | <0.01, rejected (tree extrapolation risk) |

## Design decisions
- Missing numerics: instead of imputation, one ridge per missingness pattern (16), each fitted on all
  rows where its columns are observed = conditional-expectation predictor for that pattern.
- Unknown/missing categoricals: soft one-hot (expectation over known levels); impossible numerics -> NaN.
- Artifact = plain numpy coefficients (no sklearn object pickled) -> no version-deserialisation risk.

## Round 2: test feature spikes, smooth profiles (shiftW = shift-weighted masked RMSE)
- **Test feature spikes.** A target-free consistency model (predict each feature from the others) finds
  |z|>3 spikes in 16.6% of test rows vs 4.1% of train rows, and test spikes are larger (prev 99.9% z 8.6 vs 5.6).
  Medians are unchanged, so this is contamination by events, not a wider distribution overall.
- **The spikes are real.** On train spike rows the target follows the spike (corr 0.92 for prev).
  Linear model RMSE on spike rows: 3.44 using the feature vs 11.8 when ignoring it. The response is ~linear
  at the extremes (residual slope -0.07 per SD). So extrapolation is correct, and tree saturation is the failure mode.
- **Shift-weighted CV.** Spike rows are up-weighted x4.05 to test frequency (`src/shift_weights.py`).
  Under it the linear lead over LGB grows: l03 3.573 vs m02 3.912 vs b03 4.282.
- Importance-weighted fitting: sqrt-weights give -0.022 shiftW but +0.009 masked. Not adopted (fit/eval coupling).
- Fourier hour basis K=3 per type: 3.080 / 3.308 / 3.561 (vs one-hot 3.087 / 3.314 / 3.573), with fewer params.
- Shared weekend hour shape: masked 3.308 -> 3.282. The gain sits mainly on occupancy-missing rows,
  where hour x weekend has to stand in for occupancy.
- Richer time design only in sub-models lacking occupancy/prev (Fourier 6 + weekend shape per type): 3.278 / 3.516.
- Per missing pattern, linear beats LGB(aug) on 13/16 patterns; a blend helps only the occupancy-missing rows (~0.07) -> not worth it.
- **V3 confirmed** (5x4 CV, paired): clean 3.077 vs 3.087; masked 3.297 vs 3.334; shiftW 3.544 vs 3.595.
  Masked MSE gain 0.249, bootstrap 95% CI [0.13, 0.36]. Better in every repeat. -> **c03 candidate**.
- Oracle reasoning: ~250 effective params / 6400 rows gives a variance inflation of about 1.02, so noise sigma ~3.02.
  The clean CV of 3.077 is within ~2% of the irreducible error.

## Public LB
- c03: public 3.36875 (#1/5; #2 3.37652). Local: clean 3.077 < test-like 3.297 < public < shiftW 3.544.
  Read: test-like CV is the closest proxy, and shift-weighting with x4 spike weights is conservative. c03 frozen (read-only).

## Round 3 (after LB): residual analysis -> imputation hybrid (c04)
OOF segment analysis of c03 (`src/analysis_c03.py`, `src/evalkit.py`):
- Complete rows: RMSE 3.009, about the noise floor; they carry 64% of the test-like squared error but have no headroom.
  Rows missing occupancy or prev: RMSE 4.55, 28% of the error -> this is the only real headroom.
- The masked spike bias (+0.77) is a missingness effect. Clean OOF spike biases are small and noisy.
- Residential has a schedule STEP (energy 51 -> 45 at 08:00, 47 -> 56 at 21:00) that smooth Fourier misses
  (+-1.5 biases, >3 SE) -> per-hour deviations shrunk toward the smooth profile (scale 0.25): ~-0.007 on all metrics (V4).
- No academic-calendar month effect in occupancy; weekday/weekend dominates.
Key mechanism found: a no-occupancy sub-model inflates the prev coefficient (prev acts as an occupancy proxy).
  Prev spikes are not accompanied by occupancy (spike z corr -0.39), so that inflated slope mis-extrapolates.
  Fix: impute the missing feature with its conditional expectation (target-free LightGBM fitted on train+test FEATURES,
  test-like mask augmentation), then apply the full linear model. Exact for a model linear in that feature.
  Imputer accuracy: occupancy R2 0.915 (vs 0.865 implied by the linear design), prev 0.850 (vs 0.823).
| id | model | clean | masked | shiftW |
|---|---|---|---|---|
| i00 | V4 pattern only | 3.072 | 3.270 | 3.510 |
| i01 | V3 + LGB impute occ/prev, train-only imputer | 3.065 | 3.268 | 3.456 |
| i03 | V4 + impute occ/prev, train+test imputer | 3.054 | 3.246 | 3.376 | (both-missing rows worse: imputer never saw NaNs)
| i07 | + imputer mask augmentation | 3.053 | 3.219 | 3.364 |
| i10 | + impute temp & hum too (all 4) | 3.049 | 3.209 | 3.343 |
| blend | w*imputed + (1-w)*pattern | w 0.7-1.0 flat | best 0.8 (-0.005) | best 1.0 | -> w=1 (simplest, best on spikes)
**c04 = i10 config, 5x4 paired vs c03: clean -0.029, masked -0.068, shiftW -0.180, spike -0.725, non-spike -0.029;
4/4 wins on every metric; masked MSE delta -0.447, 95% CI [-0.666, -0.255].**
Test preds vs c03: corr 0.9987; complete rows rms change 0.22; changes concentrated on rows with missing values (rms 2-3.7).
Stress curves (reports/figures/stress_curves.png): monotone, linear when observed. Small tree-imputer kinks
  inside the support (<=4 units). Gentler out-of-range slope when a proxy feature is missing.

## Public LB: c04
- c04 public 3.33681 (#1/6) vs c03 3.36875: -0.032. Local test-like predicted -0.068 -> ~47% transferred.
  Gap public - test-like: c03 +0.072, c04 +0.109. Direction confirmed; magnitude over-predicted by the masked CV.
  c04 frozen (read-only). Next focus: private robustness of the imputers (fitted on train + public-test features).
