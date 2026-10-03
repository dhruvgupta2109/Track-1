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

## Round 4: private-robustness of c04 imputers (c05 experiments)
- A. Train-only imputers (c05a, 5x4 CV): 3.053 / 3.233 / 3.431 (c04 3.048 / 3.228 / 3.364). Keeps most of the test-like gain, loses 1/3 of the shiftW gain.
- Imputer accuracy on HELD-OUT rows (features only), two private-set hypotheses (`src/exp22_minimax.py`), energy-weighted imputation MSE:
  | imputer | train-like rows | test-like rows |
  |---|---|---|
  | train-only | 14.44 | 35.33 |
  | train+test x1 (c04) | 14.36 | 27.41 |
  | test rows x2 | 14.90 | 26.50 |
  | test rows x4 | 15.77 | 25.69 |
  -> c04 is no worse than train-only on train-like rows and much better on test-like rows. The transductive
     imputer carries no measurable regret. Test up-weighting is a bet on private=public; not adopted.
- Imputation is much harder on test rows (occupancy RMSE 34 vs 22) because the spikes are unpredictable. This explains
  why train-row CV over-predicts the c04 gain.
- B. Bagging 5 seeds, shallower/deeper imputers, 2 augmentation copies: all within +-0.1 of c04 test-row imputation MSE, or worse.
- C. Imputation-uncertainty blend: not pursued. Seeded bags barely disagree (bag5 ~ single model), so there is no reliable confidence signal.
- D. c03/c04 OOF blend: global w=0.85-0.9 gives -0.004 masked (4/4) but worse shiftW (+0.003) and spike rows (+0.05) -> rejected.
  Per regime: prev-missing w~0.7 (4.089->4.050), both-missing w~0.6 (5.96->5.82), occ-missing w=1.
  About -0.007 RMSE overall, spike effect unknown -> not adopted.
- Public-delta noise: the c04-c03 per-row MSE differences imply a public RMSE delta of -0.069 +- 0.04-0.055 (1 SE).
  The observed -0.032 is within 1 SE. The public LB cannot resolve differences < ~0.05.

## Round 5: is there signal left beyond c04? (paired vs c04 config s00, 5x2; delta = cand - c04)
- **Residual predictability** (2nd-level CV on c04 OOF residuals): LightGBM R2 -0.019 (clean) / -0.017 (masked),
  ExtraTrees -0.011 / -0.011. The best shrinkage of any residual correction is 0. -> residual = noise on train-distribution rows.
- Slope heterogeneity (ridge-shrunk interactions, scale 0.5): prev x type +0.003, prev x building +0.004, occ x weekend +0.0005,
  occ x hourband +0.0005, prev x hourband +0.0007, prev x weekend +0.0006, temp hinges x type +0.0014, hum x type +0.0018,
  temp x hourband +0.0008 (masked deltas). All null or worse -> shared slopes are correct.
- Density ratio: train-vs-test classifier on feature values, AUC 0.70 (prev, occ, temp drive it). Spike rows get mean
  weight 2.98 vs 0.92 -> independently confirms the spike-weighting scheme. Density-weighted fit: +0.018 masked (worse);
  sqrt: +0.003; sqrt spike weights: +0.002. Ridge alpha 1 / 10: +0.002 / +0.004 (alpha 3 optimal).
- Huber: dropped. Residuals are near-Gaussian (kurtosis 0.16), and lbfgs failed to converge on the unscaled design.
- Ensembles: c04 + LGB(aug) energy model, residual corr 0.885; any weight > 0.05 worse on all metrics.
- Imputer kinds: chained (occ -> prev) +0.004 masked / +0.008 shiftW; ExtraTrees +0.068; LGB+ET average +0.017.
**Conclusion: residual signal is exhausted on train-distribution rows. c04 remains the champion.**
The remaining uncertainty is test-only behaviour (spike magnitudes beyond the train range), which labels cannot validate.

## Round 6: dataset forensics + new information sources (all vs c04)
- **previous_usage does not chain.** Searching for a same-building next-hour row (same month, weekday rolled at midnight)
  whose prev equals y_i: exact matches 0.5% at +1h vs 0.3-0.7% for the +0h/-1h/+2h/+5h controls (`src/forensics_chain.py`).
- **ID order is random.** Consecutive ids share a building 8.1% (chance 8.5%), are 1h apart 3.9% (chance 4.2%);
  corr(prev[n+1], y[n]) = -0.005; c04 residual flat across id deciles (corr -0.004). Rows are i.i.d. samples, so there is
  no time series to exploit and state-space / AR / sequence methods have no support.
- **previous_usage is the SAME slot in an earlier period, not the previous hour.** Its hour profile matches energy(h),
  not energy(h-1) (residential corr 0.97 vs 0.52; the 21:00 step appears at the same hour in both). It behaves like a
  noisy, slightly lower copy of current usage, which is consistent with the shared 0.35 coefficient.
- **Spikes are scheduled.** prev spikes sit at 22-23h (test 27-29% of rows, train 8-9%) and 0-4h (test ~9-10%);
  occupancy spikes at 07-09h (test 18-32%, train 4-7%), never at night. Test has ~3x the train rate in the same cells.
- **Anomaly-aware slopes** (`src/anomaly.py`): adds d_c = x_c - E[x_c | others] (target-free, out-of-fold E) as extra
  linear terms. Fitted deviation coefficients: prev -0.028 / -0.025 (vs slope 0.35), occupancy ~0, temperature +0.24
  for positive anomalies. CV: split +0.0013 masked / +0.0033 shiftW; single +0.0005 / +0.0005 (CI spans 0).
  -> spike rows respond with the SAME slopes as ordinary variation; c04's linear extrapolation is the right model.
- Residual predictability, more model classes: CatBoost R2 -0.005, MLP R2 -0.004 (LightGBM / ExtraTrees earlier -0.01..-0.02).
- Not pursued, with reasons: external data (no date or year, so no join key); delta target y - prev (identical for a
  linear model, and trees lose on extrapolation); bagging the linear fit (~300 params / 8000 rows; imputer bagging
  already null); KNN local correction (ExtraTrees, a local method, already has R2 < 0).
- Edge case: an imputer may predict a slightly out-of-range value (e.g. occupancy < 0). c04 then routes that row
  to its pattern sub-model (graceful). Public test: 0 rows affected.
**Ceiling conclusion: c04 remains champion; no information source beyond it was found.**

## Round 7: chasing 3.26 (public leader)
- Searched online for the source dataset (Kaggle, GitHub, dataset journals; ids EC000001, building codes LEC_A etc.):
  not found. The dataset appears to be generated by the organisers.
- No train-test duplicates (exact, or ignoring prev / occ / weather); weather varies per row (temp sd within month/dow/hour 0.99).
- Missingness and spikes co-occur in BOTH train (spike rate 6.0% vs 4.0% on rows with vs without missing) and test
  (20.7% vs 15.7%). Spike magnitudes are similar (median |z| 4.0 vs 4.4) -> same generator; test is not separately corrupted.
- Only 65 test values exceed their building's train range (clipping changes predictions by rms 0.14), so
  extrapolation beyond train support is NOT a material part of the public error.
- Error budget: complete test rows (82%) sit at the ~3.0 noise floor; rows with missing values (18%) run at ~4.5,
  giving ~3.33 overall. That matches c04 public 3.337, so the lever is the missing-value rows.
- **Domain-aware imputers (c05):** the imputers get a feature dom (0 = train rows, 1 = test rows) and predict with dom=1.
  On held-out rows (features only), energy-weighted imputation MSE: test-like 26.7 -> 24.1, train-like 13.99 -> 13.68.
  Test occupancy bias -4.7 -> -0.4, prev bias -0.84 -> -0.06 (c04's pooled imputer under-imputed the test spike rate).
  Train-distribution CV with dom=0 (n/a for test): masked -0.004, non-spike -0.015 (4/4), spike +0.20.
  Expected public effect from the imputation errors: about -0.027 (3.337 -> ~3.31).
  Imputer tuning on test rows: default near-optimal (deeper -0.1/24.0; mask augmentation essential, none = 25.4).

## Public LB: domain-aware imputers (huge transfer)
- c04 3.33681 -> c05 3.25549 -> c06 3.23804 -> **c07 3.23149 (#1)**.
- The public order c07 < c06 < c05 < c04 matches the held-out TEST-row imputation proxy exactly (23.57 < 23.65 < 24.06 < 26.7).
  The proxy (energy-weighted imputation MSE on hidden observed test-row features) is label-free and validated by the LB.
  Public gain per proxy unit is about 0.03-0.04 RMSE (bigger than the a-priori estimate). New objective: minimise the proxy.

## Round 8: proxy-driven imputer tuning (src/proxy.py: per-building slope-weighted test-row imputation MSE)
Calibration: c04 25.85 (LB 3.337), c06 22.65 (LB 3.238), c07 22.61 (LB 3.231) -> ~0.031 public RMSE per proxy unit.
| variant | test proxy | train-like |
|---|---|---|
| trio, test weight x3 / x4 / x6 / x10 | 22.64 / 22.68 / 22.86 / 23.07 | 12.64 / 12.78 / 12.98 / 13.37 |
| trio, test-only rows | 24.78 | 19.66 |
| trio w2, mask augmentation x2 / x3 / x5 / x8 | 22.44 / 22.33 / **22.21** / 22.21 | 12.53 / 12.55 / 12.56 / 12.62 |
| 3 seeds x trio (9 members) w2, aug x1 / x5 | 22.47 / 22.18 | 12.52 / 12.57 |
| trio w1, aug x5 | 22.35 | 12.48 |
| trio + ExtraTrees / + heavily regularised LGB | 23.27 / 22.61 | — |
-> **c08 = trio, test x2, aug x5** (proxy 22.21, projected LB ~3.22). Augmentation saturates at x5; extra members
   are worth ~0.001 public for 3x model size (not packaged).
Proxy breakdown at c08: occupancy 9.2 + prev 9.9 + temperature 3.0 + humidity 0.07 -> the floor is unpredictable spikes in missing features.

- **c08 public 3.20937 (#1)**: -0.022 vs c07 for proxy -0.40 -> ~0.055 public per proxy unit (transfer is ~1.8x the calibration estimate).

## Round 9: deviation features for imputers (c09)
Spikes co-occur across features, so imputers get dev_c = x_c - profile mean (target-free lookup tables).
| variant (single imputer, w2, aug2 unless noted) | test proxy | train-like |
|---|---|---|
| no dev (ref) | 22.65 | 12.78 |
| dev (bldg,hour,weekend) | 21.99 | 12.55 |
| dev (bldg,hour,dow) / + z-scores / profile per domain | 22.34 / 22.18 / 22.82 | |
| dev (type,hour) / (building) / (bldg,hour) | 21.91 / 21.73 / 21.68 | |
| dev (bldg,hour)+(bldg) | 21.40 | 12.37 |
| trio w2 aug5 + dev (bldg,hour,weekend) | 21.29 | 12.27 |
| **c09: trio w2 aug5 + dev (bldg,hour)+(bldg)** | **20.87** | **12.16** |
c09 vs c08: proxy -1.34 (both test-like and train-like improve). At 0.03-0.055 public per unit, that projects to about -0.04 to -0.07.
- **c09 public 3.20392 (#1)**: -0.0055 vs c08 despite proxy -1.34. Public delta per proxy unit across steps:
  c04->c05 ~0.031, c07->c08 0.055, c08->c09 0.004. The proxy ranks correctly but its magnitude is unreliable for this
  change. A likely reason: c09's gain sits on masked-observed values, whose co-missingness and spike structure differs
  from the truly-missing test values.

## Round 10: c10 blend, NATMISS proxy, natural-missingness indicators
- c10 = average of c08 + c09 pipelines: proxy 21.21 (between c08 22.21 and c09 20.87) -> not recommended (HOLD).
- NATMISS proxy (held-out test rows that also have a genuinely missing value elsewhere):
  c06 30.01, c07 29.88, c08 28.57, c09 27.63. It credits c07->c08 (-1.30) more than c08->c09 (-0.95), consistent with
  the public deltas (-0.022 / -0.0055) up to public noise (~+-0.005).
- Natural-missingness indicators for the other columns: single imputer 21.80 / 30.23 (count: 21.69 / 29.57) vs
  21.40 / 29.05 without; c09 + indicators 21.17 / 28.65 vs c09 20.87 / 27.63 -> worse (overfits the few naturally-incomplete rows). Rejected.
**Best: c09 (public 3.20392).**

## Big-picture accounting (step back)
- Cross-row linkage: identical (month, dow, hour, temp, hum) across rows in 5 groups vs 7 in a shuffled control.
  Weather is per-row; there are no shared timestamps. Every row is independent; no cross-row information exists.
- Complete train rows (c04 OOF): RMSE 3.001 plain, 3.058 density-weighted to test -> complete test rows (82%) ~3.0-3.06.
- Implied error on test rows WITH missing values (18%): c04 4.4-4.55 -> c09 3.8-4.0. All public gains came from these rows.
- Floor: noise ~3.0 plus the unrecoverable imputation variance (spikes hidden in the missing value; c09 per-feature
  imputation energy-MSE: occupancy 8.4, prev 9.8, temperature 2.6). Perfect imputation would give ~3.03 public;
  realistic remaining headroom is a few hundredths, at rising cost per step and inside public noise.

## Round 11: teammate's submission 007 (public 3.19799) + research-driven ideas
- 007 (external/submission_007) built on c09: (a) found a MASK-ORDER BUG in c09 imputer augmentation (common.NUM vs
  features.NUMERIC order: test missing patterns applied to the wrong columns); (b) energy model with residential
  day/night steps + hour-deviation 0.1; (c) cyclic weather features, profile blends, clipping. 007 vs c09 rms 0.22.
- Research (Le Morvan/Josse NeurIPS'21; missing-indicator method; MIRRAMS): impute-then-regress is Bayes-optimal for
  any mechanism only if the regressor is trained on imputed rows. MNAR can hide from MCAR-masked validation.
- MNAR check on NATURALLY missing train rows (c09 imputers, mask fixed, OOF 5x2): previous_usage missing ->
  residual +1.02 (se 0.35); temperature -0.38 (0.30), humidity +0.17, occupancy +0.20 (0.40).
  Label-free mechanism test (missing rate vs cell spike rate): train prev r=3.9 (n.s.), TEST r=1.4 (n.s.) ->
  most test missingness looks ~random w.r.t. spikes; test-side bias probably small. c12 probes it (+0.6 offset).
- Energy V5 (residential steps + hour_dev 0.1), paired 5x4 complete-row RMSE: V4 3.0050 -> V5 2.9978 (4/4 wins);
  per-building WLS adds nothing (-0.0002).
- Imputer proxy (mask fixed): c09 20.87/27.63 (test/NATMISS) -> fixed 20.84/27.37 -> fixed+cyclic 20.81/26.86.
- c11 = fixed+cyclic imputers + V5 energy. c11 vs 007: complete rows rms 0.035 (same energy model reached
  independently), missing rows rms 0.40 (different imputers) -> c13 = 50/50 average.
- 007's own notebook in our clean Windows env differs from its CSV on 37 missing-value rows (max 0.067):
  not bit-reproducible across platforms. c13 matches mean(c11, 007-notebook-output) to 3e-14.
- **LB (hash-verified: site FILE HASH = sha256[:10])**: c11 3.19406, c12 **3.18937 (#1)**, c13 3.19578.
  The prev-missing offset (+0.6 on ~7% of rows) gained -0.0047 -> missing-not-at-random previous_usage is present in
  test. Averaging with 007 hurts (its imputers are weaker than c11's).

## Round 12: MNAR deep-dive (self-masking) after c12 confirmed MNAR in test
- Research: NeuMiss (Le Morvan et al. NeurIPS 2020): Bayes predictor for linear models under self-masking MNAR =
  imputed prediction + pattern-dependent shift; pattern-mixture "delta adjustment"; Mercaldo & Blume pattern submodels.
- prev bias NOT concentrated in spike hours (+0.52 in 22-04h vs +1.26 elsewhere); by type: SPT +2.5, SCI +2.1, ADM +1.9,
  RES +0.1, ENG +0.2 -> consistent with self-masking (shift scales with conditional SD).
- Conditional SD per (domain, building) from OOF imputation errors (experiments/imputation_oof_errors.csv):
  test > train for prev (e.g. ENG_A 11.1 vs 7.2) and occupancy (RES_A 46 vs 27).
- Self-masking fit on single-missing natural rows (shift in SD units): prev kappa 0.41 (se 0.14), occupancy 0.20
  (0.20), temperature -0.19 (0.22), humidity noisy; pooled 0.25 (0.10).
- c12's LB gain (-0.0047) is only ~1.3 SE of public noise for a ~200-row change -> LB confirms direction only.
- Probes: c14 flat prev +0.85 (magnitude); c15 prev offset scaled by test SD (structure, mean 0.6);
  c16 = c12 + occupancy self-masking offsets (0.2-0.88).

## Round 13: grading-method review -> a not-leaderboard-tuned final candidate (c17)
- Booklet: final 60% = RMSE, MAE, R2 on a HIDDEN private test run through our notebook ("different observations and
  edge cases"; "any execution error ... may result in disqualification"); public LB "only an indication".
- LB results: c16 3.18618, c14 3.18808, c15 3.18852 (all < c12 3.18937). Observation from existing scores only:
  the c11/c12/c14 curvature implies prev-missing rows carry ~2.0% of the public score's sensitivity (6.8% of the file):
  the public score likely does not weight all rows equally (or is noised). Not used for any parameter. Leaderboard-
  structure probing was proposed, blocked by the tool policy, and dropped (user: do not overfit public).
- Forensics: IDs are a clean split (train 1-8000, test 8001-11000), no residual autocorrelation by id; test has more
  previous_usage dips (q01 7.9 vs 17.5) but the V5 model is unbiased on train dips and spikes (exp42_dips.py).
- c12-c16 offsets were partly hand-picked (0.6 = manual shrink of +1.02; occupancy in / temperature out by judgement)
  and then selected on public. Replaced by a PRE-REGISTERED train-only rule (exp43_mnar_trainfit.py):
  c11 imputers (dom 0) + V5 energy OOF 5x4 on naturally-missing train rows; kappa_c by WLS on single-missing rows:
  prev +0.418 (0.153), occupancy +0.154 (0.175), temperature -0.196 (0.221), humidity +1.46 (1.56).
  Empirical-Bayes shrinkage, skeptical zero-centred prior (ML tau2 0.058): 0.297 / 0.101 / -0.106 / 0.034 (chosen
  before any public score; exchangeable-REML variant 0.342 / 0.160 / -0.021 / 0.197 kept as a sensitivity check).
  Cross-fitted on train (50 splits): dMSE vs no correction -0.078 (se 0.126); prev-only raw -0.187 (0.184).
- c17_mnar_eb = c11 + kappa x slope (finite difference at the imputed value) x test conditional SD per building.
  Test mean shifts: prev +0.88, occupancy +0.41, temperature +0.13, humidity +0.19. rms vs c11 0.25, vs c16 0.12.
  Inference wrapped: exception -> plain linear model, non-finite -> train mean (never triggered on test, stress,
  dropped-column inputs). 5 fresh-kernel scenarios + image-only clean env (exact match, 2.5 s). sha256[:10] 05cb6f912c.
- **LB c17 3.18775** (c16 3.18618, c11 3.19406): train-only MNAR keeps ~80% of c16's gain over c11; the 0.0016 gap
  to c16 is inside public noise. c17 stays the principled final candidate (decision rule fixed before the score).

## Round 14: broad sweep for remaining signal (all label-free / train-validated) - all negative
- Energy response at extremes (exp44): residual ~ z + z^2 on train complete rows: no curvature for prev (t +1.1) or
  occupancy (t +0.7); residual SD rises with |z| (2.91 -> 3.40) = heteroscedastic noise, mean ~0.
- Test missingness co-occurs with ABNORMAL rows: observed-feature spike rate ~3x in rows with a natural missing value
  (prev-missing rows: occupancy spikes 28% vs 8% in complete rows); train shows ~none. But imputation errors of observed
  features in those rows have mean ~0 (prev +0.06 se 0.065, others |.|<0.04 SD), RMS 1.13-1.22 vs 0.97: abnormality
  is two-sided -> more variance, no exploitable mean shift (prior natural-missing-indicator imputers were also worse).
- Test spikes = same mechanism as train (same schedule: prev 22-03h, occupancy 07-09h; same |z| median ~4; similar
  building mix) -> linear extrapolation stays justified.
- Error budget of imputation (test domain, slope-weighted): prev 51%, occupancy 33%, temperature 16% (SCI slope
  2.5/degC), humidity ~0; ~1.2 energy-MSE per test row in total.
- Level drift train->test (exp45): test busier (matched-by-hour medians prev +3.3, occupancy +12.7), but given
  occupancy and time features, test prev is exactly as train predicts (median residual -0.04 vs -0.06) -> no
  unexplained energy-level drift. (Train: month-level energy moves 0.95x with prev level.)
- Imputer calibration in test domain: slope true-on-imputed 1.03 prev / 1.02 occupancy -> rescaling cuts imputation
  MSE <0.3%: negligible.
- Spike clustering: spike OCCURRENCE is scheduled (LOO corr 0.36 within building x dow x hour) but spike DIRECTION is
  not (corr 0.03) -> nothing for the conditional mean.
- Interaction screen on V5 OOF residuals (exp46, 20 groups incl. building x hour/dow/month, per-building prev slope,
  occ/prev curvature, temp x humidity, temp x season x building): every group worsens out-of-sample MSE (>= +0.0019).
- Conclusion: complete rows at the noise floor; imputers calibrated and saturated on the proxy; MNAR handled from
  train. Remaining public differences between principled variants are inside public noise.

## Round 15: independent imputer ideas, input recovery and final delivery (Codex branch)

Branch: `codex/final-model-improvements`, starting from `4c23bfe`; all earlier packages/prediction CSVs preserved.
No portal actions or new public submissions. The score/model records above remain historical evidence, not fresh private scores.

### A. Extrapolating gap-fillers: rejected

`src/exp47_extrapolating_imputers.py`: fixed building/time/domain ridge means, then LightGBM on feature residuals.
Compare linear, residual trees, and fixed 50/50 blends against the c11 trio. Same 5 held-out train/test folds as the
existing proxy; exclude held-out rows from fits and deviation tables. Correctly ordered x5 mask augmentation,
same trio settings/seeds, no test energy labels in fitting or evaluation. The scoring slopes are from the previously
frozen c04 fit on labelled train data; the train-feature proxy is not a nested validation of those slopes.

| Variant | Test-feature proxy | Naturally incomplete slice | Train-feature proxy |
|---|---:|---:|---:|
| c11 trio reference | 20.8086 | 26.8569 | 12.2480 |
| Linear | 42.6594 | 51.6153 | 19.5501 |
| Linear + residual trees | 22.9377 | 28.5963 | 12.7531 |
| Fixed blend with linear | 26.9431 | 34.3995 | 13.9491 |
| Fixed blend with residual trees | 21.2568 | 27.2083 | 12.1462 |

Residual blend dMSE +0.4482, paired row-bootstrap 95% interval [+0.2141,+0.6881] on held-out test features.
The train-domain improvement did not transfer to test-domain feature reconstruction. Retain the existing imputers.
Full errors: `experiments/extrapolating_imputers_5fold_trio_errors.csv.gz`; the two-fold single-member pilot is also retained.

### B. Train imputers for their effect on energy: uncertain small gain, not promoted

`src/exp49_energy_weighted_imputers.py`: multiply each feature's training weights by its squared per-building energy
slope, normalized to preserve overall weight. Shared-slope features stay identical. Fixed 50/50 blend is a pre-specified
diagnostic; no blending weight was fitted. Pair against the exact cached c11 baseline on the same feature values/folds.

| Variant | Test-feature proxy | Naturally incomplete slice | Train-feature proxy |
|---|---:|---:|---:|
| Energy-weighted trio | 20.8176 | 26.6898 | 12.2476 |
| Fixed blend with reference | 20.7342 | 26.7044 | 12.2023 |

Blend dMSE -0.0744, paired SE 0.0527; row-bootstrap 95% interval [-0.1772,+0.0265]. Naturally-incomplete slice:
dMSE -0.1525, interval [-0.4835,+0.1405]. This possible gain does not justify more model size and complexity.
Intervals resample rows jointly across feature errors, after scaling contributions to preserve the proxy's sum-of-feature-MSE
estimand; they condition on fitted folds and do not include model-selection uncertainty. They are not public RMSE intervals.
Errors: `experiments/energy_weighted_imputers_errors.csv.gz`; uncertainty: `experiments/imputer_trial_uncertainty.json`.

### C. c18_private_safe: preserve c17, recover inputs and isolate failures

`src/exp48_contract_audit.py` identified a recoverable input defect: removing/lowercasing building_type changed c17
predictions by up to 2.3042 on the first 400 supplied-test rows even when building_id remained known.
c18 derives known building types from the fixed building map and canonicalizes other recognized types.
An individual imputer failure leaves only that feature missing, preserving other fills; linear sub-models handle the
remaining pattern. MNAR shifts require successful complete imputation. If both energy paths fail, return the finite train mean.
File loading and dependency failures remain outside these prediction fallbacks.

- c18 supplied-test CSV is **byte-identical to c17**, SHA-256 prefix `05cb6f912c`; 3000 rows, sample ID order, finite.
- c17's existing public RMSE 3.18775 applies to that identical CSV; c18 was not separately submitted. c16 remains best
  observed public at 3.18618, with partly public-selected offsets. No private-score improvement is claimed.
- Five fresh-kernel notebook scenarios pass. Seven regression tests cover type recovery, all 16 numeric patterns x
  12 buildings, empty/absent/malformed inputs, shuffled/chunked/duplicate-index rows and injected failures.
- Fresh environment created offline from cached platform-pinned libraries plus Jupyter tooling: 7.7 s including startup,
  exact prediction/ID match. Actual portal execution remains unchecked.
- `src/train_final.py` independently rebuilds the energy model, all 12 imputer members, five-fold conditional variances
  and train-only EB calibration from raw files, without fitted experiment outputs. Executed presentation: 40 s rebuild,
  max prediction difference 2.84e-14; 15 code cells executed, zero errors.
- `src/build_package.py` now refuses to overwrite existing packages/CSVs and records hashes/source-artifact provenance.
- README, one-page report, executed presentation, Q&A and checklist now explain c18/c17, c16's public result and assumptions.

Evidence corrections in the final explanation: previous_usage's same-slot interpretation is inferred, not the exact known
generator; ~3.0 complete-row error supports limited headroom, not proof of a random irreducible floor; the feature proxy
is approximate for correlated joint missingness and temperature squared; train-domain proxy success does not prove
private robustness when deployed with the test-domain flag. The c17 MNAR cross-fitted improvement remains uncertain.
