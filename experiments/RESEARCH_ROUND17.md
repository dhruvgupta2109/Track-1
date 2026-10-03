# Round 17: foundation models, generator hypotheses and deployable distillation

Work on `codex/final-model-improvements`, 3-4 October 2026. Confirmed starting score:
**c19, public RMSE 3.18380**, submission #2284 reported by the user; SHA prefix `db7a2d0d52` matches.
No portal uploads or public-label queries were performed. New candidates remain unscored.

## Result and files

Try [c23_tabicl_compact.csv](../submissions/c23_tabicl_compact.csv) for the strongest measured components.
For an executable private package, use the separate
[c22_distilled_compact.csv](../submissions/c22_distilled_compact.csv) and
[c22 package](../package/c22_distilled_compact/).

| Candidate | Feature imputers | Energy response | Platform inference |
|---|---|---|---|
| c19, confirmed baseline | Energy-weighted occupancy blend | V5 | Supported |
| c20 | Fixed half blend with TabICL for occupancy/previous usage | V5 | Torch/TabICL absent from image |
| c21 | c19 imputers | Fixed half compact/V5 blend | Supported |
| c22 | Fixed half distilled-TabICL/c19 blend for those two features | Same compact/V5 blend | Supported, verified |
| c23 | c20 foundation imputers | Same compact/V5 blend | Torch/TabICL absent from image |

c20 and c21 isolate the changes; c22/c23 combine them. c23 has a validated prediction CSV, **not an uploadable
private inference package**. c22 contains only plain arrays, dictionaries and LightGBM booster strings; no neural
checkpoint, Torch, TabICL, training data or network access is needed at inference.

## Research and decisions

Reviewed primary research on [TabPFN](https://www.nature.com/articles/s41586-024-08328-6),
[TabICLv2](https://arxiv.org/abs/2602.11139),
[incomplete-data Gaussian mixtures](https://arxiv.org/abs/0905.2979),
[symbolic regression](https://arxiv.org/abs/2305.01582), and
[knowledge distillation](https://arxiv.org/abs/1503.02531).
Checked the current [TabICL implementation](https://github.com/soda-inria/tabicl) and API through Context7.

Selected TabICL because its released model supports regression and inference with missing numeric features,
and its open checkpoint could be tested locally. Its pretraining on varied synthetic tables is a plausible match
to this problem, not evidence that it contains this generator. Installed TabICL 2.2.0 in an isolated research
environment; reused working CUDA Torch 2.14.0 and the RTX 5060 Ti. The platform-pinned `.venv` is unchanged.
Only public model weights were downloaded; no dataset was uploaded to a model API.

Instead of a generic large-model blend, use the foundation model where the baseline has most uncertainty:
missing occupancy and previous usage. Observed values still pass through the extrapolating energy formula.
Temperature's test-feature score worsened, so its existing imputer remains. The half blend was fixed;
the feature choice is exploratory, based on observed validation results.

## Held-out gap-filling results

Five matched train/test feature folds, same mask seeds as the saved c19 comparator. Held-out rows are excluded
from context, fits and deviation tables. Public-test **feature** targets are observed entries deliberately hidden;
public energy labels are unavailable. The proxy sums per-feature energy-slope-weighted MSEs.

| Imputer variant, energy formula held fixed | Test-feature proxy | Naturally incomplete test slice | Train-feature proxy |
|---|---:|---:|---:|
| c19 | 20.7249 | 26.6991 | 12.2003 |
| Pure TabICL, all four features | 19.4223 | 26.7116 | 11.6911 |
| Half TabICL/c19, occupancy + previous only | **19.1328** | **25.3363** | **11.6182** |
| Half distilled student/c19, same two features | **20.0802** | **26.1447** | **11.9570** |

Foundation blend: test proxy change **-1.5921**, row-bootstrap 95% interval **[-1.9838, -1.1976]**, with
negative changes in all five folds. Naturally incomplete change -1.3628, interval [-2.8330, -0.0070].
Student blend: test change **-0.6447**, interval **[-0.9606, -0.3255]**, also negative in all five folds.
Its naturally incomplete change -0.5544 has interval [-1.8143, +0.5697], so that slice's gain is uncertain.

These are **not leaderboard RMSE changes**. They exclude covariance of joint feature errors and approximate the
temperature-squared response. Scoring slopes come from the previously frozen labelled c04 fit; train-feature
validation is not a nested energy-slope fit. Row bootstraps preserve within-row feature correlations and scale
contributions to the exact sum-of-feature-MSE estimand. Intervals condition on fitted models and do not cover
architecture/feature selection uncertainty or train-to-private distribution changes.

## Reverse-engineering checks

Weather has a clear shared sinusoidal daily/annual component. Robust train temperature mean approximately:

`28.216 - 1.490 sin(2 pi hour/24) - 1.518 cos(2 pi hour/24)`
`       + 0.253 sin(2 pi month/12) - 0.492 cos(2 pi month/12)`.

Temperature's daily amplitude is about 2.13 degrees with an afternoon peak. Humidity regressed on temperature
plus clock terms has temperature coefficient **-1.782**, consistent with a simple -1.8 relationship.
These are fitted fingerprints, not recovered source code. Central residual variation and spike contamination remain.
The same partial humidity coefficient is attenuated to -1.418 on test features, so a single clean physical
relationship is an inadequate model of the supplied test corruption. Exact estimates are reproducible with
`exp54_weather_generator.fingerprint()` and saved in `weather_generator_fingerprint.json`.

Tested an explicit feature generator: robust building/time/domain mean profiles, per-building residual scales,
and conditional Gaussian mixtures fitted to incomplete feature rows using EM. Domain/hour-band component
probabilities accommodate different spike frequencies. Components 1/8/16 were tested on all five folds.
Best half blend, 16 components: test proxy **22.4889**, worse than c19's 20.7249; rejected.

Also tested a focused two-dimensional weather generator with sinusoidal means and 1/4/8 residual components.
Best half blend's test proxy **21.2295**, also worse; rejected. A simple-looking weather relationship does not
make inverse reconstruction reliable under this dataset's mixed spikes and missingness.
Two analytical conditional-Gaussian tests verify the EM first/second moments.

The useful generator hypothesis was **a more compact energy formula**: type-specific daily waves, a library
evening bump, a sports twice-daily component, residential day/night steps, seasonal terms and the existing
numeric response. It uses **77 design columns versus V5's 301**. Across three repeated five-fold complete-row
splits, with both models refitted on identical folds:

| Energy model | Complete-row RMSE |
|---|---:|
| V5 | 2.994955 |
| Compact | 2.990468 |
| Compact with coarse coefficient rounding | 2.991421 |
| Fixed half V5/compact blend | **2.988815** |

Energy blend paired MSE change -0.03674, row-bootstrap interval [-0.05693, -0.01639]. Daily-shape hypotheses
were informed by full-training profile inspection; the intervals do not account for that selection. Simple
coefficient rounding is unnecessary. The compact function is projected into V5's existing design, matching
the direct compact predictor within 1e-8, so its row-local input and fallback behavior are retained.
The slight numerical difference from earlier V5 CV comes from these freshly matched fold sets.

Direct foundation energy prediction was worse: RMSE 3.06014 versus refitted V5's 2.99966 on matched complete
rows. Its residual learner gave 2.99968, a negligible change. Keep the structured energy response and use neural
inference only for missing feature expectations.
Complete-row errors remain near 3, which supports limited headroom but does **not prove** an irreducible floor.

## Distillation and deployment

First tested teachers cross-fitted within each outer fold, with the held-out feature target absent from the
teacher's context. The two-fold pilot lost most of the foundation gain and worsened the incomplete slice.

The retained student uses new unlabeled queries: five copies of training-fold rows with test-like masks and
15% per-building-SD jitter on the other numeric features. The full outer-fold teacher labels these queries;
held-out feature rows never enter context or queries. Jitter avoids exact context-row matches. Three LightGBM
capacities learn the soft feature targets, with domain and energy-slope weights, then form a fixed half blend
with c19. Counterfactual-query targets are teacher estimates, not true feature labels; training RMS differences
are diagnostic. The student was selected by independent outer feature errors, not those training diagnostics.

c22 combines that student with c21's energy blend. Both components were validated separately; the combined
pipeline has not been measured on public/private energy labels. Train-estimated MNAR parameters remain frozen
to isolate the changes, so recalibration is a remaining potential research direction.

Checks: c21 and c22 passed all five copied-package fresh-kernel scenarios. c22 matched all 3000 predictions
exactly in a fresh image-pinned environment, actual interpreter verified, **2.9 s** including kernel startup.
Its seven input-contract tests cover type recovery, row/chunk/duplicate-index invariance, all 16 patterns x
12 buildings, malformed/absent inputs and injected failures. c23 predictions preserve sample ID order, are
finite, and agree after idless shuffling/chunking to below 5e-14. Portal execution remains untested.

## Reproduction

All earlier packages/CSVs are frozen; builders refuse overwriting them. Fit routines and per-row error caches
are retained, including rejected trials. Research environment: `tmp/venv-tabicl` (ignored), Python 3.12,
TabICL 2.2.0, CUDA Torch 2.14.0, image-pinned numpy/pandas/scipy/sklearn/LightGBM.

```powershell
tmp/venv-tabicl/Scripts/python.exe -X utf8 src/exp50_tabicl.py --folds 5
.venv/Scripts/python.exe -X utf8 src/summarize_tabicl.py
.venv/Scripts/python.exe -X utf8 src/exp51_generator_mixture.py --folds 5
.venv/Scripts/python.exe -X utf8 src/exp52_compact_formula.py
tmp/venv-tabicl/Scripts/python.exe -X utf8 src/exp53_distill_tabicl.py --folds 5 --trio --synthetic
.venv/Scripts/python.exe -X utf8 src/exp54_weather_generator.py
.venv/Scripts/python.exe -X utf8 src/summarize_distillation.py
```

New candidate names are required to rebuild packages. c22 fitting needs the research environment, but
`src/verify_existing_package.py` and its finished prediction notebook use only the pinned platform environment.
A missing `jupyter-nbconvert` launcher in the research environment interrupted the first package verification;
the already-saved artifact/CSV were then verified in `.venv`, without retraining or changing their contents.

Hashes and validation records: `c20_tabicl_gap_blend.json`, `c21_compact_energy_blend.json`,
`c22_distilled_compact.json`, `c23_tabicl_compact.json`; uncertainty: `tabicl_blend_summary.json` and
`research_improvement_uncertainty.json`. No current evidence warrants promising a dramatic public RMSE drop.
