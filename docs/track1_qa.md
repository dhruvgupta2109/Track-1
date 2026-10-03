# Track 1 - Judge Q&A (c18 package, c17 statistical model)

Sources: raw CSVs, `experiments/NOTES.md`, ledger/OOF predictions, new imputer error caches and the executed analysis notebook.

| Evidence | Result | Meaning |
|---|---|---|
| c04 historical full-pipeline CV | 3.048 clean / 3.228 test-like / 3.363 shift-weighted | Paired 5-fold x4; not c17 CV |
| V5 complete-row energy-model CV | 2.998, versus V4 3.005 | 7537 complete train rows; 4/4 repeat wins |
| c11 test-feature proxy | 20.809; naturally incomplete slice 26.857 | Feature reconstruction errors, not energy RMSE |
| Best recorded public | c16 3.18618 | Offsets partly selected using public results |
| Final statistical model | c17 public 3.18775 | Train-estimated EB correction rule |
| Final package | c18; byte-identical c17 CSV | Better input recovery; no separate score |
| New weighted-imputer blend | Proxy 20.734; delta CI [-0.177,+0.027] | Uncertain gain; not promoted |

**1. Give us the solution in one minute.**

Test rows contain more missing inputs and more extreme occupancy/previous-usage values. Energy follows the extremes
linearly on labelled rows, so we use a structured linear energy equation. LightGBM fills missing inputs from the other
features; test-aware training improves those conditional estimates. Naturally missing train rows then reveal bias that
random masking misses, so a fixed empirical-Bayes rule adds small missing-value corrections. The c18 wrapper preserves
c17's predictions and handles missing building types or individual imputer failures more carefully.

**2. Why use linear energy prediction rather than a tree model?**

The historical tree baseline scored 3.964 on test-like CV versus 3.228 for the imputed linear model. Train spikes follow
the ordinary energy response (previous-usage correlation 0.92 in the diagnostic). Trees saturate at their support limits;
linear occupancy/previous-usage terms keep responding. This supports extrapolation within the labelled extremes, and
assumes the response continues beyond them. It does not prove all possible private extremes follow the same equation.

**3. What is previous_usage? Can rows form a time series?**

The official description says previous period's usage. Its profile follows energy at the same hour, rather than the
preceding hour (residential correlations 0.97 versus 0.52). A same-slot reference is consistent with the data; we cannot
recover the generator's exact definition. ID order, candidate hourly chains, duplicate searches and shared weather keys
yielded no useful sequence. We therefore treat rows as independent samples.

**4. Why impute instead of using one model per missingness pattern?**

Dropping occupancy makes previous usage absorb its correlated effect: its slope grows from about 0.34 to 0.46.
That mispredicts a previous-usage spike when occupancy is missing. Filling occupancy first preserves the full equation.
Mean imputation matches a linear conditional-mean equation for occupancy and previous usage; temperature's quadratic
term makes it approximate, and MNAR can require an additional shift.

**5. What are the test-aware imputer settings?**

Each numeric feature has three LightGBM members fitted on train plus unlabelled public-test features. A domain flag
distinguishes distributions; prediction uses the test flag. Public-test rows have weight x2. Five augmented copies
use test joint missingness patterns, in the corrected numeric-column order. Deviation features use building/hour and
building means; cyclic hour/month features improve time handling. No energy target enters imputer fitting.

**6. What did c11 change after c09?**

It fixed a mask-order bug (`common.NUM` differed from `features.NUMERIC`), added cyclic imputer features and adopted
explicit residential day/night steps with stronger hour-correction shrinkage. The complete-row V5 energy-model CV
improved from 3.005 to 2.998. Proxy gains were small; public improved from c09 3.20392 to c11 3.19406.

**7. Why add a missing-value correction?**

Random masking tests observed values and can miss selection bias in genuinely missing values. On naturally missing
training rows, OOF energy residuals imply previous usage sits above its estimate: raw kappa 0.418 conditional SDs,
SE 0.153. That is evidence consistent with MNAR, but cannot distinguish every possible source of imputation bias.
The shift is `kappa x energy slope x conditional SD`; conditional SDs are estimated from target-free feature errors.

**8. How were correction sizes chosen?**

Fit weighted shifts on single-missing labelled train rows using 5-fold x4 energy OOF residuals. Estimate a zero-centred
empirical-Bayes variance and shrink noisy feature estimates toward zero. Final kappas: previous usage 0.297, occupancy
0.101, temperature -0.106, humidity 0.034. Numerical sizes use training evidence, not public-score fitting.
Some earlier c12-c16 offsets were hand-picked; we replaced those with the fixed rule before observing c17's score.

**9. Does the correction definitely improve private performance?**

No. Cross-fitted single-missing train MSE change was -0.078 with SE 0.126. Public c17 improved over c11, but public is a
sample with unknown labels and membership. The mechanism and conditional-SD scaling must transfer to private data.
We present this as a defensible bias adjustment with uncertainty, rather than a proven private gain.

**10. Why choose c17 when c16 scored better?**

c16's 3.18618 is the best observed public score. Its difference from c17's 3.18775 is only 0.00157, too small to establish
a private winner. c17's explicit train-estimated rule is easier to justify than choosing offset sizes on repeated public
results. c18 keeps that same statistical model.

**11. What exactly improves in c18?**

A known building ID supplies its type when absent, lowercased or contradictory. On 400 test rows, c17 changed predictions
by up to 2.304 units when type was removed/lowercased; c18 recovers the original predictions. A broken feature imputer
leaves only that input for a linear sub-model, preserving other fills. MNAR shifts apply only after successful full
imputation. A further linear/mean fallback covers energy prediction failures. The normal test CSV remains byte-identical.

**12. Does "fallback" guarantee the notebook can never fail?**

It covers prediction failures and non-finite results after loading the model and input. It cannot cover missing
dependencies, unreadable files, artifact-loading errors or platform failures. We tested the actual copied notebook in
fresh kernels and a clean environment, but still need the portal sandbox run.

**13. How did you validate without public-test labels?**

Hold out train/test feature rows, hide observed values with test-like co-missingness, and reconstruct them. The proxy
sums per-feature MSE weighted by squared energy slopes. A naturally-incomplete slice checks harder rows. It ranked the
six c04-c09 variants correctly, but gain sizes transferred inconsistently. It is not exact final energy MSE: joint
missing-feature errors can covary, temperature is slightly nonlinear, and genuinely missing values are unobserved.

**14. What new improvement ideas did this branch test?**

Linear gap-fillers, linear plus residual trees, and fixed blends: the residual blend worsened the full five-fold test
proxy to 21.257 versus 20.809. Weighting feature fits by per-building energy slopes produced a fixed blend at 20.734,
but its row-bootstrap MSE interval includes zero. Extra model size and selection uncertainty outweighed this small
possible gain. Neither was promoted or submitted. Those intervals condition on the fitted folds, not full model-selection uncertainty.

**15. What does the noise-floor claim mean?**

Complete-row RMSE is about 3.0, residual learners tested had negative out-of-sample R-squared, and additional
interactions/weighting/blends failed to improve reliably. This supports limited headroom with the available features.
It does not prove the remaining error is random or that every possible method has been exhausted.

**16. How do you avoid leakage?**

The energy fit excludes validation targets. Imputers use only feature values; proxy folds exclude evaluated rows from
fitting and deviation tables. Final imputation is transductive: it fits unlabelled public features, then uses fixed
lookups at inference. Rows do not use IDs, batch order or other private-test rows. Train-only MNAR calibration reuses
target-free fitted imputers while cross-fitting the energy equation; this is distinct from a fully nested pipeline CV.

**17. Is transductive fitting allowed and safe for private data?**

No test energy labels are used. The supplied rules allow additional/public data and do not explicitly prohibit fitting
unlabelled test features; they do not explicitly guarantee this particular method either. Private-domain transfer remains
an assumption. Strong train-domain proxy results do not remove that risk because deployed prediction uses the test flag.

**18. Can you reproduce the final model from raw files?**

Yes. The executed notebook refits the energy model, 12 imputer members, conditional variances and train-only calibration.
It completed in 40 s, matching the frozen prediction CSV to 2.84e-14. The c18 package builder reuses frozen c17 parameters
for packaging; it is distinct from the independent rebuild in `src/train_final.py`.

**19. What did you test for private execution?**

Five fresh-kernel cases: shuffled rows without IDs, template defaults, stress inputs, a dropped numeric column and one row.
Seven regression tests cover type recovery, all 16 numeric patterns across 12 buildings, row/chunk/index invariance,
empty/all-missing/malformed inputs and injected failures. A fresh pinned-library environment plus Jupyter tooling
matched predictions exactly (7.7 s including startup). Portal execution and private scores remain untested.

**20. What campus insights and next data would help?**

An additional 100 occupants is associated with about 13.4 energy units in the lecture hall and 4.4-4.7 in residences.
Science buildings have the highest modelled base loads (~39-43 at the stated empty/low-history 03:00 reference).
Residences show sharp morning/evening schedule changes. These are associations, not causal effects.
Actual timestamps, missingness causes and more labelled extreme events would help verify transfer, link weather and
calendar data, and quantify prediction uncertainty.
