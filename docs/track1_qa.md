# Track 1 — Q&A preparation (final model: c09)

Sources: `experiments/log.jsonl`, `experiments/NOTES.md`, `experiments/proxy_results.csv`, the analysis notebook.

| | Clean CV | Test-like CV | Shift-weighted CV | Imputation proxy (test rows) | Public |
|---|---|---|---|---|---|
| c00 LightGBM (5x2) | 3.341 | 3.964 | 4.281 | — | — |
| c03 linear, per-pattern sub-models | 3.077 | 3.296 | 3.543 | — | 3.36875 |
| c04 + LightGBM imputers | 3.048 | 3.228 | 3.363 | 25.85 | 3.33681 |
| c05 + domain flag | | | | 23.04 | 3.25549 |
| c08 + mask augmentation ×5 | | | | 22.21 | 3.20937 |
| **c09 + deviation features (final)** | | | | **20.87** | **3.20392 (rank 1)** |

c04 → c09 share the same linear energy model; they differ only in how missing inputs are filled, so the energy-model CV
columns apply to all of them.

---

**1. Why a linear model instead of LightGBM?**
Test has about four times train's rate of extreme occupancy and previous_usage values. Energy keeps responding linearly
to them, and trees flatten beyond their training range. On test-like CV LightGBM scores 3.96 against 3.23 for the linear
model. LightGBM is still used, but only to fill missing inputs.

**2. How did you detect the train/test shift?**
- Missing rates: 18.2% of test rows have a missing value vs 5.8% of train.
- A target-free consistency model (each feature predicted from the others): 16.6% of test rows have a spike vs 4.1% of train.
- A train-vs-test classifier on feature values: AUC 0.70.

**3. Why are the spikes not treated as outliers?**
On labelled train rows, energy follows them. The energy left unexplained without the spiked feature tracks the
feature's deviation (r = 0.92 for previous_usage, 0.80 for occupancy), on the same line as ordinary rows.

**4. Why does linear extrapolation help?**
Spiked values have the same marginal effect as ordinary ones (a separate spike slope came out at about 0), so the
straight line is the right extension and trees under-predict large spikes.

**5. What does the final model do, step by step?**
- Clean the inputs; unknown or missing categories become averages over the known levels.
- Fill each missing numeric input with the average of 3 LightGBM imputers.
- Apply the linear energy model.

The model is about 20 MB, inference takes about 3 s, and it uses only numpy, pandas and LightGBM 4.5.0 from the base image.

**6. Why did imputation matter so much (3.337 → 3.204)?**
Rows without missing values (82% of test) are already at the noise floor (~3.0). Nearly all the remaining error is in
the 18% of rows with a missing value. A missing test value comes from the test distribution, which has more spikes.
c04's imputers learned from train and test pooled, so they filled test values too low (occupancy bias −4.7).

**7. What are the "test-aware" imputer changes?**
- **Domain flag (c05):** a feature saying train or test; the imputers predict "as test". The bias drops to −0.4.
- **3-imputer average and test rows ×2 (c06–c07).**
- **Mask augmentation ×5 (c08):** copies of the data with test-like co-missingness, so the imputers work when several inputs are missing at once.
- **Deviation features (c09):** each observed input minus its building×hour and building mean, because spikes co-occur across features.

**8. How did you validate the imputers without test labels?**
A label-free proxy:
- Hide *observed* values in held-out public-test rows, using test-like co-missingness.
- Re-impute them.
- Weight each error by that feature's energy slope for the building.

Because the energy model is linear, this is exactly the energy error caused by imputation. It ranked all six submitted
models (c04–c09) in public-score order. Its magnitude is less reliable (c09's proxy gain transferred only −0.005), so
we stopped when further steps fell inside public noise.

**9. How did you avoid leakage?**
- Imputers never see `energy_usage`; they predict features from features.
- In CV and in the proxy, the evaluated rows are excluded from imputer fitting.
- All features are row-local, with no ids and no cross-row features.

**10. Is using unlabelled test features legitimate?**
We believe so. No test labels are used, and fitting on features only is standard transductive / semi-supervised
practice. The booklet allows additional or public data and does not prohibit unlabelled test features. The deployed
notebook needs only the uploaded model file.

**11. What if the private test looks like train instead of the public test?**
The test-aware changes are a bet that private resembles public (more spikes and missing values), as the handbook's
"edge cases" wording suggests. The downside is small: c09 is also better than c04 on held-out *train-like* rows
(proxy 12.2 vs 13.0).

**12. What do test-like and shift-weighted CV mean?**
Test-like CV gives validation rows missing-value masks drawn from the test file's joint missing patterns.
Shift-weighted CV additionally up-weights spike rows ×4.05 to test frequency; the train-vs-test classifier
independently gave them about 3×.

**13. Why should we trust the validation?**
- CV comparisons are paired on the same 5-fold × 4-repeat splits: c04 beat c03 in 4/4 repeats on every view, with an MSE CI excluding 0.
- The CV and the proxy predicted the direction of every public submission correctly.

**14. What evidence says the remaining error is noise?**
On rows without missing values, LightGBM, ExtraTrees, CatBoost and an MLP trained on out-of-fold residuals reach
out-of-sample R² below 0, and the best correction weight is 0. Group slopes, spike slopes, density weighting and blends
did not help.

**15. Why not time-series methods?**
- A row's target does not reappear as the next hour's previous_usage (chance-level matches).
- IDs are randomly ordered.
- Weather is not shared across rows, so rows can't be linked to timestamps.
- There are no dates.

Every row is independent.

**16. What exactly is previous_usage?**
A reference reading for the same time slot in an earlier period, not the previous hour. Its daily profile matches
energy at the same hour (residential correlation 0.97 vs 0.52 one hour back). It enters with a shared slope of about 0.34.

**17. Could the model fail on private-test spikes?**
Spikes larger than any labelled example are extrapolated linearly; that is supported by train spikes and assumed
beyond them. A spike hidden *inside a missing value* can't be recovered, which is the main remaining error.

**18. Main assumptions?**
- Energy is additive and linear in the inputs, with per-building occupancy and temperature slopes.
- Missing values can be estimated from the other features.
- The private test resembles the public test.

**19. Campus insights?**
- +13.4 energy units per 100 occupants in the lecture hall vs about +4.6 in residences.
- Science buildings carry the largest modelled base load (~39–43 vs 16–27 elsewhere).
- Residences step down at 08:00 and up at 21:00.
- Spikes are scheduled: previous_usage at night, occupancy in the morning.

These are associations, not causal effects.

**20. With more time or data?**
- Timestamps, to join weather and the academic calendar.
- Labelled large spikes, to verify the linear response beyond the current range.
- The actual cause of missing values (sensor faults?) to model it directly.
- Prediction intervals.
