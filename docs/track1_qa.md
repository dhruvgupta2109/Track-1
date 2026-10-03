# Track 1 — Q&A preparation

Numbers come from `experiments/log.jsonl`, `experiments/NOTES.md` and the analysis notebook. RMSE values are means over CV repeats.

| | Clean CV | Test-like | Shift-weighted | Spike rows | Public |
|---|---|---|---|---|---|
| c00 LightGBM (5x2) | 3.341 | 3.964 | 4.281 | 5.967 | — |
| c03 linear | 3.077 | 3.296 | 3.543 | 4.875 | 3.36875 |
| **c04 final** | **3.048** | **3.228** | **3.363** | **4.150** | **3.33681** |

---

**1. Why linear regression instead of LightGBM?**
The test set has about four times train's rate of extreme occupancy and `previous_usage` values, and many lie outside the
training range. Trees cannot go past their largest leaf value, so they flatten there. A linear model keeps its slope.
On our test-like validation LightGBM scores 3.96 against 3.23 for c04. We still use LightGBM, but only to fill in missing inputs.

**2. How did you detect the train/test shift?**
Three independent ways:
- Missing rates: 18.2% of test rows have a missing value vs 5.8% of train.
- A target-free consistency model that predicts each feature from the others: 16.6% of test rows have a spike (|z|>3) vs 4.1% of train.
- A train-vs-test classifier on feature values: AUC 0.70, driven by `previous_usage`, occupancy and temperature.

**3. Why are the spikes not treated as outliers?**
On labelled train rows, energy follows the spike. Predict energy without the spiked feature, then compare the
unexplained energy with the feature's deviation: correlation 0.92 for `previous_usage` and 0.80 for occupancy. The
spike rows sit on the same line as ordinary rows. An extreme value is not a corrupted value.

**4. Why does linear extrapolation help?**
The marginal effect of a spike matches that of ordinary variation. A separate slope for spiked values came out at about 0
(prev −0.03 next to its 0.34 slope; occupancy ~0). A straight line is therefore the right extension, while trees
under-predict large spikes. c04's error on spike rows is 4.15, against 5.97 for LightGBM.

**5. Why use LightGBM if the final model is linear?**
Only to *estimate missing inputs* (occupancy, previous_usage, temperature, humidity) from the other features.
Predicting a feature like occupancy from building × hour × weekday is a lookup-table problem where trees do well
(occupancy R² 0.915 vs 0.865 for a linear design). The energy equation stays linear.

**6. How does c04 handle missing values?**
- Every missing numeric input is filled by its LightGBM imputer.
- The full linear energy model then predicts.
- Missing or unseen categories (building, hour, day, month) enter as averages over the known levels.
- If an imputed value were invalid (e.g. negative occupancy), that row falls back to a per-pattern sub-model, so nothing crashes.

**7. Why not separate models for each missingness pattern?** That was c03, and error analysis showed the flaw.
When occupancy is missing, the sub-model's `previous_usage` slope inflates from 0.34 to 0.46 because prev absorbs
part of the occupancy effect. That inflated slope then mis-extrapolates on prev spikes. Imputing first keeps the
correct slopes. c04 beats c03 on test-like CV by −0.068, wins 4/4 repeats, and has a test-like MSE CI of [−0.67, −0.25].

**8. How did you avoid leakage when training imputers?**
- The imputers never see `energy_usage`; they predict features from features.
- Inside CV, each fold's imputers are fitted without that fold's validation rows.
- All features are row-local; nothing depends on ids or other rows.

**9. Why are public-test features used in imputer fitting?**
They show the feature distribution the model will face (more spikes, more co-missingness). We checked the risk
directly on held-out rows. The train+test imputers are as good as train-only imputers on train-like rows
(energy-weighted imputation MSE 14.36 vs 14.44) and much better on test-like rows (27.4 vs 35.3).

**10. Is using unlabelled test features legitimate?**
We believe so. No test labels are used; it is standard transductive / semi-supervised use of features. The booklet
explicitly allows additional or public datasets and does not prohibit unlabelled test features. The deployed notebook
needs only the uploaded model file. If judges disagree, the train-only-imputer variant is ready (test-like CV 3.233 vs 3.228).

**11. What does shift-weighted validation mean?**
Test-like CV with train spike rows up-weighted ×4.05, so spikes count as often as they appear in test (16.6% vs 4.1%).
The train-vs-test classifier independently gave spike rows about 3× the weight of ordinary rows.

**12. Why should we trust the validation?**
- Every comparison is paired on the same 5-fold × 4-repeat splits, and c04 wins every repeat on every view.
- It predicted the public result's direction: c03 3.369 → c04 3.337.
- The ordering clean < test-like < public < shift-weighted is consistent for both submitted models.

**13. What evidence says the remaining error is noise?**
LightGBM, ExtraTrees, CatBoost and an MLP trained on c04's out-of-fold residuals all score out-of-sample
R² below 0 (−0.019, −0.011, −0.005, −0.004), and the best correction weight is 0. Residuals are near-Gaussian
(kurtosis 0.16), with no bias by building, hour, month or ID range.

**14. Why not time-series methods?**
There is no sequence to model:
- A row's target does not reappear as the next hour's `previous_usage`: 0.5% exact matches, the same as the 0.3–0.7% at control offsets.
- Consecutive IDs share a building at chance rate (8.1% vs 8.5%).
- There are no dates or years.

The rows behave as independent samples.

**15. What exactly is `previous_usage`?**
A reference reading for the *same time slot in an earlier period*, not the previous hour. Its daily profile matches
energy at the same hour: residential correlation 0.97, against 0.52 for one hour back. It acts as a noisy copy of
current usage with a slope of about 0.34.

**16. Could the model fail on private-test spikes?**
Spikes larger than any labelled example are extrapolated linearly. That is supported up to the largest train spikes
and is an assumption beyond them. A spike in a feature that is *missing* cannot be recovered, so those rows keep a
higher error. Those are the main private-test risks.

**17. What are the main model assumptions?**
- Energy is additive and linear in occupancy, previous_usage, temperature and humidity.
- Building-specific occupancy and temperature slopes; shared prev and humidity slopes.
- A smooth daily profile per building type with small hour corrections.
- Missing values occur roughly at random given the other features.

**18. What are the main limitations?**
- Unverifiable extrapolation beyond train spikes.
- Irreducible noise of about 3.0 RMSE on complete rows.
- No dates, so no external weather or calendar data.
- Coefficients are associations, not causal effects.

**19. What campus insight did the model produce?**
- Occupancy sensitivity differs about 3×: +13.4 energy units per 100 occupants in the lecture hall vs about +4.6 in residences.
- Science buildings carry the largest modelled base load (~39–43 vs 16–27 elsewhere).
- Residences show schedule steps at 08:00 and 21:00.
- Extreme prev values cluster late at night and occupancy surges in the morning.

**20. What would you improve with another week or more data?**
- Timestamps, to join real weather and the academic calendar and to model true sequences.
- Labelled examples of large spikes, to verify the linear response beyond the current range.
- Prediction intervals, since noise grows with building load.
- Monitoring of the residual against model expectation to detect faults.
