# Public leaderboard log (Track 1, RMSE)
| when | candidate | file | local clean / test-like / shiftW | public RMSE | rank | notes |
|---|---|---|---|---|---|---|
| 2026-10-03 | c03_linear_v3 | submissions/c03_linear_v3.csv | 3.077 / 3.297 / 3.544 | 3.36875 | 1/5 (#2 3.37652) | first LB submission; public sits between test-like and shiftW |
| 2026-10-03 | c04_linear_imputed | submissions/c04_linear_imputed.csv | 3.048 / 3.228 / 3.363 | 3.33681 | 1/6 (#2 3.37652) | -0.0319 vs c03 public (local test-like predicted -0.068; ~47% transferred) |
| 2026-10-03 17:28 | c06_domain_ens | submissions/c06_domain_ens.csv | test-row imputation proxy 23.65 | 3.23804 | 2 | domain-aware imputer trio |
| 2026-10-03 17:30 | c07_domain_ens_testw2 | submissions/c07_domain_ens_testw2.csv | proxy 23.57 | **3.23149** | **1** | + test rows x2 in imputer fit |
| 2026-10-03 17:34 | c05_domain_imputed | submissions/c05_domain_imputed.csv | proxy 24.06 | 3.25549 | 1 | single domain-aware imputer |
| 2026-10-03 17:46 | c08_domain_ens_aug5 | submissions/c08_domain_ens_aug5.csv | proxy 22.21 | **3.20937** | **1** | + mask augmentation x5 |
| 2026-10-03 17:56 | c09_dev_imputed | submissions/c09_dev_imputed.csv | proxy 20.87 | **3.20392** | **1** | + deviation features for imputers; proxy -1.34 transferred only -0.0055 |
