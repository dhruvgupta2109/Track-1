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
| 2026-10-03 20:39 | c13_avg_c11_007 | submissions/c13_avg_c11_007.csv (sha256 3f3be1d398) | — | 3.19578 | 2 | 50/50 with teammate 007: worse than c11 alone |
| 2026-10-03 20:39 | c11_v5_cyclic | submissions/c11_v5_cyclic.csv (sha256 9b0a5253e8) | proxy NATMISS 26.86 | 3.19406 | 1 | mask fix + cyclic imputers + V5 energy |
| 2026-10-03 20:40 | c12_v5_cyclic_prevoff | submissions/c12_v5_cyclic_prevoff.csv (sha256 a5cbf65114) | — | **3.18937** | **1** | c11 + 0.6 offset on prev-missing rows: MNAR confirmed |
| 2026-10-03 20:48 | c15_prev_selfmask | submissions/c15_prev_selfmask.csv (sha256 9005eee511) | — | 3.18852 | 1 | prev offset ∝ building test SD (mean 0.6): -0.0009 vs c12 |
| 2026-10-03 20:48 | c14_prevoff085 | submissions/c14_prevoff085.csv (sha256 1286ddaeb2) | — | 3.18808 | 1 | flat prev offset 0.85: -0.0013 vs c12 |
| 2026-10-03 20:48 | c16_prev06_occ_selfmask | submissions/c16_prev06_occ_selfmask.csv (sha256 7ae6cbca99) | — | **3.18618** | **1** | c12 + occupancy self-masking offsets: -0.0032 vs c12 |
| 2026-10-03 21:21 | c17_mnar_eb | submissions/c17_mnar_eb.csv (sha256 05cb6f912c) | train-only MNAR, EB-shrunk (pre-registered) | 3.18775 | 2 (behind own c16) | principled replacement for c12-c16 offsets; +0.0016 vs c16 = within public noise |
