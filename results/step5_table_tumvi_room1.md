| method | ATE RMSE [m] | ATE global SE3 [m] | RPE trans 1 s RMSE [m] | RPE rot 1 s RMSE [deg] | RPE trans 5 s RMSE [m] |
|---|---|---|---|---|---|
| vision-only (oracle Sim3/segment) | 0.343 | nan | 0.2504 | 2.650 | 0.4618 |
| fused_est_scale | 0.546 | 0.321 | 0.1101 | 2.599 | 0.1834 |
| fused_gt_scale_PLUMBING_ONLY | 0.402 | 0.243 | 0.0798 | 2.519 | 0.1119 |
| imu_only_dead_reckoning | 246.385 | 165.340 | 4.6795 | 2.531 | 23.2571 |
