| setting | ATE unaligned [m] | ATE global SE3 [m] | ATE per-segment SE3 [m] | note |
|---|---|---|---|---|
| baseline (update_every=5, sigma_t_rel=0.02) | 0.546 | 0.321 | 0.161 |  |
| vision-only, oracle Sim3 per segment | - | - | 0.343 | reference |
| A update_every: 1 | 0.562 | 0.436 | 0.188 |  |
| A update_every: 2 | 0.549 | 0.364 | 0.169 |  |
| A update_every: 5 | 0.546 | 0.321 | 0.161 |  |
| A update_every: 10 | 0.479 | 0.275 | 0.168 |  |
| A update_every: 20 | 0.857 | 0.509 | 0.328 |  |
| B sigma_t_rel: 0.01 | 0.474 | 0.282 | 0.149 |  |
| B sigma_t_rel: 0.02 | 0.546 | 0.321 | 0.161 |  |
| B sigma_t_rel: 0.05 | 0.637 | 0.366 | 0.194 |  |
| B sigma_t_rel: 0.1 | 0.810 | 0.458 | 0.244 |  |
| B sigma_t_rel: 0.15 | 0.989 | 0.566 | 0.287 |  |
| B sigma_t_rel: 0.3 | 1.401 | 0.839 | 0.419 |  |
| B sigma_t_rel: 0.6 | 2.123 | 1.272 | 0.705 |  |
| C imu_noise_x: 0.5 | 0.487 | 0.288 | 0.159 |  |
| C imu_noise_x: 1.0 | 0.546 | 0.321 | 0.161 |  |
| C imu_noise_x: 2.0 | 0.609 | 0.374 | 0.169 |  |
| C imu_noise_x: 4.0 | 0.679 | 0.447 | 0.182 |  |
| D vo_dropped: 30% (3 seeds) | 0.532 | 0.308 | 0.164 | std 0.004 |
| D vo_dropped: 60% (3 seeds) | 0.611 | 0.410 | 0.233 | std 0.015 |
| D vo_dropped: 90% (3 seeds) | 25.535 | 17.929 | 14.549 | std 6.997 |
| E clahe_off | 1.112 | 0.641 | 0.213 | VO-only 0.282; OK 77%; 24 segs |
| E mask_radius_180 | 0.929 | 0.564 | 0.275 | VO-only 0.568; OK 89%; 11 segs |
| E no_depth_gating | 0.498 | 0.330 | 0.132 | VO-only 0.280; OK 92%; 9 segs |
