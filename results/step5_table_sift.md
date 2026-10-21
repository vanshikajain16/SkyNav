Front end: sift. ATE = position RMSE [m]; RPE = relative pose error over 1 s (20 frames) / 5 s (100 frames), identical frame pairs for vision-only and fused.

| sequence | method | alignment | ATE [m] | RPE trans 1 s [m] | RPE trans 5 s [m] | RPE rot 1 s RMSE / median [deg] |
|---|---|---|---|---|---|---|
| tumvi_room1 | vision-only | oracle Sim(3) per segment | 0.164 | 0.127 | 0.211 | 2.25 / 0.67 |
| tumvi_room1 | fused (est. scale) | none | 0.449 | 0.080 | 0.124 | 1.86 / 0.46 |
| tumvi_room1 | fused (est. scale) | one global SE(3) | 0.245 | same | same | same |
| tumvi_room1 | fused (est. scale) | SE(3) per segment (like-for-like) | 0.117 | same | same | same |
| tumvi_room1 | IMU only | none / global SE(3) | 246.9 / 166.2 | - | - | - |
| tumvi_room4 | vision-only | oracle Sim(3) per segment | 0.038 | 0.036 | 0.053 | 1.17 / 0.56 |
| tumvi_room4 | fused (est. scale) | none | 0.470 | 0.092 | 0.077 | 0.91 / 0.39 |
| tumvi_room4 | fused (est. scale) | one global SE(3) | 0.275 | same | same | same |
| tumvi_room4 | fused (est. scale) | SE(3) per segment (like-for-like) | 0.063 | same | same | same |
| tumvi_room4 | IMU only | none / global SE(3) | 234.7 / 164.4 | - | - | - |

| sequence | frames | scored | VO tracked | VO segments |
|---|---|---|---|---|
| tumvi_room1 | 2821 | 2640 | 93.6% | 5 |
| tumvi_room4 | 2228 | 1935 | 87.4% | 11 |
