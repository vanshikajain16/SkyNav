# SkyNav: monocular visual-inertial odometry + detection/tracking (Python, native Windows)

A from-scratch Python pipeline: monocular visual odometry front end with a sparse map and PnP, an error-state
EKF that fuses IMU and VO relative poses and **estimates the monocular scale**, a YOLOX-nano ONNX detector with a
SORT-style tracker, and an evaluation suite (ATE, RPE, `evo` cross-check). No ROS, no simulator, no GPU required.
All numbers below are produced by the scripts in this repository (see `results/`).

## Scope and honest limitations (read first)

* **One real sequence:** TUM VI `room1` (512x512 equidistant fisheye, 20 Hz camera, 200 Hz IMU, mocap ground truth,
  2613 frames compared). The filter hyperparameter `sigma_t_rel` was tuned on this same
  sequence, so the fused numbers are **not** a held-out result. A second sequence (room4) and EuRoC have **not** been run yet.
* **Ground-truth initialisation:** the filter starts from the ground-truth pose and velocity at the first tracked frame.
* **Vision-only numbers are generous:** monocular VO has an arbitrary scale and frame, so it is scored with an
  *oracle* Sim(3) alignment per map segment (8 segments). Fused numbers are also reported without any alignment.
* **No loop closure, no bundle adjustment:** position and yaw drift (relative measurements only).
* **The target in the integration test is synthetic** (a projected point with detector-like noise). The detector was
  not evaluated on real targets: room1 contains no people or vehicles.
* **Tracker metrics are on synthetic scenes** generated in this repository.
* The "GT-scaled VO" run gives the filter the true translation magnitude. It only validates plumbing and is never a result.
* Not built: ROS 2 wrapper, a target-following controller, PX4/Gazebo (dropped from scope).

## Results (TUM VI room1)

### Trajectory accuracy (position error against ground truth)

| method | ATE RMSE [m] | ATE after one global SE(3) alignment [m] | RPE 1 s trans RMSE [m] | RPE 5 s trans RMSE [m] |
|---|---|---|---|---|
| vision-only, oracle Sim(3) per segment | 0.343 | n/a | 0.250 | 0.462 |
| **fused, scale estimated by the filter** | **0.546** (no alignment) | 0.321 | 0.110 | 0.183 |
| fused, GT-scaled VO (plumbing check only) | 0.402 | 0.243 | 0.080 | 0.112 |
| IMU-only dead reckoning | 246.385 | 165.340 | 4.679 | 23.257 |

* Cross-check with `evo` (fused, estimated scale): ATE 0.324 m, RPE 1 s 0.110 m.
* Comparing like with like (per-segment alignment, fused uses rigid SE(3) only, VO gets scale too):
  fused 0.161 m vs vision-only 0.343 m.
* The scale state agrees with the per-segment reference to a median relative error of
  13.6% over 2453 frames
  (the reference is itself a constant fit per segment, so this is a loose yardstick).
* Rotation RPE (1 s) is dominated by outliers shared by all methods including IMU-only (RMSE 2.60 deg,
  median 0.49 deg); this is likely ground-truth glitches, not verified.

### Visual odometry front end

* LK front end (`vo/frontend_lk.py`, used for every fused result above): 59 frames/s on CPU including PNG loading;
  92.6% of frames tracked; 8 map segments (the map is re-initialised after tracking loss, each segment has its own scale).
* Descriptor front end (`vo/frontend.py`, ORB default, SIFT switch; central fisheye mask, essential-matrix bootstrap,
  PnP against a triangulated map). Vision-only, Sim(3) per segment, same sequence:

| front end | frames tracked | map segments | ATE RMSE [m] | speed [fps] |
|---|---|---|---|---|
| ORB | 75.3% | 40 | 0.433 | 14.9 |
| SIFT | 93.6% | 5 | 0.164 | 15.4 |

  SIFT is clearly better than ORB on this dark fisheye sequence. The fused (filter) results have not been re-run with it.
  The `T_cam_imu` direction check is weak: its rotation is 179 degrees (its own inverse), so only the
  lever arm separates the two conventions; hypothesis A (maps IMU points into the camera frame) wins
  14 of 20 pairs, while trajectory-level ATE is indistinguishable.

### Ablations and robustness (fused, estimated scale; table in `results/step8_experiments_tumvi_room1.md`)

| setting | ATE unaligned [m] | ATE global SE3 [m] |
|---|---|---|
| baseline | 0.546 | 0.321 |
| A update_every:1 | 0.562 | 0.436 |
| A update_every:2 | 0.549 | 0.364 |
| A update_every:5 | 0.546 | 0.321 |
| A update_every:10 | 0.479 | 0.275 |
| A update_every:20 | 0.857 | 0.509 |
| B sigma_t_rel:0.01 | 0.474 | 0.282 |
| B sigma_t_rel:0.02 | 0.546 | 0.321 |
| B sigma_t_rel:0.05 | 0.637 | 0.366 |
| B sigma_t_rel:0.1 | 0.810 | 0.458 |
| B sigma_t_rel:0.15 | 0.989 | 0.566 |
| B sigma_t_rel:0.3 | 1.401 | 0.839 |
| B sigma_t_rel:0.6 | 2.123 | 1.272 |
| C imu_noise_x:0.5 | 0.487 | 0.288 |
| C imu_noise_x:1.0 | 0.546 | 0.321 |
| C imu_noise_x:2.0 | 0.609 | 0.374 |
| C imu_noise_x:4.0 | 0.679 | 0.447 |
| D vo_dropped:0.3 | 0.532 | 0.308 |
| D vo_dropped:0.6 | 0.611 | 0.410 |
| D vo_dropped:0.9 | 25.535 | 17.929 |
| E clahe_off | 1.112 | 0.641 |
| E mask_radius_180 | 0.929 | 0.564 |
| E no_depth_gating | 0.498 | 0.330 |

* With 90% of VO frames removed the filter fails (17.9 m global SE(3) ATE): the
  pipeline needs the vision updates; the IMU alone drifts to 165 m.
* Front-end ablations were run on that variant's own tracked frames, so they are not exactly comparable to the baseline row.

### Detection and tracking

* Detector: YOLOX-nano (ONNX Runtime, CPU), 14.9 ms/frame on 512x512 room1 images. Detections on
  room1 are incidental objects (clock, tv, chair); no quantitative detector accuracy is claimed.
* Tracker on synthetic scenes (5 targets, 600 frames): clean MOTA 0.940, 0 identity switches;
  hard (25% misses, noisier boxes, more false positives) MOTA 0.747, 3 identity switches.

### Integration: tracked target to world position (synthetic target, real flight)

* Rotation-compensated tracking (using the fused orientation) reports the target on
  81% of visible frames with 34 identity switches,
  versus 68% and 59 without compensation.
* World-position error of the target (ray against a known ground plane): 0.045 m RMSE with the
  true camera pose (detector noise only), 0.607 m RMSE with the fused pose.

## Layout

```
vo/          datasets.py (loaders, camera models), frontend.py (ORB/SIFT VO), frontend_lk.py (LK VO), run_vo.py
vio/         eskf.py (error-state EKF), run_vio.py (IMU + VO driver, modes)
perception/  detector.py (YOLOX ONNX), tracker.py (SORT), localize.py, check_perception.py
eval/        ate.py, metrics.py, evaluate.py, vio_report.py, integration.py, experiments.py, make_readme.py
data/        tumvi/room1 (extracted), models/yolox_nano.onnx    (not versioned)
results/     metrics JSON, tables, plots, TUM trajectories
```

## Reproduce (PowerShell, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m eval.check_dataset          # Step 2 checks, plots
python -m vo.run_vo                   # VO trajectory -> results\vo_tumvi_room1.npz
python -m eval.vio_report             # Step 4: fused vs vision-only, scale plot
python -m eval.evaluate               # Step 5: ATE/RPE table + evo cross-check
python -m perception.check_perception # Step 6
python -m eval.integration            # Step 7
python -m eval.experiments            # Step 8
python -m eval.make_readme            # regenerate this file
```

Data: download `dataset-room1_512_16.tar` (1.6 GB, EuRoC/DSO export) from the TUM VI dataset page and extract
`mav0/cam0`, `mav0/imu0`, `mav0/mocap0` and `dso/{camchain.yaml,imu_config.yaml,gt_imu.csv}` into `data\tumvi\room1\`.
Model: `yolox_nano.onnx` from the YOLOX 0.1.1rc0 release into `data\models\`.

## Method notes

* **Camera model:** read from `camchain.yaml` (equidistant, intrinsics `[fx, fy, cx, cy]`); the loader switches
  between equidistant and radial-tangential through the dataset spec. 16-bit images are shifted to 8-bit
  (values are 12-bit data stored in 16 bits).
* **VO:** LK tracking with forward-backward check, central disc mask against the fisheye rim, essential-matrix
  initialisation (|t| = 1 defines one VO unit), PnP-RANSAC against the map, triangulation of new points with
  parallax and depth-ratio checks, constant-velocity pose guess.
* **Filter:** 16-state error-state EKF (position, velocity, orientation, two biases, scale) plus a 6-D stochastic
  clone of the pose at the last update. Relative VO pose (rotation + scaled translation) every 5 tracked frames;
  noise is inflated rather than rejected beyond a chi-square gate; scale is initialised per map segment from the
  IMU-predicted displacement over the first VO pairs.
