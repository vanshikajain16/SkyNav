"""Step 9: render README.md from the metric files in results/.

Every number printed in those documents is read from results/*.json or results/vo_*.npz here, never typed by hand.
"""
from __future__ import annotations                                   # forward type hints
import json                                                          # metric files
import numpy as np                                                   # npz access
from vo.datasets import PROJECT_ROOT                                 # project folder


def main(name: str = "tumvi_room1") -> None:                         # render documents
    r = PROJECT_ROOT / "results"                                     # results folder
    j = lambda f: json.load(open(r / f, encoding="utf-8"))           # json loader
    m4, m5, m6, m7, m8 = (j(f"metrics_step4_{name}.json"), j(f"metrics_step5_{name}.json"), j("metrics_step6.json"),
                          j(f"metrics_step7_{name}.json"), j(f"metrics_step8_{name}.json"))   # metric sets
    o3, s3 = j(f"metrics_step3_orb_{name}.json"), j(f"metrics_step3_sift_{name}.json")   # descriptor front-end metrics
    vo = np.load(r / f"vo_{name}.npz")                               # VO output for speed and coverage
    fps = float(vo["fps"])                                           # VO speed incl. image loading
    ok_pct = 100.0 * float((vo["status"] == 0).mean())               # tracked frame percentage
    n_seg = int(vo["map_id"].max()) + 1                              # map segments
    est, gts, imu, vis = m5["fused_est_scale"], m5["fused_gt_scale_PLUMBING_ONLY"], m5["imu_only_dead_reckoning"], m5["vision_only"]   # rows
    evo = m5["evo_cross_check_fused_est_scale"]                      # evo numbers
    sc, sh = m6["synthetic_clean_tracker"], m6["synthetic_hard_tracker"]   # tracker metrics
    det = m6["real_room1_detector"]["raw"]                           # detector timing
    t_n, t_f = m7["none"], m7["fused"]                               # target localisation cases
    drop = {k: m8[k] for k in m8 if k.startswith("D vo_dropped")}    # outage rows
    f3 = lambda x: f"{x:.3f}"                                        # format helper

    readme = f"""# SkyNav: monocular visual-inertial odometry + detection/tracking (Python, native Windows)

A from-scratch Python pipeline: monocular visual odometry front end with a sparse map and PnP, an error-state
EKF that fuses IMU and VO relative poses and **estimates the monocular scale**, a YOLOX-nano ONNX detector with a
SORT-style tracker, and an evaluation suite (ATE, RPE, `evo` cross-check). No ROS, no simulator, no GPU required.
All numbers below are produced by the scripts in this repository (see `results/`).

## Scope and honest limitations (read first)

* **One real sequence:** TUM VI `room1` (512x512 equidistant fisheye, 20 Hz camera, 200 Hz IMU, mocap ground truth,
  {m5['frames_compared']} frames compared). The filter hyperparameter `sigma_t_rel` was tuned on this same
  sequence, so the fused numbers are **not** a held-out result. A second sequence (room4) and EuRoC have **not** been run yet.
* **Ground-truth initialisation:** the filter starts from the ground-truth pose and velocity at the first tracked frame.
* **Vision-only numbers are generous:** monocular VO has an arbitrary scale and frame, so it is scored with an
  *oracle* Sim(3) alignment per map segment ({n_seg} segments). Fused numbers are also reported without any alignment.
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
| vision-only, oracle Sim(3) per segment | {f3(vis['ate_rmse_m'])} | n/a | {f3(vis['rpe_trans_1s']['rmse'])} | {f3(vis['rpe_trans_5s']['rmse'])} |
| **fused, scale estimated by the filter** | **{f3(est['ate_unaligned_rmse_m'])}** (no alignment) | {f3(est['ate_global_se3_rmse_m'])} | {f3(est['rpe_trans_1s']['rmse'])} | {f3(est['rpe_trans_5s']['rmse'])} |
| fused, GT-scaled VO (plumbing check only) | {f3(gts['ate_unaligned_rmse_m'])} | {f3(gts['ate_global_se3_rmse_m'])} | {f3(gts['rpe_trans_1s']['rmse'])} | {f3(gts['rpe_trans_5s']['rmse'])} |
| IMU-only dead reckoning | {f3(imu['ate_unaligned_rmse_m'])} | {f3(imu['ate_global_se3_rmse_m'])} | {f3(imu['rpe_trans_1s']['rmse'])} | {f3(imu['rpe_trans_5s']['rmse'])} |

* Cross-check with `evo` (fused, estimated scale): ATE {evo['evo_ape_global_se3_rmse_m']:.3f} m, RPE 1 s {evo['evo_rpe_1s_trans_rmse_m']:.3f} m.
* Comparing like with like (per-segment alignment, fused uses rigid SE(3) only, VO gets scale too):
  fused {f3(m4['estscale']['per_segment_se3_rmse_m'])} m vs vision-only {f3(m4['vision_only_sim3_per_segment_rmse_m'])} m.
* The scale state agrees with the per-segment reference to a median relative error of
  {100 * m4['scale_median_abs_rel_error_vs_segment_reference']:.1f}% over {m4['scale_frames_evaluated']} frames
  (the reference is itself a constant fit per segment, so this is a loose yardstick).
* Rotation RPE (1 s) is dominated by outliers shared by all methods including IMU-only (RMSE {est['rpe_rot_deg_1s']['rmse']:.2f} deg,
  median {est['rpe_rot_deg_1s']['median']:.2f} deg); this is likely ground-truth glitches, not verified.

### Visual odometry front end

* LK front end (`vo/frontend_lk.py`, used for every fused result above): {fps:.0f} frames/s on CPU including PNG loading;
  {ok_pct:.1f}% of frames tracked; {n_seg} map segments (the map is re-initialised after tracking loss, each segment has its own scale).
* Descriptor front end (`vo/frontend.py`, ORB default, SIFT switch; central fisheye mask, essential-matrix bootstrap,
  PnP against a triangulated map). Vision-only, Sim(3) per segment, same sequence:

| front end | frames tracked | map segments | ATE RMSE [m] | speed [fps] |
|---|---|---|---|---|
| ORB | {o3['ok_percent']:.1f}% | {o3['map_segments_used']} | {f3(o3['ate_rmse_sim3_per_segment_m'])} | {o3['fps']:.1f} |
| SIFT | {s3['ok_percent']:.1f}% | {s3['map_segments_used']} | {f3(s3['ate_rmse_sim3_per_segment_m'])} | {s3['fps']:.1f} |

  SIFT is clearly better than ORB on this dark fisheye sequence. The fused (filter) results have not been re-run with it.
  The `T_cam_imu` direction check is weak: its rotation is {s3['T_cam_imu_check']['Rc_rotation_angle_deg']:.0f} degrees (its own inverse), so only the
  lever arm separates the two conventions; hypothesis A (maps IMU points into the camera frame) wins
  {s3['T_cam_imu_check']['pairs_won_by_A']} of {s3['T_cam_imu_check']['pairs']} pairs, while trajectory-level ATE is indistinguishable.

### Ablations and robustness (fused, estimated scale; table in `results/step8_experiments_{name}.md`)

| setting | ATE unaligned [m] | ATE global SE3 [m] |
|---|---|---|
""" + "\n".join(
        f"| {k} | {v['unaligned_m']:.3f} | {v['global_se3_m']:.3f} |" for k, v in m8.items()
        if isinstance(v, dict) and "unaligned_m" in v) + f"""

* With 90% of VO frames removed the filter fails ({drop['D vo_dropped:0.9']['global_se3_m']:.1f} m global SE(3) ATE): the
  pipeline needs the vision updates; the IMU alone drifts to {imu['ate_global_se3_rmse_m']:.0f} m.
* Front-end ablations were run on that variant's own tracked frames, so they are not exactly comparable to the baseline row.

### Detection and tracking

* Detector: YOLOX-nano (ONNX Runtime, CPU), {det['ms_per_frame']:.1f} ms/frame on 512x512 room1 images. Detections on
  room1 are incidental objects (clock, tv, chair); no quantitative detector accuracy is claimed.
* Tracker on synthetic scenes (5 targets, 600 frames): clean MOTA {sc['MOTA']:.3f}, {sc['IDSW']} identity switches;
  hard (25% misses, noisier boxes, more false positives) MOTA {sh['MOTA']:.3f}, {sh['IDSW']} identity switches.

### Integration: tracked target to world position (synthetic target, real flight)

* Rotation-compensated tracking (using the fused orientation) reports the target on
  {100 * t_f['target_reported_fraction']:.0f}% of visible frames with {t_f['identity_switches_on_target']} identity switches,
  versus {100 * t_n['target_reported_fraction']:.0f}% and {t_n['identity_switches_on_target']} without compensation.
* World-position error of the target (ray against a known ground plane): {t_f['gt_pose']['rmse_m']:.3f} m RMSE with the
  true camera pose (detector noise only), {t_f['fused_est_scale']['rmse_m']:.3f} m RMSE with the fused pose.

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
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python -m eval.check_dataset          # Step 2 checks, plots
python -m vo.run_vo                   # VO trajectory -> results\\vo_tumvi_room1.npz
python -m eval.vio_report             # Step 4: fused vs vision-only, scale plot
python -m eval.evaluate               # Step 5: ATE/RPE table + evo cross-check
python -m perception.check_perception # Step 6
python -m eval.integration            # Step 7
python -m eval.experiments            # Step 8
python -m eval.make_readme            # regenerate this file
```

Data: download `dataset-room1_512_16.tar` (1.6 GB, EuRoC/DSO export) from the TUM VI dataset page and extract
`mav0/cam0`, `mav0/imu0`, `mav0/mocap0` and `dso/{{camchain.yaml,imu_config.yaml,gt_imu.csv}}` into `data\\tumvi\\room1\\`.
Model: `yolox_nano.onnx` from the YOLOX 0.1.1rc0 release into `data\\models\\`.

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
"""
    (PROJECT_ROOT / "README.md").write_text(readme, encoding="utf-8")   # write README
    print("wrote README.md")                                         # done


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
