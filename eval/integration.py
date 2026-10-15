"""Step 7: perception + fused pose -> world-frame target estimate, on a synthetic target in the real room1 flight.

The camera motion, calibration, IMU and VO are real (TUM VI room1). The TARGET is synthetic: a point moving on a
circle at a fixed height, projected through the real camera model at the true camera pose, with detector-like noise.
Compares tracking with and without ego-motion (rotation) compensation of the tracks.
"""
from __future__ import annotations                                   # forward type hints
import json                                                          # metrics file
import numpy as np                                                   # arrays
import matplotlib                                                    # backend selection
matplotlib.use("Agg")                                                # no GUI
import matplotlib.pyplot as plt                                      # plotting

from vo.datasets import Sequence, PROJECT_ROOT                       # data access
from vio.run_vio import run_filter                                   # filter driver
from perception.tracker import SortTracker, TrackerConfig, Track     # tracker
from perception.localize import pixel_to_world_on_plane              # localisation

PLANE_Z = 0.3                                                        # target height above the mocap floor [m]
TARGET_W = 0.30                                                      # target width [m] (for the synthetic box size)
R_VIS = 225.0                                                        # usable image disc radius [px]


def target_world(t_s: np.ndarray) -> np.ndarray:                     # synthetic target path (circle, 0.5 m/s)
    r, w = 1.2, 0.5 / 1.2                                            # radius and angular speed
    return np.stack([r * np.cos(w * t_s), r * np.sin(w * t_s), np.full_like(t_s, PLANE_Z)], axis=1)   # (N,3)


def make_warp(cam, R_prev_cur):                                      # pixel warp for a pure camera rotation
    def warp(px):                                                    # px in the previous image
        n = cam.undistort_to_normalized(np.asarray(px, float).reshape(1, 2))[0]   # normalized ray
        r = R_prev_cur @ np.array([n[0], n[1], 1.0])                 # ray expressed in the current camera
        if r[2] <= 1e-3:                                             # behind the camera
            return None                                              # no warp
        return cam.project((r / r[2]).reshape(1, 3))[0]              # back to pixels
    return warp                                                      # callable


def run_case(seq, vo, runs, comp: str, seed: int = 1) -> dict:       # one tracking configuration
    rng = np.random.default_rng(seed)                                # same noise for every case
    gt_cam, valid = runs["estscale"]["gt_cam"], runs["estscale"]["valid"]   # true poses, validity
    t = vo["t_ns"]                                                   # frame times
    ts = (t - t[0]) * 1e-9                                           # seconds
    X = target_world(ts)                                             # true target positions
    cam, fx = seq.camera, float(seq.calib.K[0, 0])                   # camera model and focal length
    cx, cy = seq.calib.K[0, 2], seq.calib.K[1, 2]                    # principal point
    sources = {"gt_pose": gt_cam, "fused_est_scale": runs["estscale"]["T_w_c"], "imu_only": runs["imuonly"]["T_w_c"]}   # pose sources
    rot_src = {"fused": runs["estscale"]["T_w_c"], "gt": gt_cam}.get(comp)   # rotation used for compensation
    Track._next_id = 1                                               # reset ids
    tracker = SortTracker(TrackerConfig(r_meas=3.0))                 # tracker
    est = {k: np.full((len(t), 3), np.nan) for k in sources}         # estimated target positions
    visible, reported = np.zeros(len(t), bool), np.zeros(len(t), bool)   # flags
    tid_main, n_idsw, prev_k = None, 0, None                         # identity bookkeeping
    for k in range(len(t)):                                          # frame loop
        if not valid[k]:                                             # no filter state yet
            continue                                                 # skip
        if rot_src is not None and prev_k is not None:               # ego-motion compensation
            R_pc = rot_src[k][:3, :3].T @ rot_src[prev_k][:3, :3]    # previous camera frame -> current camera frame
            tracker.compensate(make_warp(cam, R_pc))                 # shift tracks
        prev_k = k                                                   # remember
        Xc = np.linalg.inv(gt_cam[k]) @ np.append(X[k], 1.0)         # target in the true camera frame
        dets, px_true = [], None                                     # detections, true pixel
        if Xc[2] > 0.2:                                              # in front of the camera
            px = cam.project(Xc[None, :3])[0]                        # true pixel
            if np.hypot(px[0] - cx, px[1] - cy) < R_VIS:             # inside the usable disc
                visible[k], px_true = True, px                       # visible
                if rng.random() > 0.10:                              # 10% missed detections
                    wpx = fx * TARGET_W / Xc[2]                      # apparent width
                    c = px + rng.normal(0, 2.0, 2)                   # centre noise 2 px
                    dets.append([c[0] - wpx / 2, c[1] - wpx / 2, c[0] + wpx / 2, c[1] + wpx / 2, 0.8, 0])   # box
        for _ in range(rng.poisson(0.3)):                            # a few false positives
            c = rng.uniform(100, 400, 2)                             # random place
            dets.append([c[0] - 15, c[1] - 15, c[0] + 15, c[1] + 15, 0.4, 0])   # box
        out = tracker.update(np.array(dets, dtype=np.float64).reshape(-1, 6))   # track
        if px_true is None or len(out) == 0:                         # nothing to localise
            continue                                                 # skip
        d = np.hypot((out[:, 1] + out[:, 3]) / 2 - px_true[0], (out[:, 2] + out[:, 4]) / 2 - px_true[1])   # distance to truth
        j = int(np.argmin(d))                                        # nearest reported track
        if d[j] > 15:                                                # not the target
            continue                                                 # skip
        reported[k] = True                                           # target reported
        if tid_main is not None and int(out[j, 0]) != tid_main:      # identity switched
            n_idsw += 1                                              # count
        tid_main = int(out[j, 0])                                    # follow
        pc = np.array([(out[j, 1] + out[j, 3]) / 2, (out[j, 2] + out[j, 4]) / 2])   # tracked centre
        for src, Tw in sources.items():                              # localise with each pose source
            p = pixel_to_world_on_plane(pc, cam, Tw[k], PLANE_Z)     # ray-plane intersection
            if p is not None:                                        # valid
                est[src][k] = p                                      # store
    m = {"compensation": comp, "frames": int(valid.sum()), "target_visible_frames": int(visible.sum()),
         "target_reported_frames": int(reported.sum()), "target_reported_fraction": float(reported.sum() / max(visible.sum(), 1)),
         "identity_switches_on_target": n_idsw}                      # coverage
    for src in sources:                                              # errors per pose source
        ok = np.isfinite(est[src][:, 0]) & reported                  # frames with an estimate
        e = np.linalg.norm(est[src][ok] - X[ok], axis=1)             # error [m]
        m[src] = {"n": int(ok.sum()), "rmse_m": float(np.sqrt(np.mean(e ** 2))), "median_m": float(np.median(e)),
                  "p95_m": float(np.percentile(e, 95))}              # statistics
    m["_arrays"] = (est, reported, X, ts)                            # for plotting
    return m                                                         # metrics


def main(name: str = "tumvi_room1") -> dict:                         # run all cases
    res = PROJECT_ROOT / "results"                                   # output folder
    seq = Sequence(name)                                             # data
    vo = dict(np.load(res / f"vo_{name}.npz"))                       # VO output
    runs = {m: run_filter(seq, vo, m, verbose=False) for m in ("estscale", "imuonly")}   # pose sources
    cases = {c: run_case(seq, vo, runs, c) for c in ("none", "fused", "gt")}   # tracking variants
    summary = {c: {k: v for k, v in r.items() if k != "_arrays"} for c, r in cases.items()}   # drop arrays
    with open(res / f"metrics_step7_{name}.json", "w", encoding="utf-8") as f:   # save
        json.dump(summary, f, indent=2)                              # write
    for c, r in summary.items():                                     # print compact table
        print(f"compensation={c:5s} visible {r['target_visible_frames']}  reported {r['target_reported_frames']} "
              f"({100 * r['target_reported_fraction']:.0f}%)  id switches {r['identity_switches_on_target']}  "
              f"target err RMSE gt-pose {r['gt_pose']['rmse_m']:.3f} m | fused-pose {r['fused_est_scale']['rmse_m']:.3f} m "
              f"(median {r['fused_est_scale']['median_m']:.3f}) | imu-only {r['imu_only']['rmse_m']:.3f} m (n={r['imu_only']['n']})")   # line
    est, reported, X, ts = cases["fused"]["_arrays"]                 # plot the compensated case
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))                    # figure
    ax[0].plot(X[reported, 0], X[reported, 1], "k.", ms=2, label="true target")   # truth
    for src, col in (("gt_pose", "tab:green"), ("fused_est_scale", "tab:red")):   # sources
        ok = np.isfinite(est[src][:, 0]) & reported                  # frames
        ax[0].plot(est[src][ok, 0], est[src][ok, 1], ".", color=col, ms=2, alpha=0.5, label=f"estimate ({src})")   # estimates
    ax[0].set_aspect("equal")                                        # true shape
    ax[0].legend(fontsize=8)                                         # legend
    ax[0].set_title("Target position (synthetic target), top view")  # title
    for src, col in (("gt_pose", "tab:green"), ("fused_est_scale", "tab:red"), ("imu_only", "gray")):   # errors
        ok = np.isfinite(est[src][:, 0]) & reported                  # frames
        ax[1].plot(ts[ok], np.linalg.norm(est[src][ok] - X[ok], axis=1), ".", color=col, ms=2, label=src)   # errors
    ax[1].set_yscale("log")                                          # wide range
    ax[1].set_xlabel("time [s]")                                     # label
    ax[1].set_ylabel("target position error [m]")                    # label
    ax[1].legend(fontsize=8)                                         # legend
    fig.tight_layout()                                               # spacing
    fig.savefig(res / f"step7_{name}.png", dpi=120)                  # save
    return summary                                                   # metrics


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
