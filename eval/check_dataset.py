"""Step 2 checkpoint: rates, calibration, GT plot, undistorted frame, z-up gravity check."""
from __future__ import annotations                                   # forward type hints
import sys                                                           # argv access
from pathlib import Path                                             # paths
import numpy as np                                                   # arrays
import cv2                                                           # image saving
import matplotlib                                                    # plotting backend selection
matplotlib.use("Agg")                                                # no GUI needed
import matplotlib.pyplot as plt                                      # plotting

from vo.datasets import Sequence, rate_hz, PROJECT_ROOT              # our loaders


def main(name: str = "tumvi_room1") -> None:                         # run all checks
    seq = Sequence(name)                                             # load everything
    out = PROJECT_ROOT / "results"                                   # output folder
    out.mkdir(exist_ok=True)                                         # make sure it exists
    print(f"dataset            : {name}")                            # dataset name
    print(f"frames             : {len(seq)}")                        # number of frames
    print(f"IMU samples        : {len(seq.imu.t_ns)}")               # number of IMU samples
    print(f"GT samples         : {len(seq.gt.t_ns)}")                # number of GT samples
    print(f"IMU rate [Hz]      : {rate_hz(seq.imu.t_ns):.2f}")       # IMU rate
    print(f"camera rate [Hz]   : {rate_hz(seq.cam.t_ns):.2f}")       # camera rate
    print(f"GT rate [Hz]       : {rate_hz(seq.gt.t_ns):.2f}")        # GT rate
    dt_cam = np.diff(seq.cam.t_ns) * 1e-3                            # camera period in microseconds
    print(f"cam dt min/max [us]: {dt_cam.min():.0f} / {dt_cam.max():.0f}")   # jitter
    print(f"timestamps sorted  : {bool(np.all(np.diff(seq.imu.t_ns) > 0) and np.all(np.diff(seq.cam.t_ns) > 0))}")  # monotonic
    print(f"camera model       : {seq.calib.model}, size {seq.calib.size}")  # model
    print("K =")                                                     # label
    print(np.array2string(seq.calib.K, precision=3, suppress_small=True))    # intrinsics
    print(f"distortion         : {seq.calib.dist}")                  # distortion
    print("T_cam_imu =")                                             # label
    print(np.array2string(seq.calib.T_cam_imu, precision=4, suppress_small=True))  # extrinsic
    print(f"IMU noise          : {seq.noise}")                       # noise parameters

    t0, t1 = seq.cam.t_ns[0], seq.cam.t_ns[-1]                       # camera time span
    g0, g1 = seq.gt.t_ns[0], seq.gt.t_ns[-1]                         # GT time span
    print(f"cam span inside GT : {bool(g0 <= t0 and t1 <= g1)} (GT {g0}..{g1}, cam {t0}..{t1})")  # coverage

    p, _ = seq.gt.interp(seq.cam.t_ns[(seq.cam.t_ns >= g0) & (seq.cam.t_ns <= g1)])  # GT at frame times
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))                  # two views
    ax[0].plot(p[:, 0], p[:, 1], lw=1)                               # top-down path
    ax[0].plot(p[0, 0], p[0, 1], "go", label="start")                # start marker
    ax[0].plot(p[-1, 0], p[-1, 1], "rx", label="end")                # end marker
    ax[0].set_xlabel("x [m]")                                        # label
    ax[0].set_ylabel("y [m]")                                        # label
    ax[0].set_aspect("equal")                                        # true shape
    ax[0].legend()                                                   # legend
    ax[0].set_title("Ground truth, top view")                        # title
    ax[1].plot(p[:, 0], p[:, 2], lw=1)                               # side view
    ax[1].set_xlabel("x [m]")                                        # label
    ax[1].set_ylabel("z [m]")                                        # label
    ax[1].set_title("Ground truth, side view")                       # title
    fig.tight_layout()                                               # spacing
    fig.savefig(out / "gt_path_room1.png", dpi=120)                  # save plot
    plt.close(fig)                                                   # free memory
    path_len = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))     # path length
    print(f"GT path length [m] : {path_len:.2f}")                    # length

    raw = seq.frame_raw(0)                                           # raw 16-bit frame
    img8 = seq.frame_u8(0)                                           # 8-bit frame
    und = seq.camera.undistort_image(img8)                           # undistorted frame
    print(f"frame0 dtype/shape : {raw.dtype} {raw.shape}, u8 range {img8.min()}..{img8.max()}")  # image info
    side = np.hstack([img8, und])                                    # side by side
    cv2.imwrite(str(out / "undistort_room1.png"), side)              # save image
    c = seq.calib.size[0] // 2                                       # image centre
    rt = seq.camera.project(np.hstack([seq.camera.undistort_to_normalized(np.array([[100.0, 150.0]])), [[1.0]]]))   # round trip
    print(f"project(undistort(px)) for (100,150): {rt.ravel()}")     # should be (100,150)
    print(f"principal point -> normalized: {seq.camera.undistort_to_normalized(seq.calib.K[:2, 2][None])}")  # should be 0,0

    t_start = max(int(seq.imu.t_ns[0]), int(g0))                     # IMU may start before GT
    mask = (seq.imu.t_ns >= t_start) & (seq.imu.t_ns <= t_start + int(0.5e9))  # first 0.5 s with GT
    a_m = seq.imu.acc[mask].mean(axis=0)                             # mean specific force
    _, R0 = seq.gt.interp(np.array([t_start]))                       # GT orientation at t0 (IMU->world)
    a_w = R0.as_matrix()[0] @ a_m                                    # rotate into world
    print(f"z-up check R_gt(t0) @ mean(a_m, 0.5 s) = {np.round(a_w, 3)}  (expect about [0, 0, +9.8])")  # gravity check


if __name__ == "__main__":                                           # required guard on Windows
    main(sys.argv[1] if len(sys.argv) > 1 else "tumvi_room1")        # optional dataset name
