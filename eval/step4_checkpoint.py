"""Step 4 checkpoint on the Step 3 descriptor front end: sanity tests, 4a (GT scale, plumbing only), 4b (estimated scale).

usage: python -m eval.step4_checkpoint --frontend sift      (writes results\\fused.tum, vo_only.tum, scale.csv, gt.tum)
       python -m eval.step4_checkpoint --frontend orb       (same files with an _orb suffix)
"""
from __future__ import annotations                                   # forward type hints
import argparse                                                      # command line
import json                                                          # metrics file
import subprocess                                                    # evo
import sys                                                           # interpreter path
from dataclasses import replace                                      # config variants
from pathlib import Path                                             # paths
import numpy as np                                                   # arrays
import matplotlib                                                    # backend selection
matplotlib.use("Agg")                                                # no GUI
import matplotlib.pyplot as plt                                      # plotting
from scipy.spatial.transform import Rotation                         # TUM quaternions

from vo.datasets import Sequence, PROJECT_ROOT                       # data access
from vo.frontend_lk import OK                                        # status code
from vo.run_vo import out_path                                       # VO result naming
from vio.run_vio import run_filter                                   # filter driver
from vio.eskf import FilterConfig                                    # filter config
from eval.ate import umeyama, fmt_ns                                 # alignment, exact timestamps


def write_tum(path: Path, t_ns, R, p):                               # TUM file from rotations and positions
    q = Rotation.from_matrix(R).as_quat()                            # x, y, z, w
    with open(path, "w", encoding="utf-8") as f:                     # write
        for t, pp, qq in zip(t_ns, p, q):                            # one pose per line
            f.write(f"{fmt_ns(t)} {pp[0]:.6f} {pp[1]:.6f} {pp[2]:.6f} {qq[0]:.7f} {qq[1]:.7f} {qq[2]:.7f} {qq[3]:.7f}\n")   # line


def evo_rmse(gt: Path, est: Path, align: bool) -> float:             # evo_ape RMSE (translation)
    exe = Path(sys.executable).parent / "evo_ape.exe"                # venv evo
    cmd = [str(exe), "tum", str(gt), str(est), "--no_warnings"] + (["-a"] if align else [])   # command
    r = subprocess.run(cmd, capture_output=True, text=True)          # run
    for line in r.stdout.splitlines():                               # parse
        if line.strip().startswith("rmse"):                          # rmse line
            return float(line.split()[1])                            # value
    raise RuntimeError(r.stderr[-500:])                              # surface evo errors


def unaligned_rmse(out, frames) -> float:                            # position RMSE without alignment
    e = out["T_w_c"][frames, :3, 3] - out["gt_cam"][frames, :3, 3]   # errors
    return float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))           # RMSE


def main():                                                          # run everything
    ap = argparse.ArgumentParser()                                   # parser
    ap.add_argument("--dataset", default="tumvi_room1")              # dataset key
    ap.add_argument("--frontend", default="sift", choices=["lk", "orb", "sift"])   # VO feeding the filter
    a = ap.parse_args()                                              # parse
    res = PROJECT_ROOT / "results"                                   # output folder
    sfx = "" if a.frontend == "sift" else f"_{a.frontend}"           # file suffix (SIFT is the primary)
    seq = Sequence(a.dataset)                                        # data
    vo = dict(np.load(out_path(a.dataset, a.frontend)))              # Step 3 VO output
    t, T_vo, st, mid = vo["t_ns"], vo["T_w_c"], vo["status"], vo["map_id"]   # unpack
    M = {"frontend": a.frontend, "filter_config": FilterConfig().__dict__}   # metrics container

    # ---------------- sanity 1: z-up gravity check
    g0 = int(seq.gt.t_ns[0])                                         # GT start
    t_start = max(int(seq.imu.t_ns[0]), g0)                          # IMU may start before GT
    w = (seq.imu.t_ns >= t_start) & (seq.imu.t_ns <= t_start + int(0.5e9))   # first 0.5 s
    _, R0 = seq.gt.interp(np.array([t_start]))                       # GT orientation (IMU -> world)
    g_w = R0.as_matrix()[0] @ seq.imu.acc[w].mean(axis=0)            # specific force in world
    M["sanity_zup_world_specific_force"] = g_w.tolist()              # expect about [0, 0, +9.8]
    print(f"[sanity] z-up: R_gt(t0) @ mean(a_m, 0.5 s) = {np.round(g_w, 3)}  (expect about [0, 0, +9.8])")   # report

    # ---------------- sanity 2: IMU-only dead reckoning
    imu = run_filter(seq, vo, "imuonly", verbose=False)              # propagation only
    v = imu["valid"]                                                 # frames with a state
    ts = (t - t[v][0]) * 1e-9                                        # seconds since filter start
    e_imu = np.linalg.norm(imu["T_w_c"][:, :3, 3] - imu["gt_cam"][:, :3, 3], axis=1)   # error
    M["sanity_imu_only_error_m"] = {f"{s}s": float(e_imu[v][np.searchsorted(ts[v], s)]) for s in (1, 2, 5, 10, 30)}   # growth
    print(f"[sanity] IMU-only position error: {M['sanity_imu_only_error_m']}")   # report

    # ---------------- sanity 3: perfect GT relative poses as measurements
    tight = replace(FilterConfig(), sigma_r=0.002, sigma_t_rel=0.01, sigma_t_floor=0.001)   # noise matched to exact data
    for lab, cfg in (("default_noise", FilterConfig()), ("tight_noise", tight)):   # two noise settings
        o = run_filter(seq, vo, "gtrel", cfg, verbose=False)         # exact GT relative poses
        M[f"sanity_gt_relative_{lab}_unaligned_rmse_m"] = unaligned_rmse(o, o["valid"])   # RMSE
    print(f"[sanity] GT relative poses -> filter RMSE: default noise {M['sanity_gt_relative_default_noise_unaligned_rmse_m']:.3f} m, "
          f"tight noise {M['sanity_gt_relative_tight_noise_unaligned_rmse_m']:.3f} m")   # report

    # ---------------- 4a and 4b
    r4a = run_filter(seq, vo, "gtscale", verbose=True)               # GT-scaled VO (plumbing only)
    r4b = run_filter(seq, vo, "estscale", verbose=True)              # estimated scale (the real run)
    gt = r4b["gt_cam"]                                               # oracle camera poses
    valid = r4b["valid"]                                             # frames with a filter state
    okv = (st == OK) & valid                                         # VO-trusted frames inside the filter span
    segs = [s for s in np.unique(mid[okv]) if (okv & (mid == s)).sum() >= 20]   # alignable segments
    vo_sel = okv & np.isin(mid, segs)                                # frames scored for vision-only

    # vision-only: oracle Sim(3) alignment per map segment (VO has no common frame or scale across segments)
    R_al = np.zeros((len(t), 3, 3))                                  # aligned rotations
    p_al = np.zeros((len(t), 3))                                     # aligned positions
    s_ref = np.full(len(t), np.nan)                                  # per-segment reference scale (metres per VO unit)
    for s in segs:                                                   # per segment
        sel = okv & (mid == s)                                       # frames
        sc, Ra, ta = umeyama(T_vo[sel, :3, 3], gt[sel, :3, 3], True)   # Sim(3) to GT
        p_al[sel] = (sc * (Ra @ T_vo[sel, :3, 3].T)).T + ta          # aligned positions
        R_al[sel] = Ra @ T_vo[sel, :3, :3]                           # aligned orientations
        s_ref[mid == s] = sc                                         # reference scale for this segment

    # files
    f_gt, f_fu, f_vo = res / f"gt{sfx}.tum", res / f"fused{sfx}.tum", res / f"vo_only{sfx}.tum"   # names
    f_fu_vo, f_4a = res / f"fused_on_vo_frames{sfx}.tum", res / f"fused_gtscale_PLUMBING_ONLY{sfx}.tum"   # names
    write_tum(f_gt, t[valid], gt[valid, :3, :3], gt[valid, :3, 3])   # ground truth camera poses
    write_tum(f_fu, t[valid], r4b["T_w_c"][valid, :3, :3], r4b["T_w_c"][valid, :3, 3])   # fused, all frames
    write_tum(f_fu_vo, t[vo_sel], r4b["T_w_c"][vo_sel, :3, :3], r4b["T_w_c"][vo_sel, :3, 3])   # fused, VO frames only
    write_tum(f_vo, t[vo_sel], R_al[vo_sel], p_al[vo_sel])           # vision-only, already Sim(3)-aligned per segment
    write_tum(f_4a, t[valid], r4a["T_w_c"][valid, :3, :3], r4a["T_w_c"][valid, :3, 3])   # 4a
    with open(res / f"scale{sfx}.csv", "w", encoding="utf-8") as f:  # scale over time
        f.write("t_ns,t_s,map_id,vo_status,s_est,s_sigma,s_ref_sim3_segment\n")   # header
        for k in np.where(valid)[0]:                                 # one row per filter frame
            f.write(f"{t[k]},{(t[k] - t[valid][0]) * 1e-9:.4f},{mid[k]},{st[k]},{r4b['s'][k]:.6f},"
                    f"{r4b['s_sigma'][k]:.6f},{s_ref[k]:.6f}\n")     # row (NaN before scale init)

    # ATE via evo
    M["evo_ate_rmse_m"] = {
        "vision_only_oracle_sim3_per_segment (evo, no further alignment)": evo_rmse(f_gt, f_vo, False),
        "fused_4b_all_frames_unaligned": evo_rmse(f_gt, f_fu, False),
        "fused_4b_all_frames_se3": evo_rmse(f_gt, f_fu, True),
        "fused_4b_on_vision_frames_unaligned": evo_rmse(f_gt, f_fu_vo, False),
        "fused_4b_on_vision_frames_se3": evo_rmse(f_gt, f_fu_vo, True),
        "fused_4a_gtscale_PLUMBING_ONLY_se3": evo_rmse(f_gt, f_4a, True),
    }                                                                # all ATE numbers from evo
    # like-for-like: fused aligned per segment with SE(3) only (no scale) on the vision-only frames
    e_seg = []                                                       # errors
    for s in segs:                                                   # per segment
        sel = okv & (mid == s)                                       # frames
        _, Ra, ta = umeyama(r4b["T_w_c"][sel, :3, 3], gt[sel, :3, 3], False)   # rigid alignment
        e_seg.append(np.linalg.norm((Ra @ r4b["T_w_c"][sel, :3, 3].T).T + ta - gt[sel, :3, 3], axis=1))   # errors
    M["fused_4b_per_segment_se3_rmse_m"] = float(np.sqrt(np.mean(np.concatenate(e_seg) ** 2)))   # like-for-like
    fin = np.isfinite(r4b["s"]) & vo_sel                             # frames with an initialised scale
    M["scale_median_abs_rel_err_vs_segment_ref"] = float(np.median(np.abs(r4b["s"][fin] / s_ref[fin] - 1)))   # accuracy
    M["counts"] = {"frames": int(len(t)), "filter_frames": int(valid.sum()), "vision_frames_scored": int(vo_sel.sum()),
                   "segments": len(segs), "updates_4b": int(r4b["n_upd"]), "noise_inflated_4b": int(r4b["n_gated"])}   # context
    json.dump(M, open(res / f"metrics_step4{sfx}_{a.dataset}.json", "w", encoding="utf-8"), indent=2, default=float)   # save
    print(json.dumps({k: M[k] for k in ("evo_ate_rmse_m", "fused_4b_per_segment_se3_rmse_m",
                                        "scale_median_abs_rel_err_vs_segment_ref", "counts")}, indent=1))   # report

    # plot 1: trajectories
    tt = (t - t[valid][0]) * 1e-9                                    # seconds
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))                  # figure
    ax[0].plot(gt[valid, 0, 3], gt[valid, 1, 3], "k-", lw=0.8, label="ground truth")   # GT
    for i, s in enumerate(segs):                                     # vision-only segments
        sel = okv & (mid == s)                                       # frames
        ax[0].plot(p_al[sel, 0], p_al[sel, 1], color="tab:blue", lw=0.8, alpha=0.8,
                   label="vision-only (oracle Sim3 per segment)" if i == 0 else None)   # VO
    ax[0].plot(r4b["T_w_c"][valid, 0, 3], r4b["T_w_c"][valid, 1, 3], "r-", lw=0.9, label="fused, estimated scale (no alignment)")   # fused
    ax[0].set_aspect("equal")                                        # true shape
    ax[0].set_xlabel("x [m]")                                        # label
    ax[0].set_ylabel("y [m]")                                        # label
    ax[0].legend(fontsize=8)                                         # legend
    ax[0].set_title(f"{a.frontend.upper()} VO: ground truth, vision-only, fused (top view)")   # title
    ax[1].plot(tt[valid], np.linalg.norm(r4b["T_w_c"][valid, :3, 3] - gt[valid, :3, 3], axis=1), "r-", lw=0.8, label="fused (no alignment)")   # fused error
    ax[1].plot(tt[vo_sel], np.linalg.norm(p_al[vo_sel] - gt[vo_sel, :3, 3], axis=1), ".", color="tab:blue", ms=1.5, label="vision-only (Sim3 per segment)")   # VO error
    ax[1].set_xlabel("time [s]")                                     # label
    ax[1].set_ylabel("position error [m]")                           # label
    ax[1].legend(fontsize=8)                                         # legend
    ax[1].set_title("position error over time")                     # title
    fig.tight_layout()                                               # spacing
    fig.savefig(res / f"step4_trajectories{sfx}.png", dpi=120)       # save
    plt.close(fig)                                                   # free

    # plot 2: scale with covariance
    fig, ax = plt.subplots(figsize=(12, 4.5))                        # figure
    s_, ss_ = r4b["s"], r4b["s_sigma"]                               # estimate and sigma
    ax.plot(tt[valid], s_[valid], "r-", lw=1, label="filter scale s")   # estimate
    ax.fill_between(tt[valid], (s_ - 2 * ss_)[valid], (s_ + 2 * ss_)[valid], color="r", alpha=0.2, label="+-2 sigma")   # band
    for i, s in enumerate(segs):                                     # per-segment reference
        sel = np.where(okv & (mid == s))[0]                          # frames
        ax.hlines(s_ref[sel[0]], tt[sel[0]], tt[sel[-1]], colors="k", linestyles="--", lw=1,
                  label="per-segment Sim3 reference (oracle)" if i == 0 else None)   # reference
    ax.set_xlabel("time [s]")                                        # label
    ax.set_ylabel("metres per VO unit")                              # label
    ax.legend(fontsize=8)                                            # legend
    ax.set_title(f"{a.frontend.upper()} VO: estimated scale factor with 2-sigma band (NaN until initialised per segment)")   # title
    fig.tight_layout()                                               # spacing
    fig.savefig(res / f"step4_scale{sfx}.png", dpi=120)              # save
    plt.close(fig)                                                   # free


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
