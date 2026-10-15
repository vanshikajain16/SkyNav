"""Step 4 checkpoint: vision-only vs fused trajectories, ATE for each, scale estimate over time."""
from __future__ import annotations                                   # forward type hints
import json                                                          # metrics file
import numpy as np                                                   # arrays
import matplotlib                                                    # backend selection
matplotlib.use("Agg")                                                # no GUI
import matplotlib.pyplot as plt                                      # plotting

from vo.datasets import Sequence, PROJECT_ROOT                       # data access
from vo.frontend import OK                                           # status code
from vio.run_vio import run_filter                                   # filter driver
from eval.ate import ate_rmse                                        # metrics


def rmse(e: np.ndarray) -> float:                                    # root mean square
    return float(np.sqrt(np.mean(np.asarray(e) ** 2)))               # RMSE


def main(name: str = "tumvi_room1") -> dict:                         # run all comparisons
    res = PROJECT_ROOT / "results"                                   # output folder
    seq = Sequence(name)                                             # data
    vo = dict(np.load(res / f"vo_{name}.npz"))                       # VO output
    t, T_vo, st, mid = vo["t_ns"], vo["T_w_c"], vo["status"], vo["map_id"]   # unpack
    runs = {m: run_filter(seq, vo, m, verbose=True) for m in ("imuonly", "gtscale", "estscale")}   # filter runs
    gt = runs["estscale"]["gt_cam"]                                  # oracle camera poses
    ok = (st == OK) & runs["estscale"]["valid"]                      # frames present in every method
    segs = [m for m in np.unique(mid[ok]) if (ok & (mid == m)).sum() >= 20]   # usable segments
    m_out = {"frames_compared": int(ok.sum()), "segments": [int(s) for s in segs]}   # metrics dict

    e_vo, ref_scale = [], {}                                         # vision-only errors, reference scales
    for s in segs:                                                   # per-segment oracle Sim3 alignment
        sel = ok & (mid == s)                                        # frames
        r, _, e, sc = ate_rmse(T_vo[sel, :3, 3], gt[sel, :3, 3], "sim3")   # align + score
        e_vo.append(e)                                               # keep
        ref_scale[int(s)] = sc                                       # metres per VO unit (reporting only)
    m_out["vision_only_sim3_per_segment_rmse_m"] = rmse(np.concatenate(e_vo))   # headline vision-only number

    def fused_numbers(run):                                          # metrics for one filter run
        p_f, p_g = run["T_w_c"][ok, :3, 3], gt[ok, :3, 3]            # positions
        out = {"unaligned_rmse_m": rmse(np.linalg.norm(p_f - p_g, axis=1))}   # no alignment
        out["global_se3_rmse_m"] = ate_rmse(p_f, p_g, "se3")[0]      # one rigid alignment for the whole run
        e_seg = []                                                   # per-segment rigid alignment (same frames as VO)
        for s in segs:                                               # loop segments
            sel = ok & (mid == s)                                    # frames
            e_seg.append(ate_rmse(run["T_w_c"][sel, :3, 3], gt[sel, :3, 3], "se3")[2])   # errors
        out["per_segment_se3_rmse_m"] = rmse(np.concatenate(e_seg))  # comparable with the vision-only number
        return out                                                   # metrics

    for m in ("imuonly", "gtscale", "estscale"):                     # all modes
        m_out[m] = fused_numbers(runs[m])                            # store
    s_est = runs["estscale"]["s"]                                    # estimated scale series
    rel = np.array([s_est[k] / ref_scale[int(mid[k])] - 1 for k in np.where(ok)[0]
                    if int(mid[k]) in ref_scale and np.isfinite(s_est[k])])   # relative errors after initialisation
    m_out["scale_median_abs_rel_error_vs_segment_reference"] = float(np.median(np.abs(rel)))   # accuracy
    m_out["scale_frames_evaluated"] = int(len(rel))                  # how many frames contributed

    print("\n=== Step 4 results (TUM VI room1, 135 s, 146 m path) ===")   # header
    print(f"frames compared: {m_out['frames_compared']}, VO segments used: {len(segs)}")   # context
    print(f"vision-only  (oracle Sim3 per segment)         ATE RMSE: {m_out['vision_only_sim3_per_segment_rmse_m']:.3f} m")   # VO
    for m, lab in (("estscale", "fused, scale estimated (REAL)"), ("gtscale", "fused, GT-scaled VO (plumbing only)"),
                   ("imuonly", "IMU-only dead reckoning")):          # modes
        r = m_out[m]                                                 # metrics
        print(f"{lab:36s} unaligned {r['unaligned_rmse_m']:7.3f} m | global SE3 {r['global_se3_rmse_m']:7.3f} m | "
              f"per-segment SE3 {r['per_segment_se3_rmse_m']:7.3f} m")   # line
    print(f"scale estimate median |relative error| vs per-segment Sim3 reference: "
          f"{100 * m_out['scale_median_abs_rel_error_vs_segment_reference']:.1f}%")   # accuracy

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))                  # figure
    ax[0].plot(gt[ok, 0, 3], gt[ok, 1, 3], "k-", lw=0.8, label="ground truth")   # GT
    for i, s in enumerate(segs):                                     # VO segments, aligned
        sel = ok & (mid == s)                                        # frames
        _, al, _, _ = ate_rmse(T_vo[sel, :3, 3], gt[sel, :3, 3], "sim3")   # aligned
        ax[0].plot(al[:, 0], al[:, 1], lw=0.8, color="tab:blue", alpha=0.7, label="vision-only (Sim3 per segment)" if i == 0 else None)   # VO
    pf = runs["estscale"]["T_w_c"][ok, :3, 3]                        # fused path
    ax[0].plot(pf[:, 0], pf[:, 1], "r-", lw=1.0, label="fused, scale estimated")   # fused
    ax[0].set_aspect("equal")                                        # true shape
    ax[0].set_xlabel("x [m]")                                        # label
    ax[0].set_ylabel("y [m]")                                        # label
    ax[0].legend(fontsize=8)                                         # legend
    ax[0].set_title("Trajectories vs ground truth (top view)")       # title
    tt = (t - t[0]) * 1e-9                                           # seconds
    ef = np.linalg.norm(runs["estscale"]["T_w_c"][:, :3, 3] - gt[:, :3, 3], axis=1)   # fused error
    ax[1].plot(tt[ok], ef[ok], "r-", lw=0.8, label="fused (no alignment)")   # fused error
    ei = np.linalg.norm(runs["imuonly"]["T_w_c"][:, :3, 3] - gt[:, :3, 3], axis=1)   # IMU-only error
    ax[1].plot(tt[ok], ei[ok], color="gray", lw=0.8, label="IMU only")   # IMU-only error
    for i, s in enumerate(segs):                                     # VO errors per segment
        sel = ok & (mid == s)                                        # frames
        e = ate_rmse(T_vo[sel, :3, 3], gt[sel, :3, 3], "sim3")[2]    # errors
        ax[1].plot(tt[sel], e, color="tab:blue", lw=0.8, label="vision-only (Sim3 per segment)" if i == 0 else None)   # VO
    ax[1].set_yscale("log")                                          # wide dynamic range
    ax[1].set_xlabel("time [s]")                                     # label
    ax[1].set_ylabel("position error [m]")                           # label
    ax[1].legend(fontsize=8)                                         # legend
    ax[1].set_title("Position error over time")                      # title
    ss = runs["estscale"]["s_sigma"]                                 # scale sigma
    ax[2].plot(tt[ok], s_est[ok], "r-", lw=1, label="filter scale s")   # estimate
    ax[2].fill_between(tt[ok], (s_est - ss)[ok], (s_est + ss)[ok], color="r", alpha=0.2, label="+-1 sigma")   # band
    for s in segs:                                                   # per-segment reference
        sel = ok & (mid == s)                                        # frames
        ax[2].hlines(ref_scale[int(s)], tt[sel][0], tt[sel][-1], colors="k", linestyles="--", lw=1,
                     label="per-segment Sim3 reference" if s == segs[0] else None)   # reference
    ax[2].set_xlabel("time [s]")                                     # label
    ax[2].set_ylabel("metres per VO unit")                           # label
    ax[2].legend(fontsize=8)                                         # legend
    ax[2].set_title("Estimated scale factor")                        # title
    fig.tight_layout()                                               # spacing
    fig.savefig(res / f"step4_{name}.png", dpi=120)                  # save
    with open(res / f"metrics_step4_{name}.json", "w", encoding="utf-8") as f:   # save metrics
        json.dump(m_out, f, indent=2)                                # write
    return m_out                                                     # metrics


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
