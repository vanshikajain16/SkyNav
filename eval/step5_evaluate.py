"""Step 5 checkpoint: vision-only vs fused, per sequence, ATE and RPE, plus aligned trajectory plots.

usage: python -m eval.step5_evaluate --frontend sift --datasets tumvi_room1 tumvi_room4
Alignment is stated in every row. Vision-only has no global frame or scale, so it gets an ORACLE Sim(3) fit per map
segment; fused is reported with no alignment and with one global SE(3). A like-for-like row aligns fused per segment (SE3).
"""
from __future__ import annotations                                   # forward type hints
import argparse                                                      # command line
import json                                                          # metrics file
import numpy as np                                                   # arrays
import matplotlib                                                    # backend selection
matplotlib.use("Agg")                                                # no GUI
import matplotlib.pyplot as plt                                      # plotting

from vo.datasets import Sequence, PROJECT_ROOT                       # data access
from vo.frontend_lk import OK                                        # status code
from vo.run_vo import out_path                                       # VO result naming
from vio.run_vio import run_filter                                   # filter driver
from eval.ate import umeyama, write_tum                              # my alignment and TUM writer
from eval.metrics import rpe, summarize                              # my RPE
from eval.step4_checkpoint import evo_rmse                           # evo cross-check


def rms(x) -> float:                                                 # root mean square
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))               # RMSE


def evaluate(ds: str, frontend: str) -> tuple[dict, dict]:           # metrics + plot data for one sequence
    res = PROJECT_ROOT / "results"                                   # output folder
    seq = Sequence(ds)                                               # data
    vo = dict(np.load(out_path(ds, frontend)))                       # VO output
    runs = {m: run_filter(seq, vo, m, verbose=False) for m in ("estscale", "imuonly")}   # filter runs
    fu, im = runs["estscale"], runs["imuonly"]                        # shorthand
    t, T_vo, st, mid = vo["t_ns"], vo["T_w_c"], vo["status"], vo["map_id"]   # unpack
    gt, valid = fu["gt_cam"], fu["valid"]                            # GT camera poses, filter validity
    okv = (st == OK) & valid                                         # frames scored by every method
    segs = [int(s) for s in np.unique(mid[okv]) if (okv & (mid == s)).sum() >= 20]   # alignable segments
    sc = np.isin(mid, segs) & okv                                    # scored frames
    M = {"dataset": ds, "frontend": frontend, "frames": int(len(t)), "frames_scored": int(sc.sum()),
         "vo_ok_percent": 100.0 * float((st == OK).mean()), "segments": len(segs)}   # context

    # --- vision-only: oracle Sim(3) per segment (scale kept for RPE)
    scale, e_vo, p_al = {}, [], np.zeros((len(t), 3))                # per-segment scale, errors, aligned positions
    for s in segs:                                                   # per segment
        sel = okv & (mid == s)                                       # frames
        sg, R, tt = umeyama(T_vo[sel, :3, 3], gt[sel, :3, 3], True)  # my Sim(3)
        scale[s] = sg                                                # remember scale
        p_al[sel] = (sg * (R @ T_vo[sel, :3, 3].T)).T + tt           # aligned
        e_vo.append(np.linalg.norm(p_al[sel] - gt[sel, :3, 3], axis=1))   # errors
    row_vo = {"alignment": "oracle Sim(3) per segment", "ate_rmse_m": rms(np.concatenate(e_vo))}   # headline
    # --- fused variants
    pf, pg = fu["T_w_c"][sc, :3, 3], gt[sc, :3, 3]                   # positions on scored frames
    sg, R, tt = umeyama(pf, pg, False)                               # one global rigid alignment
    pf_se3 = (R @ fu["T_w_c"][:, :3, 3].T).T + tt                    # whole fused path aligned
    e_seg = []                                                       # per-segment SE(3) errors
    for s in segs:                                                   # per segment
        sel = okv & (mid == s)                                       # frames
        _, Rs, ts = umeyama(fu["T_w_c"][sel, :3, 3], gt[sel, :3, 3], False)   # rigid fit
        e_seg.append(np.linalg.norm((Rs @ fu["T_w_c"][sel, :3, 3].T).T + ts - gt[sel, :3, 3], axis=1))   # errors
    row_fu = {"ate_unaligned_rmse_m": rms(np.linalg.norm(pf - pg, axis=1)),
              "ate_global_se3_rmse_m": rms(np.linalg.norm(pf_se3[sc] - pg, axis=1)),
              "ate_per_segment_se3_rmse_m": rms(np.concatenate(e_seg))}   # three views
    pi, = (im["T_w_c"][sc, :3, 3],)                                  # IMU-only positions
    sgi, Ri, ti = umeyama(pi, pg, False)                             # global rigid
    row_im = {"ate_unaligned_rmse_m": rms(np.linalg.norm(pi - pg, axis=1)),
              "ate_global_se3_rmse_m": rms(np.linalg.norm((Ri @ pi.T).T + ti - pg, axis=1))}   # IMU-only
    # --- RPE on identical pairs: both frames scored, same map segment (vision uses its per-segment scale)
    for lab, d in (("1s", 20), ("5s", 100)):                         # intervals (camera is 20 Hz)
        dv, rv, df, rf = [], [], [], []                              # collectors
        for s in segs:                                               # per segment
            v = okv & (mid == s)                                     # frames
            a, b = rpe(T_vo, gt, v, d, mid, scale=scale[s]); dv.append(a); rv.append(b)   # vision-only
            a, b = rpe(fu["T_w_c"], gt, v, d, mid); df.append(a); rf.append(b)   # fused
        row_vo[f"rpe_trans_{lab}"], row_vo[f"rpe_rot_deg_{lab}"] = summarize(np.concatenate(dv)), summarize(np.concatenate(rv))   # vision
        row_fu[f"rpe_trans_{lab}"], row_fu[f"rpe_rot_deg_{lab}"] = summarize(np.concatenate(df)), summarize(np.concatenate(rf))   # fused
    M["vision_only"], M["fused_est_scale"], M["imu_only"] = row_vo, row_fu, row_im   # store

    # --- evo cross-check of the headline numbers (files written for inspection too)
    write_tum(res / f"gt_{ds}_{frontend}.tum", t[valid], gt[valid])  # GT
    write_tum(res / f"fused_{ds}_{frontend}.tum", t[valid], fu["T_w_c"][valid])   # fused (all frames)
    Tv = np.tile(np.eye(4), (len(t), 1, 1)); Tv[:, :3, 3] = p_al     # vision-only with aligned positions
    Tv[:, :3, :3] = T_vo[:, :3, :3]                                  # (orientation not used by ATE)
    write_tum(res / f"vo_only_{ds}_{frontend}.tum", t[sc], Tv[sc])   # vision-only aligned
    g, f_, v_ = (res / f"gt_{ds}_{frontend}.tum", res / f"fused_{ds}_{frontend}.tum", res / f"vo_only_{ds}_{frontend}.tum")   # paths
    M["evo_cross_check_m"] = {"vision_only_aligned": evo_rmse(g, v_, False), "fused_unaligned": evo_rmse(g, f_, False),
                              "fused_se3": evo_rmse(g, f_, True)}    # evo numbers (all-frame fused, so slightly different frame set)
    return M, dict(t=t, gt=gt, p_al=p_al, pf=fu["T_w_c"][:, :3, 3], pf_se3=pf_se3, valid=valid, sc=sc, okv=okv, mid=mid, segs=segs)   # plot data


def plot(ds, frontend, P):                                           # aligned trajectory figure
    t, gt, sc = P["t"], P["gt"], P["sc"]                             # shorthand
    tt = (t - t[P["valid"]][0]) * 1e-9                               # seconds
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))                   # figure
    for a_, pos, ttl, col in ((ax[0, 0], P["p_al"], "vision-only (oracle Sim3 per segment)", "tab:blue"),
                              (ax[0, 1], P["pf_se3"], "fused (one global SE3 alignment)", "tab:red")):   # two panels
        a_.plot(gt[P["valid"], 0, 3], gt[P["valid"], 1, 3], "k-", lw=0.7, label="ground truth")   # GT
        a_.plot(pos[sc, 0], pos[sc, 1], color=col, lw=0.8, label=ttl)   # estimate
        a_.set_aspect("equal")                                       # true shape
        a_.set_xlabel("x [m]")                                       # label
        a_.set_ylabel("y [m]")                                       # label
        a_.legend(fontsize=8)                                        # legend
        a_.set_title(ttl)                                            # title
    e_v = np.linalg.norm(P["p_al"][sc] - gt[sc, :3, 3], axis=1)      # vision-only error
    e_f = np.linalg.norm(P["pf_se3"][sc] - gt[sc, :3, 3], axis=1)    # fused SE3 error
    e_u = np.linalg.norm(P["pf"][sc] - gt[sc, :3, 3], axis=1)        # fused unaligned error
    ax[1, 0].plot(tt[sc], e_v, ".", ms=1.5, color="tab:blue", label="vision-only (Sim3/segment)")   # error vs time
    ax[1, 0].plot(tt[sc], e_f, "-", lw=0.8, color="tab:red", label="fused (global SE3)")   # error vs time
    ax[1, 0].plot(tt[sc], e_u, "-", lw=0.8, color="tab:orange", label="fused (unaligned)")   # error vs time
    ax[1, 0].set_xlabel("time [s]")                                  # label
    ax[1, 0].set_ylabel("position error [m]")                        # label
    ax[1, 0].legend(fontsize=8)                                      # legend
    for e, lab, col in ((e_v, "vision-only", "tab:blue"), (e_f, "fused SE3", "tab:red"), (e_u, "fused unaligned", "tab:orange")):   # CDFs
        ax[1, 1].plot(np.sort(e), np.linspace(0, 1, len(e)), color=col, label=lab)   # error CDF
    ax[1, 1].set_xlabel("position error [m]")                        # label
    ax[1, 1].set_ylabel("fraction of poses")                         # label
    ax[1, 1].legend(fontsize=8)                                      # legend
    fig.suptitle(f"{ds}, {frontend.upper()} VO")                     # title
    fig.tight_layout()                                               # spacing
    fig.savefig(PROJECT_ROOT / "results" / f"step5_{ds}_{frontend}.png", dpi=120)   # save
    plt.close(fig)                                                   # free


def main():                                                          # run the evaluation
    ap = argparse.ArgumentParser()                                   # parser
    ap.add_argument("--frontend", default="sift", choices=["lk", "orb", "sift"])   # VO front end
    ap.add_argument("--datasets", nargs="+", default=["tumvi_room1", "tumvi_room4"])   # sequences
    a = ap.parse_args()                                              # parse
    res = PROJECT_ROOT / "results"                                   # output folder
    allm = {}                                                        # all metrics
    for ds in a.datasets:                                            # each sequence
        M, P = evaluate(ds, a.frontend)                              # evaluate
        allm[ds] = M                                                 # store
        plot(ds, a.frontend, P)                                      # plots
    json.dump(allm, open(res / f"metrics_step5_{a.frontend}.json", "w", encoding="utf-8"), indent=2, default=float)   # save
    L = [f"Front end: {a.frontend}. ATE = position RMSE [m]; RPE = relative pose error over 1 s (20 frames) / 5 s (100 frames), "
         "identical frame pairs for vision-only and fused.", "",
         "| sequence | method | alignment | ATE [m] | RPE trans 1 s [m] | RPE trans 5 s [m] | RPE rot 1 s RMSE / median [deg] |",
         "|---|---|---|---|---|---|---|"]                           # markdown table
    for ds, M in allm.items():                                       # rows
        v, f = M["vision_only"], M["fused_est_scale"]                # shorthand
        rr = lambda r: f"{r['rpe_rot_deg_1s']['rmse']:.2f} / {r['rpe_rot_deg_1s']['median']:.2f}"   # rotation cell
        L += [f"| {ds} | vision-only | oracle Sim(3) per segment | {v['ate_rmse_m']:.3f} | {v['rpe_trans_1s']['rmse']:.3f} | {v['rpe_trans_5s']['rmse']:.3f} | {rr(v)} |",
              f"| {ds} | fused (est. scale) | none | {f['ate_unaligned_rmse_m']:.3f} | {f['rpe_trans_1s']['rmse']:.3f} | {f['rpe_trans_5s']['rmse']:.3f} | {rr(f)} |",
              f"| {ds} | fused (est. scale) | one global SE(3) | {f['ate_global_se3_rmse_m']:.3f} | same | same | same |",
              f"| {ds} | fused (est. scale) | SE(3) per segment (like-for-like) | {f['ate_per_segment_se3_rmse_m']:.3f} | same | same | same |",
              f"| {ds} | IMU only | none / global SE(3) | {M['imu_only']['ate_unaligned_rmse_m']:.1f} / {M['imu_only']['ate_global_se3_rmse_m']:.1f} | - | - | - |"]   # rows
    L += ["", "| sequence | frames | scored | VO tracked | VO segments |", "|---|---|---|---|---|"]   # context table
    L += [f"| {ds} | {M['frames']} | {M['frames_scored']} | {M['vo_ok_percent']:.1f}% | {M['segments']} |" for ds, M in allm.items()]   # rows
    (res / f"step5_table_{a.frontend}.md").write_text("\n".join(L) + "\n", encoding="utf-8")   # save
    print("\n".join(L))                                              # show
    for ds, M in allm.items():                                       # evo cross-check
        print(ds, "evo cross-check:", {k: round(v, 4) for k, v in M["evo_cross_check_m"].items()})   # numbers


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
