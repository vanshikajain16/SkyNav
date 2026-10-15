"""Step 3 checkpoint for a VO front end: Sim(3)-aligned trajectory + ATE, T_cam_imu direction check,
inlier/match counts over time, and an honest failure report.

usage: python -m eval.vo_checkpoint --frontend orb
"""
from __future__ import annotations                                   # forward type hints
import argparse                                                      # command line
import json                                                          # metrics file
import numpy as np                                                   # arrays
import matplotlib                                                    # backend selection
matplotlib.use("Agg")                                                # no GUI
import matplotlib.pyplot as plt                                      # plotting
from scipy.spatial.transform import Rotation                         # angles

from vo.datasets import Sequence, PROJECT_ROOT                       # data access
from vo.frontend_lk import OK, LOST, UNINIT                          # status codes
from vo.run_vo import out_path                                       # result file naming
from eval.ate import ate_rmse                                        # alignment + ATE


def ang(Ra: np.ndarray, Rb: np.ndarray) -> float:                    # angle between two rotations [deg]
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))   # geodesic distance


def vec_ang(a: np.ndarray, b: np.ndarray) -> float:                  # angle between two vectors [deg]
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))))   # degrees


def runs_of(mask: np.ndarray):                                       # (start, length) of True runs
    out, s = [], None                                                # collectors
    for i, m in enumerate(mask):                                     # scan
        if m and s is None:                                          # run starts
            s = i                                                    # remember
        if not m and s is not None:                                  # run ends
            out.append((s, i - s))                                   # store
            s = None                                                 # reset
    if s is not None:                                                # run reaches the end
        out.append((s, len(mask) - s))                               # store
    return out                                                       # list


def main():                                                          # run the checkpoint
    ap = argparse.ArgumentParser()                                   # parser
    ap.add_argument("--dataset", default="tumvi_room1")              # dataset key
    ap.add_argument("--frontend", default="orb", choices=["lk", "orb", "sift"])   # front end
    a = ap.parse_args()                                              # parse
    tag = f"{a.frontend}_{a.dataset}"                                # file tag
    res = PROJECT_ROOT / "results"                                   # output folder
    d = np.load(out_path(a.dataset, a.frontend))                     # VO output
    seq = Sequence(a.dataset)                                        # data and calibration
    t, T, st, mid = d["t_ns"], d["T_w_c"], d["status"], d["map_id"]  # unpack
    ok = st == OK                                                    # trusted frames
    gt = seq.T_w_c_gt(t)                                             # GT camera poses (hypothesis A)
    n = len(t)                                                       # frames
    summ = {"frontend": a.frontend, "frames": n, "ok_frames": int(ok.sum()), "ok_percent": 100.0 * float(ok.mean()),
            "lost_frames": int((st == LOST).sum()), "uninit_frames": int((st == UNINIT).sum()),
            "map_segments_used": int(len(np.unique(mid[ok]))), "reinitialisations": int(mid.max()), "fps": float(d["fps"])}   # coverage

    # ---- 1. trajectory vs GT, Sim(3) per segment
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))                  # figure
    ax[0].plot(gt[:, 0, 3], gt[:, 1, 3], "k-", lw=0.8, label="ground truth")   # GT
    errs, segs = [], []                                              # collectors
    for s in np.unique(mid[ok]):                                     # per map segment
        sel = ok & (mid == s)                                        # frames
        if sel.sum() < 20:                                           # too short to align
            continue                                                 # skip
        r, al, e, sc = ate_rmse(T[sel, :3, 3], gt[sel, :3, 3], "sim3")   # align and score
        errs.append(e)                                               # keep
        segs.append({"segment": int(s), "poses": int(sel.sum()), "rmse_m": r, "scale_m_per_unit": sc,
                     "first_frame": int(np.where(sel)[0][0]), "last_frame": int(np.where(sel)[0][-1])})   # record
        ax[0].plot(al[:, 0], al[:, 1], lw=0.9, label=f"seg {s} ({int(sel.sum())} poses, {r:.2f} m)")   # aligned path
        ax[1].plot(((t[sel] - t[0]) * 1e-9), e, lw=0.8)              # error over time
    e_all = np.concatenate(errs) if errs else np.array([np.nan])     # all errors
    summ["ate_rmse_sim3_per_segment_m"] = float(np.sqrt(np.mean(e_all ** 2)))   # headline
    summ["ate_poses"] = int(len(e_all))                              # count
    summ["segments"] = segs                                          # per-segment numbers
    ax[0].set_aspect("equal")                                        # true shape
    ax[0].legend(fontsize=6)                                         # legend
    ax[0].set_title(f"{a.frontend}: VO vs ground truth (Sim3 per segment), ATE {summ['ate_rmse_sim3_per_segment_m']:.3f} m")   # title
    ax[1].set_xlabel("time [s]")                                     # label
    ax[1].set_ylabel("position error [m]")                           # label
    ax[1].set_title("error over time (per segment)")                 # title
    fig.tight_layout()                                               # spacing
    fig.savefig(res / f"vo_{tag}_traj.png", dpi=120)                 # save
    plt.close(fig)                                                   # free

    # ---- 2. T_cam_imu direction check on 20 frame pairs
    Rc, tc = seq.calib.T_cam_imu[:3, :3], seq.calib.T_cam_imu[:3, 3]   # extrinsic as read from camchain.yaml
    p_i, R_i = seq.gt.interp(t[ok & (t >= seq.gt.t_ns[0]) & (t <= seq.gt.t_ns[-1])])   # GT IMU poses at OK frames
    idx_ok = np.where(ok & (t >= seq.gt.t_ns[0]) & (t <= seq.gt.t_ns[-1]))[0]   # their frame indices
    pos = {int(k): n_ for n_, k in enumerate(idx_ok)}                # frame -> row
    cands = []                                                       # candidate pairs
    for i in idx_ok:                                                 # pair every frame with the one 10 later
        j = i + 10                                                   # 0.5 s apart
        if j in pos and mid[j] == mid[i] and ok[j]:                  # same segment, both trusted
            Rgi, Rgj = R_i[pos[int(i)]].as_matrix(), R_i[pos[j]].as_matrix()   # GT IMU orientations
            if ang(Rgi, Rgj) >= 3.0:                                 # enough rotation to be discriminative
                cands.append((int(i), j))                            # keep
    pick = [cands[k] for k in np.linspace(0, len(cands) - 1, min(20, len(cands))).astype(int)] if cands else []   # 20 spread out
    rows = []                                                        # per-pair results
    for i, j in pick:                                                # evaluate each pair
        Rv = T[i, :3, :3].T @ T[j, :3, :3]                           # VO relative camera rotation (cam j -> cam i)
        tv = T[i, :3, :3].T @ (T[j, :3, 3] - T[i, :3, 3])            # VO translation direction in camera i frame
        Rwi, Rwj = R_i[pos[i]].as_matrix(), R_i[pos[j]].as_matrix()  # GT IMU orientations
        pwi, pwj = p_i[pos[i]], p_i[pos[j]]                          # GT IMU positions
        out = {"i": i, "j": j}                                       # record
        for hyp in ("A", "B"):                                       # A: T_wc = T_wi inv(T_ci); B: T_wc = T_wi T_ci
            if hyp == "A":                                           # T_cam_imu maps IMU -> camera
                Rci_, Rcj_ = Rwi @ Rc.T, Rwj @ Rc.T                  # camera orientations
                ci = pwi + Rwi @ (-Rc.T @ tc)                        # camera centre i
                cj = pwj + Rwj @ (-Rc.T @ tc)                        # camera centre j
            else:                                                    # inverse convention
                Rci_, Rcj_ = Rwi @ Rc, Rwj @ Rc                      # camera orientations
                ci, cj = pwi + Rwi @ tc, pwj + Rwj @ tc              # camera centres
            out[f"rot_err_{hyp}"] = ang(Rv, Rci_.T @ Rcj_)           # relative-rotation disagreement
            out[f"dir_err_{hyp}"] = vec_ang(tv, Rci_.T @ (cj - ci))  # translation-direction disagreement
        rows.append(out)                                             # store
    # The pair test alone cannot separate A from B when T_cam_imu's rotation is a 180 degree rotation (its own inverse),
    # so add a trajectory-level test: Sim(3) ATE of the VO against the GT camera path built with A and with B.
    p_g, R_g = seq.gt.interp(t[(t >= seq.gt.t_ns[0]) & (t <= seq.gt.t_ns[-1])])   # GT IMU poses at covered frames
    cov = np.where((t >= seq.gt.t_ns[0]) & (t <= seq.gt.t_ns[-1]))[0]   # covered frame indices
    Rg = R_g.as_matrix()                                             # (n,3,3)
    C_A = p_g + np.einsum("nij,j->ni", Rg, -Rc.T @ tc)               # camera centres, hypothesis A
    C_B = p_g + np.einsum("nij,j->ni", Rg, tc)                       # camera centres, hypothesis B
    ate_ab = {}                                                      # per-hypothesis ATE
    for hyp, C in (("A", C_A), ("B", C_B)):                          # both conventions
        e_h = []                                                     # errors
        for s in np.unique(mid[ok]):                                 # per segment
            sel = np.where(ok & (mid == s))[0]                       # frames
            sel = sel[np.isin(sel, cov)]                             # must have GT
            if len(sel) < 20:                                        # too short
                continue                                             # skip
            rows_ = np.searchsorted(cov, sel)                        # GT rows
            e_h.append(ate_rmse(T[sel, :3, 3], C[rows_], "sim3")[2])   # errors
        ate_ab[hyp] = float(np.sqrt(np.mean(np.concatenate(e_h) ** 2))) if e_h else float("nan")   # RMSE
    rot_deg_of_Rc = float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rc).as_rotvec())))   # how far Rc is from identity
    if rows:                                                         # summarise
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ("rot_err_A", "rot_err_B", "dir_err_A", "dir_err_B")}   # means
    else:                                                            # no usable pairs
        mean = {}                                                    # nothing
    traj_rel_diff = abs(ate_ab["A"] - ate_ab["B"]) / max(min(ate_ab["A"], ate_ab["B"]), 1e-9)   # relative difference
    score_A = (mean.get("rot_err_A", 0) + mean.get("dir_err_A", 0)) if rows else float("nan")   # pair-test score A
    score_B = (mean.get("rot_err_B", 0) + mean.get("dir_err_B", 0)) if rows else float("nan")   # pair-test score B
    wins_A = int(sum((r["rot_err_A"] + r["dir_err_A"]) < (r["rot_err_B"] + r["dir_err_B"]) for r in rows))   # pairs won by A
    if traj_rel_diff >= 0.02:                                        # trajectory test is decisive
        better, how = ("A" if ate_ab["A"] < ate_ab["B"] else "B"), "trajectory ATE differs by >= 2%"
    else:                                                            # fall back to the pair test, flagged as weak
        better, how = ("A" if score_A < score_B else "B"), "trajectory ATE indistinguishable (<2%); pair test only, weak evidence"
    names = {"A": "A: T_cam_imu maps IMU-frame points to the camera frame (T_wc = T_wi inv(T_cam_imu))", "B": "B: the inverse"}
    summ["T_cam_imu_check"] = {
        "pairs": len(rows), **mean, "pairs_won_by_A": wins_A, "pairs_won_by_B": len(rows) - wins_A,
        "Rc_rotation_angle_deg": rot_deg_of_Rc, "trajectory_ate_sim3_A_m": ate_ab["A"], "trajectory_ate_sim3_B_m": ate_ab["B"],
        "verdict": names[better], "evidence": how, "per_pair": rows}   # store

    # ---- 3. inlier counts over time
    tt = (t - t[0]) * 1e-9                                           # seconds
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)       # figure
    ax[0].plot(tt, d["n_match"], lw=0.7, label="matches used for PnP")   # matches
    ax[0].plot(tt, d["n_inl"], lw=0.7, label="PnP inliers")          # inliers
    ax[0].plot(tt, d["n_map"], lw=0.7, alpha=0.6, label="map size")  # map
    ax[0].set_yscale("symlog")                                       # wide range incl. zeros
    ax[0].legend(fontsize=8)                                         # legend
    ax[0].set_ylabel("count")                                        # label
    ax[0].set_title(f"{a.frontend}: counts per frame")               # title
    ax[1].fill_between(tt, 0, 1, where=(st == OK), color="tab:green", alpha=0.5, step="mid", label="tracked")   # OK
    ax[1].fill_between(tt, 0, 1, where=(st == LOST), color="tab:orange", alpha=0.8, step="mid", label="lost (coasting)")   # LOST
    ax[1].fill_between(tt, 0, 1, where=(st == UNINIT), color="tab:red", alpha=0.6, step="mid", label="not initialised")   # UNINIT
    ax[1].set_yticks([])                                             # no y ticks
    ax[1].set_xlabel("time [s]")                                     # label
    ax[1].legend(fontsize=8, ncol=3)                                 # legend
    fig.tight_layout()                                               # spacing
    fig.savefig(res / f"vo_{tag}_inliers.png", dpi=120)              # save
    plt.close(fig)                                                   # free

    # ---- 4. failure report
    lost_runs = runs_of(st == LOST)                                  # coasting episodes
    uninit_runs = runs_of(st == UNINIT)                              # re-initialisation waits
    summ["failures"] = {"lost_episodes": len(lost_runs), "longest_lost_run_frames": max([l for _, l in lost_runs], default=0),
                        "uninit_episodes": len(uninit_runs), "longest_uninit_run_frames": max([l for _, l in uninit_runs], default=0),
                        "uninit_runs_start_length": [(int(s), int(l)) for s, l in uninit_runs][:20],
                        "min_inliers_when_ok": int(d["n_inl"][ok].min()) if ok.any() else 0,
                        "median_inliers_when_ok": float(np.median(d["n_inl"][ok])) if ok.any() else 0.0}   # statistics
    json.dump(summ, open(res / f"metrics_step3_{tag}.json", "w", encoding="utf-8"), indent=2, default=float)   # save
    print(json.dumps({k: v for k, v in summ.items() if k not in ("segments",)}, indent=1, default=float)[:6000])   # show


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
