"""Step 5: full evaluation table (ATE + RPE) for every method, with an evo cross-check.

Reads results\\vo_<ds>.npz, runs the filter modes, writes results\\metrics_step5_<ds>.json,
results\\step5_table_<ds>.md and TUM trajectory files.
"""
from __future__ import annotations                                   # forward type hints
import json                                                          # metrics file
import subprocess                                                    # evo cross-check
import sys                                                           # interpreter path
from pathlib import Path                                             # paths
import numpy as np                                                   # arrays

from vo.datasets import Sequence, PROJECT_ROOT                       # data access
from vo.frontend import OK                                           # status code
from vio.run_vio import run_filter                                   # filter driver
from eval.ate import ate_rmse, write_tum                             # alignment and writers
from eval.metrics import rpe, summarize                              # RPE

DELTAS = {"1s": 20, "5s": 100}                                       # RPE intervals in frames (camera is 20 Hz)


def evo_check(gt_path: Path, est_path: Path) -> dict:                # run evo_ape / evo_rpe on TUM files
    exe = Path(sys.executable).parent                                # venv Scripts folder
    out = {}                                                         # results
    r = subprocess.run([str(exe / "evo_ape.exe"), "tum", str(gt_path), str(est_path), "-a", "--no_warnings"],
                       capture_output=True, text=True)               # ATE, SE3 alignment
    for line in r.stdout.splitlines():                               # parse
        if line.strip().startswith("rmse"):                          # rmse line
            out["evo_ape_global_se3_rmse_m"] = float(line.split()[1])   # value
    r = subprocess.run([str(exe / "evo_rpe.exe"), "tum", str(gt_path), str(est_path), "-r", "trans_part",
                        "-d", "20", "-u", "f", "--all_pairs", "-a", "--no_warnings"],
                       capture_output=True, text=True)               # RPE over 20 frames
    for line in r.stdout.splitlines():                               # parse
        if line.strip().startswith("rmse"):                          # rmse line
            out["evo_rpe_1s_trans_rmse_m"] = float(line.split()[1])   # value
    return out                                                       # numbers


def main(name: str = "tumvi_room1") -> dict:                         # run the evaluation
    res = PROJECT_ROOT / "results"                                   # output folder
    seq = Sequence(name)                                             # data
    vo = dict(np.load(res / f"vo_{name}.npz"))                       # VO output
    t, T_vo, st, mid = vo["t_ns"], vo["T_w_c"], vo["status"], vo["map_id"]   # unpack
    ok = st == OK                                                    # trusted VO frames
    runs = {m: run_filter(seq, vo, m, verbose=False) for m in ("estscale", "gtscale", "imuonly")}   # filter runs
    gt = runs["estscale"]["gt_cam"]                                  # oracle camera poses
    valid_f = runs["estscale"]["valid"]                              # frames where the filter has a state
    cmp_ok = ok & valid_f                                            # frames common to all methods
    segs = [s for s in np.unique(mid[cmp_ok]) if (cmp_ok & (mid == s)).sum() >= 20]   # usable segments
    out = {"dataset": name, "frames_compared": int(cmp_ok.sum()), "n_segments": len(segs)}   # metrics

    # vision-only: oracle Sim(3) alignment per segment (this is generous, stated in the table)
    e_all, sc = [], {}                                               # errors, scales
    T_vo_m = T_vo.copy()                                             # trajectory in metres (scaled per segment)
    for s in segs:                                                   # loop segments
        sel = cmp_ok & (mid == s)                                    # frames
        _, _, e, scl = ate_rmse(T_vo[sel, :3, 3], gt[sel, :3, 3], "sim3")   # align
        e_all.append(e)                                              # keep
        sc[int(s)] = scl                                             # scale
    vo_row = {"ate_rmse_m": float(np.sqrt(np.mean(np.concatenate(e_all) ** 2))),
              "alignment": "oracle Sim(3) per map segment"}          # row
    for lab, d in DELTAS.items():                                    # RPE per interval
        dts, drs = [], []                                            # collectors
        for s in segs:                                               # per segment (scale known)
            v = cmp_ok & (mid == s)                                  # frames
            a, b = rpe(T_vo, gt, v, d, mid, scale=sc[int(s)])        # errors
            dts.append(a)                                            # keep
            drs.append(b)                                            # keep
        vo_row[f"rpe_trans_{lab}"] = summarize(np.concatenate(dts))  # translation
        vo_row[f"rpe_rot_deg_{lab}"] = summarize(np.concatenate(drs))   # rotation
    out["vision_only"] = vo_row                                      # store

    for m, label in (("estscale", "fused_est_scale"), ("gtscale", "fused_gt_scale_PLUMBING_ONLY"),
                     ("imuonly", "imu_only_dead_reckoning")):        # fused methods
        r = runs[m]                                                  # run
        Tf = r["T_w_c"]                                              # fused poses
        p_f, p_g = Tf[cmp_ok, :3, 3], gt[cmp_ok, :3, 3]              # positions on common frames
        row = {"ate_unaligned_rmse_m": float(np.sqrt(np.mean(np.sum((p_f - p_g) ** 2, axis=1)))),
               "ate_global_se3_rmse_m": ate_rmse(p_f, p_g, "se3")[0]}   # ATE
        for lab, d in DELTAS.items():                                # RPE per interval
            a, b = rpe(Tf, gt, valid_f, d)                           # errors over all valid frames
            row[f"rpe_trans_{lab}"] = summarize(a)                   # translation
            row[f"rpe_rot_deg_{lab}"] = summarize(b)                 # rotation
        out[label] = row                                             # store

    gt_p, est_p = res / f"gt_full_{name}.tum", res / f"fused_{name}.tum"   # TUM files for evo
    write_tum(gt_p, t[valid_f], gt[valid_f])                         # ground truth
    write_tum(est_p, t[valid_f], runs["estscale"]["T_w_c"][valid_f])   # fused estimate
    out["evo_cross_check_fused_est_scale"] = evo_check(gt_p, est_p)  # cross-check numbers

    with open(res / f"metrics_step5_{name}.json", "w", encoding="utf-8") as f:   # save metrics
        json.dump(out, f, indent=2)                                  # write
    rows = [("vision-only (oracle Sim3/segment)", vo_row)] + [(k, out[k]) for k in
            ("fused_est_scale", "fused_gt_scale_PLUMBING_ONLY", "imu_only_dead_reckoning")]   # table rows
    lines = ["| method | ATE RMSE [m] | ATE global SE3 [m] | RPE trans 1 s RMSE [m] | RPE rot 1 s RMSE [deg] | RPE trans 5 s RMSE [m] |",
             "|---|---|---|---|---|---|"]                           # markdown header
    for lab, r in rows:                                              # one row each
        a = r.get("ate_rmse_m", r.get("ate_unaligned_rmse_m"))       # headline ATE
        g = r.get("ate_global_se3_rmse_m", float("nan"))             # global SE3 ATE
        lines.append(f"| {lab} | {a:.3f} | {g:.3f} | {r['rpe_trans_1s'].get('rmse', float('nan')):.4f} | "
                     f"{r['rpe_rot_deg_1s'].get('rmse', float('nan')):.3f} | {r['rpe_trans_5s'].get('rmse', float('nan')):.4f} |")   # row
    (res / f"step5_table_{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")   # save table
    print("\n".join(lines))                                          # show it
    print("evo cross-check (fused, est scale):", out["evo_cross_check_fused_est_scale"])   # evo numbers
    print(f"frames compared {out['frames_compared']}, segments {out['n_segments']}")   # context
    return out                                                       # metrics


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
