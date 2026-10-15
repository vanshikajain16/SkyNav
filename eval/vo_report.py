"""Step 3 checkpoint: compare the VO trajectory with ground truth (Sim(3)-aligned) and plot it."""
from __future__ import annotations                                   # forward type hints
import argparse                                                      # command line
import subprocess                                                    # run evo for a cross-check
import sys                                                           # interpreter path
from pathlib import Path                                             # paths
import numpy as np                                                   # arrays
import matplotlib                                                    # backend selection
matplotlib.use("Agg")                                                # no GUI
import matplotlib.pyplot as plt                                      # plotting

from vo.datasets import Sequence, PROJECT_ROOT                       # data access
from vo.frontend import OK                                           # status code
from eval.ate import ate_rmse, write_tum                             # metrics and writers


def main():                                                          # report
    ap = argparse.ArgumentParser()                                   # parser
    ap.add_argument("--dataset", default="tumvi_room1")              # dataset key
    a = ap.parse_args()                                              # parse
    res = PROJECT_ROOT / "results"                                   # output folder
    d = np.load(res / f"vo_{a.dataset}.npz")                         # VO output
    seq = Sequence(a.dataset)                                        # ground truth and camera data
    t, T, st, mid = d["t_ns"], d["T_w_c"], d["status"], d["map_id"]  # unpack
    ok = st == OK                                                    # trusted poses only
    gt = seq.T_w_c_gt(t)                                             # GT camera poses at frame times
    print(f"frames {len(t)}  OK {int(ok.sum())} ({100 * ok.mean():.1f}%)  map segments {int(mid.max()) + 1}")   # coverage
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))                    # figure
    ax[0].plot(gt[:, 0, 3], gt[:, 1, 3], "k-", lw=1, label="ground truth")   # GT path
    all_est, all_gt = [], []                                         # for the global number
    for m in np.unique(mid[ok]):                                     # one alignment per map segment
        sel = ok & (mid == m)                                        # frames of this segment
        if sel.sum() < 20:                                           # too short to align
            continue                                                 # skip
        for mode in ("sim3",):                                       # monocular needs scale alignment
            rmse, al, err, s = ate_rmse(T[sel, :3, 3], gt[sel, :3, 3], mode)   # align and score
            print(f"segment {m}: {int(sel.sum())} poses, {mode} ATE RMSE = {rmse:.4f} m, scale = {s:.4f}")   # report
        ax[0].plot(al[:, 0], al[:, 1], lw=1, label=f"VO segment {m} (Sim3-aligned)")   # aligned path
        all_est.append(al)                                           # collect
        all_gt.append(gt[sel, :3, 3])                                # collect
        ax[1].plot(t[sel] * 1e-9 - t[0] * 1e-9, err, lw=1)           # error over time
    if all_est:                                                      # global figure of merit
        e = np.linalg.norm(np.vstack(all_est) - np.vstack(all_gt), axis=1)   # per-pose errors
        print(f"overall (per-segment aligned) RMSE = {np.sqrt(np.mean(e ** 2)):.4f} m over {len(e)} poses")   # number
    ax[0].set_aspect("equal")                                        # true shape
    ax[0].set_xlabel("x [m]")                                        # label
    ax[0].set_ylabel("y [m]")                                        # label
    ax[0].legend(fontsize=7)                                         # legend
    ax[0].set_title("VO (monocular, Sim3-aligned) vs ground truth")  # title
    ax[1].set_xlabel("time [s]")                                     # label
    ax[1].set_ylabel("position error [m]")                           # label
    ax[1].set_title("error over time")                               # title
    fig.tight_layout()                                               # spacing
    fig.savefig(res / f"vo_vs_gt_{a.dataset}.png", dpi=120)          # save
    seg = mid[ok] == np.bincount(mid[ok]).argmax()                   # largest segment for evo
    write_tum(res / f"gt_{a.dataset}.tum", t[ok][seg], gt[ok][seg])  # GT in TUM format
    write_tum(res / f"vo_{a.dataset}.tum", t[ok][seg], T[ok][seg])   # estimate in TUM format
    evo = Path(sys.executable).parent / "evo_ape.exe"                # evo from the venv
    r = subprocess.run([str(evo), "tum", str(res / f"gt_{a.dataset}.tum"), str(res / f"vo_{a.dataset}.tum"),
                        "-a", "-s", "--no_warnings"], capture_output=True, text=True)   # cross-check
    print("evo_ape (largest segment, -a -s):")                       # label
    print("\n".join(l for l in r.stdout.splitlines() if "rmse" in l or "mean" in l or "max" in l))   # key lines
    if r.returncode != 0:                                            # surface errors
        print(r.stderr[-400:])                                       # tail of stderr


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
