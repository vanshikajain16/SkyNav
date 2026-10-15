"""Run a monocular VO front end on a sequence and save the trajectory.

--frontend lk   : LK tracking version (vo/frontend_lk.py), writes results\\vo_<dataset>.npz (used by the filter reports)
--frontend orb  : ORB descriptors + map + PnP (vo/frontend.py), writes results\\vo_orb_<dataset>.npz
--frontend sift : same pipeline with SIFT,                    writes results\\vo_sift_<dataset>.npz
"""
from __future__ import annotations                                   # forward type hints
import argparse                                                      # command line
import time                                                          # timing
import numpy as np                                                   # arrays

from vo.datasets import Sequence, PROJECT_ROOT                       # data access
from vo.frontend_lk import OK, LOST, UNINIT                          # status codes (shared)


def make_vo(frontend: str, seq: Sequence, cfg=None):                 # build the requested front end
    if frontend == "lk":                                             # LK tracking version
        from vo.frontend_lk import MonoVO, VoConfig                  # lazy import
        return MonoVO(seq.camera, seq.calib.size, cfg or VoConfig())  # front end
    from dataclasses import replace                                  # config variants
    from vo.frontend import MonoVO, OrbVoConfig                      # descriptor-based version
    base = cfg or OrbVoConfig()                                      # configuration
    return MonoVO(seq.camera, seq.calib.size, replace(base, feature="sift" if frontend == "sift" else "orb"))   # front end


def run(name: str, max_frames: int | None = None, cfg=None, verbose: bool = True, frontend: str = "lk"):
    seq = Sequence(name)                                             # load the sequence
    vo = make_vo(frontend, seq, cfg)                                 # front end
    n = len(seq) if max_frames is None else min(max_frames, len(seq))   # frames to process
    T = np.tile(np.eye(4), (n, 1, 1))                                # poses (identity where invalid)
    status = np.full(n, UNINIT, np.int8)                             # per-frame status
    cols = {k: np.zeros(n, np.int32) for k in ("map_id", "n_inl", "n_map", "n_match", "n_kp", "n_new")}   # per-frame counts
    t0 = time.time()                                                 # start timer
    for i in range(n):                                               # frame loop
        pose, st, info = vo.process(seq.frame_u8(i))                 # process one frame
        status[i] = st                                               # status
        for k in cols:                                               # log counters (LK has no n_match / n_kp)
            cols[k][i] = info.get(k, info.get("n_tracks", 0) if k in ("n_match", "n_kp") else 0)
        if pose is not None:                                         # valid or predicted pose
            T[i] = pose                                              # store
        if verbose and i % 300 == 0:                                 # progress
            print(f"frame {i:5d}  status {st}  kp {cols['n_kp'][i]:4d}  match {cols['n_match'][i]:4d}  "
                  f"inl {cols['n_inl'][i]:4d}  map {cols['n_map'][i]:5d}")   # line
    dt = time.time() - t0                                            # elapsed
    out = dict(t_ns=seq.cam.t_ns[:n], T_w_c=T, status=status, fps=np.float64(n / dt), **cols)   # results
    if verbose:                                                      # summary
        print(f"[{frontend}] processed {n} frames in {dt:.1f}s ({n / dt:.1f} fps)")   # speed
        print(f"OK {int((status == OK).sum())}  LOST {int((status == LOST).sum())}  UNINIT {int((status == UNINIT).sum())}  "
              f"map segments {int(cols['map_id'].max()) + 1}")       # counts
    return out, seq                                                  # results and sequence


def out_path(dataset: str, frontend: str):                           # result file for a front end
    tag = "" if frontend == "lk" else f"{frontend}_"                 # lk keeps the original name
    return PROJECT_ROOT / "results" / f"vo_{tag}{dataset}.npz"       # path


def main():                                                          # CLI entry
    ap = argparse.ArgumentParser()                                   # parser
    ap.add_argument("--dataset", default="tumvi_room1")              # dataset key
    ap.add_argument("--frontend", default="lk", choices=["lk", "orb", "sift"])   # which front end
    ap.add_argument("--max-frames", type=int, default=None)          # optional limit
    a = ap.parse_args()                                              # parse
    out, _ = run(a.dataset, a.max_frames, None, True, a.frontend)    # run
    (PROJECT_ROOT / "results").mkdir(exist_ok=True)                  # ensure
    p = out_path(a.dataset, a.frontend)                              # output file
    np.savez(p, **out)                                               # save arrays
    print(f"saved {p}")                                              # report


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
