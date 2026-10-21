"""Verify the NumPy Umeyama / ATE / RPE in eval/ against the evo library (reports every difference).

1. Umeyama on random point sets (with and without scale, including a reflection-prone configuration)
2. ATE of a real VO and fused trajectory: eval.ate.ate_rmse vs evo.core.metrics.APE
3. RPE (1 s and 5 s, all pairs): eval.metrics.rpe vs evo.core.metrics.RPE
"""
from __future__ import annotations                                   # forward type hints
import json                                                          # metrics file
import numpy as np                                                   # arrays
from scipy.spatial.transform import Rotation                         # quaternions
from evo.core import geometry, metrics, sync                         # evo internals
from evo.core.trajectory import PoseTrajectory3D                     # evo trajectory

from vo.datasets import Sequence, PROJECT_ROOT                       # data access
from vo.frontend_lk import OK                                        # status code
from vio.run_vio import run_filter                                   # filter driver
from vo.run_vo import out_path                                       # VO result naming
from eval.ate import umeyama, ate_rmse                               # my implementations
from eval.metrics import rpe                                         # my RPE


def to_evo(t_ns: np.ndarray, T: np.ndarray) -> PoseTrajectory3D:     # poses -> evo trajectory
    q = Rotation.from_matrix(T[:, :3, :3]).as_quat()                 # x, y, z, w
    return PoseTrajectory3D(positions_xyz=T[:, :3, 3], orientations_quat_wxyz=q[:, [3, 0, 1, 2]],
                            timestamps=t_ns * 1e-9)                  # evo wants w first and seconds


def part1_random() -> dict:                                          # random-data comparison
    rng = np.random.default_rng(0)                                   # reproducible
    worst = {"R": 0.0, "t": 0.0, "s": 0.0}                           # largest differences
    for trial in range(200):                                         # many random problems
        n = int(rng.integers(5, 400))                                # point count
        x = rng.normal(size=(n, 3)) * rng.uniform(0.1, 5, 3)         # anisotropic source cloud
        Rg = Rotation.random(random_state=int(rng.integers(1e9))).as_matrix()   # random rotation
        sg, tg = rng.uniform(0.05, 20), rng.normal(size=3) * 3       # random scale and translation
        y = (sg * (Rg @ x.T)).T + tg + rng.normal(scale=0.01, size=(n, 3))   # noisy target
        for ws in (True, False):                                     # with and without scale
            s1, R1, t1 = umeyama(x, y, ws)                           # mine
            R2, t2, s2 = geometry.umeyama_alignment(x.T.copy(), y.T.copy(), ws)   # evo
            worst["R"] = max(worst["R"], float(np.abs(R1 - R2).max()))   # rotation
            worst["t"] = max(worst["t"], float(np.abs(t1 - t2).max()))   # translation
            worst["s"] = max(worst["s"], abs(s1 - s2))               # scale
    return worst                                                     # maxima over all trials


def main(dataset: str = "tumvi_room1", frontend: str = "sift") -> dict:   # run the verification
    out = {"random_umeyama_max_abs_diff": part1_random()}            # part 1
    print("1. Umeyama on 200 random problems x {scale, no scale}: max |diff| vs evo =", out["random_umeyama_max_abs_diff"])   # report
    seq = Sequence(dataset)                                          # data
    vo = dict(np.load(out_path(dataset, frontend)))                  # VO output
    fu = run_filter(seq, vo, "estscale", verbose=False)              # fused
    t, st, mid = vo["t_ns"], vo["status"], vo["map_id"]              # arrays
    gt, valid = fu["gt_cam"], fu["valid"]                            # GT and filter validity
    okv = (st == OK) & valid                                         # frames used for scoring
    seg = int(np.bincount(mid[okv]).argmax())                        # longest VO segment
    sel = okv & (mid == seg)                                         # its frames
    res = {}                                                         # per-case numbers
    cases = {"vision_only_longest_segment": (t[sel], vo["T_w_c"][sel], gt[sel]),
             "fused_all_valid_frames": (t[valid], fu["T_w_c"][valid], gt[valid])}   # trajectories to test
    for name, (tt, Te, Tg) in cases.items():                         # each case
        ref, est = to_evo(tt, Tg), to_evo(tt, Te)                    # evo trajectories (timestamps identical)
        r = {}                                                       # numbers for this case
        for lab, ws, mode in (("se3", False, "se3"), ("sim3", True, "sim3")):   # alignment modes
            mine = ate_rmse(Te[:, :3, 3], Tg[:, :3, 3], mode)[0]     # my ATE
            e2 = PoseTrajectory3D(positions_xyz=est.positions_xyz.copy(), orientations_quat_wxyz=est.orientations_quat_wxyz.copy(),
                                  timestamps=est.timestamps.copy())  # copy before aligning
            e2.align(ref, correct_scale=ws)                          # evo alignment
            ape = metrics.APE(metrics.PoseRelation.translation_part)   # evo ATE
            ape.process_data((ref, e2))                              # run
            ev = float(ape.get_statistic(metrics.StatisticsType.rmse))   # evo RMSE
            r[f"ate_{lab}"] = {"mine": mine, "evo": ev, "abs_diff": abs(mine - ev)}   # compare
        for lab, d in (("1s", 20), ("5s", 100)):                     # RPE intervals
            mine = rpe(Te, Tg, np.ones(len(Te), bool), d)[0]         # my translational RPE (all pairs)
            rp = metrics.RPE(metrics.PoseRelation.translation_part, float(d), metrics.Unit.frames, 0.0, True)   # evo RPE
            rp.process_data((ref, est))                              # run (no alignment, same as mine)
            ev = float(rp.get_statistic(metrics.StatisticsType.rmse))   # evo RMSE
            m_rmse = float(np.sqrt(np.mean(mine ** 2)))              # my RMSE
            r[f"rpe_trans_{lab}"] = {"mine": m_rmse, "evo": ev, "abs_diff": abs(m_rmse - ev), "n_mine": int(len(mine)), "n_evo": int(len(rp.error))}   # compare
        res[name] = r                                                # store
        print(f"2/3. {name}:")                                       # header
        for k, v in r.items():                                       # lines
            print(f"     {k:14s} mine {v['mine']:.6f}  evo {v['evo']:.6f}  |diff| {v['abs_diff']:.2e}"
                  + (f"  pairs mine/evo {v['n_mine']}/{v['n_evo']}" if "n_mine" in v else ""))   # report
    out["trajectory_comparison"] = res                               # store
    json.dump(out, open(PROJECT_ROOT / "results" / f"verify_evo_{dataset}.json", "w", encoding="utf-8"), indent=2)   # save
    return out                                                       # numbers


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
