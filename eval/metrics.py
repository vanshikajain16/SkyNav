"""Trajectory metrics: relative pose error (RPE) and helpers shared by all reports."""
from __future__ import annotations                                   # forward type hints
import numpy as np                                                   # arrays
from scipy.spatial.transform import Rotation                         # rotation angle


def rpe(T_est: np.ndarray, T_gt: np.ndarray, valid: np.ndarray, delta: int, seg: np.ndarray | None = None,
        scale: float = 1.0):                                         # relative pose error over `delta` frames
    idx = np.arange(len(T_est) - delta)                              # start indices
    ok = valid[idx] & valid[idx + delta]                             # both ends valid
    if seg is not None:                                              # both ends in the same map segment
        ok &= seg[idx] == seg[idx + delta]                           # same segment
    idx = idx[ok]                                                    # usable pairs
    if len(idx) == 0:                                                # nothing to evaluate
        return np.array([]), np.array([])                            # empty
    dt_e, dr_e = np.empty(len(idx)), np.empty(len(idx))              # outputs
    for n, i in enumerate(idx):                                      # loop over pairs
        Te = np.linalg.inv(T_est[i]) @ T_est[i + delta]              # estimated relative motion
        Tg = np.linalg.inv(T_gt[i]) @ T_gt[i + delta]                # true relative motion
        Te[:3, 3] *= scale                                           # apply scale (VO units -> metres)
        E = np.linalg.inv(Tg) @ Te                                   # error transform
        dt_e[n] = np.linalg.norm(E[:3, 3])                           # translational error [m]
        dr_e[n] = np.degrees(np.linalg.norm(Rotation.from_matrix(E[:3, :3]).as_rotvec()))   # rotational error [deg]
    return dt_e, dr_e                                                # arrays


def summarize(x: np.ndarray) -> dict:                                # rmse / mean / median / p95 of an error array
    if len(x) == 0:                                                  # empty guard
        return {"n": 0}                                              # nothing
    return {"n": int(len(x)), "rmse": float(np.sqrt(np.mean(x ** 2))), "mean": float(np.mean(x)),
            "median": float(np.median(x)), "p95": float(np.percentile(x, 95))}   # numbers
