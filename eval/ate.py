"""Trajectory alignment and absolute trajectory error (ATE), plus TUM-format writers."""
from __future__ import annotations                                   # forward type hints
from pathlib import Path                                             # paths
import numpy as np                                                   # arrays
from scipy.spatial.transform import Rotation                         # quaternion conversion


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):     # best similarity transform src -> dst
    mu_s, mu_d = src.mean(0), dst.mean(0)                            # centroids
    xs, xd = src - mu_s, dst - mu_d                                  # centred points
    cov = xd.T @ xs / len(src)                                       # cross covariance
    U, D, Vt = np.linalg.svd(cov)                                    # SVD
    S = np.eye(3)                                                    # reflection guard
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:                     # would be a reflection
        S[2, 2] = -1                                                 # fix
    R = U @ S @ Vt                                                   # rotation
    var = (xs ** 2).sum() / len(src)                                 # source variance
    s = float((D * np.diag(S)).sum() / var) if with_scale else 1.0   # scale
    t = mu_d - s * R @ mu_s                                          # translation
    return s, R, t                                                   # dst ~ s R src + t


def ate_rmse(p_est: np.ndarray, p_gt: np.ndarray, mode: str = "sim3"):   # RMSE after alignment
    if mode == "none":                                               # no alignment
        s, R, t = 1.0, np.eye(3), np.zeros(3)                        # identity
    else:                                                            # "se3" or "sim3"
        s, R, t = umeyama(p_est, p_gt, with_scale=(mode == "sim3"))  # solve
    al = (s * (R @ p_est.T)).T + t                                   # aligned estimate
    err = np.linalg.norm(al - p_gt, axis=1)                          # per-pose error
    return float(np.sqrt(np.mean(err ** 2))), al, err, s             # rmse, aligned, errors, scale


def fmt_ns(t_ns: int) -> str:                                        # exact seconds string from int ns
    return f"{int(t_ns) // 10**9}.{int(t_ns) % 10**9:09d}"           # no float rounding


def write_tum(path: Path, t_ns: np.ndarray, T_w_c: np.ndarray) -> None:   # TUM trajectory file for evo
    q = Rotation.from_matrix(T_w_c[:, :3, :3]).as_quat()             # x,y,z,w order (TUM)
    with open(path, "w", encoding="utf-8") as f:                     # write file
        for t, T, qq in zip(t_ns, T_w_c, q):                         # one line per pose
            f.write(f"{fmt_ns(t)} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} "
                    f"{qq[0]:.7f} {qq[1]:.7f} {qq[2]:.7f} {qq[3]:.7f}\n")   # tx ty tz qx qy qz qw
