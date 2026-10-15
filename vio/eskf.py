"""Error-state EKF: IMU propagation + relative-pose update from monocular VO.

State (nominal): position p, velocity v, orientation R (body -> world), accel bias ba,
gyro bias bg, scale s (metres per VO unit). Error state is 16-D (3 for orientation):
  [dp(0:3) dv(3:6) dth(6:9) dba(9:12) dbg(12:15) ds(15)]
A 6-D stochastic clone [dp_i(16:19) dth_i(19:22)] of the pose at the previous VO frame
makes the relative-pose measurement a proper function of two states. No ROS imports.
"""
from __future__ import annotations                                   # forward type hints
from dataclasses import dataclass                                    # config container
import numpy as np                                                   # linear algebra
from scipy.spatial.transform import Rotation                         # rotation log map

G_W = np.array([0.0, 0.0, -9.81])                                    # gravity in the z-up world frame
NX, NC = 16, 6                                                       # main and clone error dimensions
SL_P, SL_V, SL_TH, SL_BA, SL_BG, SL_S = (slice(0, 3), slice(3, 6), slice(6, 9),
                                         slice(9, 12), slice(12, 15), 15)   # error-state slices
SL_CP, SL_CTH = slice(16, 19), slice(19, 22)                         # clone slices


def skew(w: np.ndarray) -> np.ndarray:                               # cross-product matrix
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=np.float64)   # 3x3


def so3_exp(w: np.ndarray) -> np.ndarray:                            # rotation vector -> matrix
    th = float(np.linalg.norm(w))                                    # angle
    if th < 1e-9:                                                    # tiny angle
        return np.eye(3) + skew(w)                                   # first order
    K = skew(w / th)                                                 # unit axis
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)   # Rodrigues


def so3_log(R: np.ndarray) -> np.ndarray:                            # rotation matrix -> vector
    return Rotation.from_matrix(R).as_rotvec()                       # via scipy


@dataclass                                                           # filter tunables
class FilterConfig:
    sigma_r: float = 0.01                                            # rotation measurement noise [rad]
    sigma_t_rel: float = 0.02                                        # translation noise, fraction of |t| (tuned on room1)
    sigma_t_floor: float = 0.005                                     # translation noise floor [m]
    gate_chi2: float = 12.59                                         # chi2(6 dof) 95%: above this noise is inflated
    update_every: int = 5                                            # VO frames between clone and update (baseline)
    n_scale_init: int = 4                                            # VO pairs used to initialise scale
    scale_init_rel_sigma: float = 0.3                                # relative sigma of s after init
    scale_rw: float = 0.05                                           # scale random walk (relative / sqrt(s))
    p0_sigma: float = 0.01                                           # initial position sigma [m]
    v0_sigma: float = 0.05                                           # initial velocity sigma [m/s]
    th0_sigma: float = 0.01                                          # initial orientation sigma [rad]
    ba0_sigma: float = 0.2                                           # initial accel bias sigma [m/s^2]
    bg0_sigma: float = 0.02                                          # initial gyro bias sigma [rad/s]


class ESKF:                                                          # the filter
    def __init__(self, p, v, R, noise: dict, T_cam_imu: np.ndarray, cfg: FilterConfig | None = None,
                 s: float = 1.0, estimate_scale: bool = True):      # initial state and models
        self.cfg = cfg or FilterConfig()                             # configuration
        self.p, self.v, self.R = np.array(p, float), np.array(v, float), np.array(R, float)   # nominal pose
        self.ba, self.bg = np.zeros(3), np.zeros(3)                  # biases start at zero
        self.s = float(s)                                            # scale
        self.estimate_scale = estimate_scale                         # False -> s frozen (4a)
        self.noise = noise                                           # IMU noise densities
        self.Rc = T_cam_imu[:3, :3].copy()                           # IMU -> camera rotation
        self.c = -self.Rc.T @ T_cam_imu[:3, 3]                       # camera origin in IMU frame
        self.R_ic = self.Rc.T                                        # camera -> IMU rotation
        c = self.cfg                                                 # shorthand
        d = np.concatenate([np.full(3, c.p0_sigma), np.full(3, c.v0_sigma), np.full(3, c.th0_sigma),
                            np.full(3, c.ba0_sigma), np.full(3, c.bg0_sigma), [1.0]])   # sigmas
        self.P = np.zeros((NX + NC, NX + NC))                        # augmented covariance
        self.P[:NX, :NX] = np.diag(d ** 2)                           # initial covariance
        self.clone_p = None                                          # clone position
        self.clone_R = None                                          # clone rotation
        self.n_gated = 0                                             # rejected updates

    def propagate(self, dt: float, gyro: np.ndarray, acc: np.ndarray):   # IMU step of length dt
        w = gyro - self.bg                                           # bias-corrected angular rate
        a = acc - self.ba                                            # bias-corrected specific force
        a_w = self.R @ a + G_W                                       # world acceleration
        F = np.eye(NX + NC)                                          # transition matrix
        F[SL_P, SL_V] = np.eye(3) * dt                               # dp <- dv
        F[SL_V, SL_TH] = -self.R @ skew(a) * dt                      # dv <- dth
        F[SL_V, SL_BA] = -self.R * dt                                # dv <- dba
        dR = so3_exp(w * dt)                                         # incremental rotation
        F[SL_TH, SL_TH] = dR.T                                       # dth <- dth
        F[SL_TH, SL_BG] = -np.eye(3) * dt                            # dth <- dbg
        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt          # position
        self.v = self.v + a_w * dt                                   # velocity
        self.R = self.R @ dR                                         # orientation
        n = self.noise                                               # noise densities
        Q = np.zeros(NX + NC)                                        # diagonal process noise
        Q[SL_V] = n["acc_nd"] ** 2 * dt                              # velocity random walk
        Q[SL_TH] = n["gyr_nd"] ** 2 * dt                             # angle random walk
        Q[SL_BA] = n["acc_rw"] ** 2 * dt                             # accel bias random walk
        Q[SL_BG] = n["gyr_rw"] ** 2 * dt                             # gyro bias random walk
        Q[SL_S] = (self.cfg.scale_rw * self.s) ** 2 * dt if self.estimate_scale else 0.0   # scale random walk
        self.P = F @ self.P @ F.T                                    # covariance propagation
        self.P[np.diag_indices(NX + NC)] += Q                        # add process noise

    def clone(self):                                                 # remember the current pose
        self.clone_p, self.clone_R = self.p.copy(), self.R.copy()    # nominal clone
        J = np.zeros((NC, NX + NC))                                  # selects dp, dth
        J[0:3, SL_P] = np.eye(3)                                     # dp
        J[3:6, SL_TH] = np.eye(3)                                    # dth
        PJ = J @ self.P                                              # cross rows
        self.P[NX:, :] = PJ                                          # clone rows
        self.P[:, NX:] = PJ.T                                        # clone columns
        self.P[NX:, NX:] = J @ self.P @ J.T                          # clone block

    def drop_clone(self):                                            # forget the clone
        self.clone_p = self.clone_R = None                           # nominal
        self.P[NX:, :] = 0.0                                         # rows
        self.P[:, NX:] = 0.0                                         # columns

    def has_clone(self) -> bool:                                     # clone available?
        return self.clone_R is not None                              # flag

    def cam_pose(self) -> np.ndarray:                                # T_w_c from the nominal state
        T = np.eye(4)                                                # allocate
        T[:3, :3] = self.R @ self.R_ic                               # camera orientation
        T[:3, 3] = self.p + self.R @ self.c                          # camera centre
        return T                                                     # 4x4

    def predict_relative(self):                                      # (R_pred, t_pred) from clone to now
        Rwci = self.clone_R @ self.R_ic                              # camera i orientation
        Ci = self.clone_p + self.clone_R @ self.c                    # camera i centre
        Rwcj = self.R @ self.R_ic                                    # camera j orientation
        Cj = self.p + self.R @ self.c                                # camera j centre
        return Rwci.T @ Rwcj, Rwci.T @ (Cj - Ci)                     # rotation, translation in camera i frame

    def set_scale(self, s: float, sigma: float):                     # (re)initialise the scale state
        self.s = float(s)                                            # value
        self.P[SL_S, :] = 0.0                                        # clear row
        self.P[:, SL_S] = 0.0                                        # clear column
        self.P[SL_S, SL_S] = sigma ** 2                              # variance

    def update_relative(self, R_m: np.ndarray, t_m: np.ndarray, use_t: bool = True):   # VO measurement
        cfg = self.cfg                                               # shorthand
        Rwci = self.clone_R @ self.R_ic                              # camera i orientation
        R_pred, t_pred = self.predict_relative()                     # prediction
        s_eff = self.s if self.estimate_scale else 1.0               # scale applied to the measurement
        tm = s_eff * t_m                                             # metric translation measurement
        r_rot = so3_log(R_pred.T @ R_m)                              # rotation residual
        r_tr = tm - t_pred                                           # translation residual
        H = np.zeros((6, NX + NC))                                   # measurement Jacobian
        H[0:3, SL_TH] = self.Rc                                      # d r_rot / d dth_j
        H[0:3, SL_CTH] = -R_pred.T @ self.Rc                         # d r_rot / d dth_i
        Rt = Rwci.T                                                  # R_ci^T
        H[3:6, SL_P] = Rt                                            # d r_tr / d dp_j
        H[3:6, SL_CP] = -Rt                                          # d r_tr / d dp_i
        H[3:6, SL_TH] = -Rt @ self.R @ skew(self.c)                  # d r_tr / d dth_j (lever arm)
        H[3:6, SL_CTH] = Rt @ self.clone_R @ skew(self.c) + skew(t_pred) @ self.Rc   # d r_tr / d dth_i
        if self.estimate_scale:                                      # scale enters the translation row
            H[3:6, SL_S] = -t_m                                      # d r_tr / d ds (sign as derived)
        sig_t = cfg.sigma_t_rel * np.linalg.norm(tm) + cfg.sigma_t_floor   # translation noise [m]
        Rn = np.diag([cfg.sigma_r ** 2] * 3 + [sig_t ** 2] * 3)      # measurement covariance
        y = np.concatenate([r_rot, r_tr])                            # residual
        if not use_t:                                                # rotation-only update
            H, y, Rn = H[:3], y[:3], Rn[:3, :3]                      # first 3 rows
        S = H @ self.P @ H.T + Rn                                    # innovation covariance
        d2 = float(y @ np.linalg.solve(S, y))                        # Mahalanobis distance squared
        gate = cfg.gate_chi2 if use_t else 7.81                      # chi2 95% (6 or 3 dof)
        if d2 > gate:                                                # inconsistent measurement
            self.n_gated += 1                                        # count (noise is inflated, not discarded)
            HPHt = S - Rn                                            # predicted part of the innovation covariance
            lam = min(d2 / gate, 1e4)                                # inflation factor (> 1 here), bounded
            Rn = Rn * lam                                            # downweight the measurement
            S = HPHt + Rn                                            # recompute
        K = self.P @ H.T @ np.linalg.inv(S)                          # Kalman gain
        dx = K @ y                                                   # error-state correction
        self.p = self.p + dx[SL_P]                                   # position
        self.v = self.v + dx[SL_V]                                   # velocity
        self.R = self.R @ so3_exp(dx[SL_TH])                         # orientation
        self.ba = self.ba + dx[SL_BA]                                # accel bias
        self.bg = self.bg + dx[SL_BG]                                # gyro bias
        if self.estimate_scale:                                      # scale
            self.s = max(self.s + dx[SL_S], 1e-4)                    # keep positive
        I = np.eye(NX + NC)                                          # identity
        A = I - K @ H                                                # Joseph form
        self.P = A @ self.P @ A.T + K @ Rn @ K.T                     # covariance update
        self.P = 0.5 * (self.P + self.P.T)                           # keep symmetric
        return True, d2                                              # accepted
