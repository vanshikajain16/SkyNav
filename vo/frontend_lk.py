"""Monocular VO front end: LK tracking, a sparse map, PnP pose, triangulation of new points.

Plain Python, no ROS. The map defines an arbitrary metric scale (1 VO unit = the baseline
of the initialization pair), kept consistent by PnP against triangulated map points.
"""
from __future__ import annotations                                   # forward type hints
from dataclasses import dataclass                                    # config container
import numpy as np                                                   # arrays
import cv2                                                           # computer vision

OK, LOST, UNINIT = 0, 1, 2                                           # per-frame status codes


@dataclass                                                           # tunables in one place
class VoConfig:
    max_corners: int = 1000                                          # corners per detection call
    min_tracks: int = 400                                            # re-detect when below this
    max_tracks: int = 1200                                           # hard cap on live tracks
    min_distance_px: int = 14                                        # spacing between corners
    quality: float = 0.005                                           # goodFeaturesToTrack quality
    mask_radius_px: float = 225.0                                    # central fisheye mask radius
    use_clahe: bool = True                                           # contrast boost (frames are dark)
    lk_win: int = 31                                                 # LK window size
    lk_levels: int = 3                                               # LK pyramid levels
    fb_thresh_px: float = 1.0                                        # forward-backward limit
    ransac_thr_px: float = 1.5                                       # RANSAC / reprojection threshold
    init_parallax_px: float = 20.0                                   # median flow needed to initialize
    init_min_tracks: int = 60                                        # tracks needed to try init
    init_min_points: int = 40                                        # triangulated points to accept init
    init_min_angle_deg: float = 2.0                                  # median triangulation angle at init
    depth_ratio_max: float = 4.0                                     # new points deeper than this x median depth rejected
    depth_ratio_min: float = 0.1                                     # ... or nearer than this x median depth
    min_pnp_points: int = 6                                          # 3D tracks needed for PnP
    min_pnp_inliers: int = 8                                         # inliers needed to accept PnP
    min_tri_angle_deg: float = 2.0                                   # parallax needed for new map point
    max_lost: int = 10                                               # lost frames before re-initialising


def _inv(T: np.ndarray) -> np.ndarray:                               # inverse of a rigid transform
    R, t = T[:3, :3], T[:3, 3]                                       # split
    out = np.eye(4)                                                  # allocate
    out[:3, :3] = R.T                                                # inverse rotation
    out[:3, 3] = -R.T @ t                                            # inverse translation
    return out                                                       # 4x4


def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:                  # build 4x4 from R, t
    out = np.eye(4)                                                  # allocate
    out[:3, :3] = R                                                  # rotation
    out[:3, 3] = np.asarray(t).reshape(3)                            # translation
    return out                                                       # 4x4


class MonoVO:                                                        # streaming monocular VO
    def __init__(self, camera, size, cfg: VoConfig | None = None):   # camera model + image size
        self.cam = camera                                            # camera model
        self.cfg = cfg or VoConfig()                                 # configuration
        self.fx = float(camera.K[0, 0])                              # focal length in pixels
        self.thr = self.cfg.ransac_thr_px / self.fx                  # threshold in normalized units
        self.size = size                                             # (w, h)
        self.mask = np.zeros((size[1], size[0]), np.uint8)           # central mask image
        cv2.circle(self.mask, (int(camera.K[0, 2]), int(camera.K[1, 2])),
                   int(self.cfg.mask_radius_px), 255, -1)            # keep only the central disc
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))   # contrast equaliser
        self.frame_idx = -1                                          # index of the last processed frame
        self.map_id = 0                                              # increments on re-initialisation
        self.hist = {}                                               # frame idx -> T_w_c (valid frames)
        self._reset()                                                # clear tracks and map

    def _reset(self):                                                # forget tracks, map, state
        self.state = "init"                                          # waiting for initialization
        self.prev_gray = None                                        # previous image
        self.pts = np.zeros((0, 2), np.float32)                      # current pixels of tracks
        self.p0 = np.zeros((0, 2), np.float32)                       # first-observation pixels
        self.n0 = np.zeros((0, 2), np.float64)                       # first-observation normalized
        self.f0 = np.zeros((0,), np.int64)                           # first-observation frame idx
        self.X = np.zeros((0, 3), np.float64)                        # 3D map point (NaN if none)
        self.n_lost = 0                                              # consecutive lost frames
        self.T_cw_prev = None                                        # last valid T_c_w
        self.T_cw_prev2 = None                                       # one before that

    def _prep(self, gray: np.ndarray) -> np.ndarray:                 # optional contrast enhancement
        return self.clahe.apply(gray) if self.cfg.use_clahe else gray   # CLAHE on 8-bit image

    def _detect(self, gray: np.ndarray, n_max: int) -> np.ndarray:   # new corners away from existing
        m = self.mask.copy()                                         # start from central mask
        for x, y in self.pts:                                        # exclude neighbourhoods of tracks
            cv2.circle(m, (int(x), int(y)), self.cfg.min_distance_px, 0, -1)   # blank them
        c = cv2.goodFeaturesToTrack(gray, n_max, self.cfg.quality,
                                    self.cfg.min_distance_px, mask=m, blockSize=7)   # corners
        return np.zeros((0, 2), np.float32) if c is None else c.reshape(-1, 2).astype(np.float32)   # (N,2)

    def _add_tracks(self, px: np.ndarray, idx: int):                 # append new 2D-only tracks
        if len(px) == 0:                                             # nothing to add
            return                                                   # done
        n = self.cam.undistort_to_normalized(px)                     # normalized coordinates
        self.pts = np.vstack([self.pts, px])                         # current pixels
        self.p0 = np.vstack([self.p0, px])                           # first observation pixels
        self.n0 = np.vstack([self.n0, n])                            # first observation normalized
        self.f0 = np.concatenate([self.f0, np.full(len(px), idx, np.int64)])   # first frame
        self.X = np.vstack([self.X, np.full((len(px), 3), np.nan)])  # no 3D yet

    def _keep(self, keep: np.ndarray):                               # drop tracks by boolean mask
        self.pts, self.p0, self.n0 = self.pts[keep], self.p0[keep], self.n0[keep]   # 2D data
        self.f0, self.X = self.f0[keep], self.X[keep]                # indices and 3D

    def _lk(self, gray: np.ndarray):                                 # track all points prev -> cur
        if len(self.pts) == 0:                                       # nothing to track
            return                                                   # done
        p = self.pts.reshape(-1, 1, 2)                               # OpenCV layout
        w = (self.cfg.lk_win, self.cfg.lk_win)                       # window
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)   # LK termination
        q, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p, None, winSize=w,
                                            maxLevel=self.cfg.lk_levels, criteria=crit)   # forward
        b, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, q, None, winSize=w,
                                             maxLevel=self.cfg.lk_levels, criteria=crit)  # backward
        fb = np.linalg.norm((b - p).reshape(-1, 2), axis=1)          # forward-backward error
        q = q.reshape(-1, 2)                                         # tracked points
        inside = self.mask[np.clip(q[:, 1].astype(int), 0, self.size[1] - 1),
                           np.clip(q[:, 0].astype(int), 0, self.size[0] - 1)] > 0   # inside the central disc
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.cfg.fb_thresh_px) & inside   # all checks
        self.pts = q.astype(np.float32)                              # store new positions
        self._keep(ok)                                               # drop failed tracks

    def _triangulate(self, n1, n2, T_c1_w, T_c2_w):                  # linear triangulation
        Xh = cv2.triangulatePoints(T_c1_w[:3], T_c2_w[:3], n1.T, n2.T)   # homogeneous (4,N)
        return (Xh[:3] / Xh[3]).T                                    # (N,3)

    def _check(self, X, n, T_cw):                                    # depth and reprojection of X in camera
        Xc = (T_cw[:3, :3] @ X.T + T_cw[:3, 3:4]).T                  # points in camera frame
        z = Xc[:, 2]                                                 # depth
        proj = Xc[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]   # normalized projection
        err = np.linalg.norm(proj - n, axis=1) * self.fx             # reprojection error in pixels
        return z, err                                                # depth and error

    def _try_init(self, n_cur: np.ndarray) -> bool:                  # essential-matrix initialisation
        n_ref = self.n0                                              # reference observations
        E, m = cv2.findEssentialMat(n_ref, n_cur, np.eye(3), cv2.RANSAC, 0.999, self.thr)   # E + mask
        if E is None or E.shape[0] != 3:                             # degenerate or stacked solutions
            if E is None or E.shape[0] < 3:                          # nothing usable
                return False                                         # fail
            E = E[:3]                                                # take the first solution
        n_in, R, t, mp = cv2.recoverPose(E, n_ref, n_cur, np.eye(3), mask=m)   # cheirality
        if n_in < self.cfg.init_min_points:                          # too few
            return False                                             # fail
        T_c2_c1 = _T(R, t)                                           # |t| = 1 VO unit
        sel = mp.ravel() > 0                                         # inliers after cheirality
        X = self._triangulate(n_ref[sel], n_cur[sel], np.eye(4), T_c2_c1)   # in reference frame
        z1, e1 = self._check(X, n_ref[sel], np.eye(4))               # reference check
        z2, e2 = self._check(X, n_cur[sel], T_c2_c1)                 # current check
        r1 = X / np.linalg.norm(X, axis=1, keepdims=True)            # ray from camera 1
        c2 = _inv(T_c2_c1)[:3, 3]                                    # camera 2 centre in frame 1
        r2 = X - c2                                                  # ray from camera 2
        r2 = r2 / np.linalg.norm(r2, axis=1, keepdims=True)          # normalise
        ang = np.degrees(np.arccos(np.clip(np.sum(r1 * r2, axis=1), -1, 1)))   # triangulation angle
        good = (z1 > 0) & (z2 > 0) & (e1 < 3 * self.cfg.ransac_thr_px) & (e2 < 3 * self.cfg.ransac_thr_px)   # valid
        if good.sum() < self.cfg.init_min_points or np.median(ang[good]) < self.cfg.init_min_angle_deg:
            return False                                             # not enough parallax or points
        idx = np.where(sel)[0][good]                                 # indices into tracks
        self.X[:] = np.nan                                           # clear
        self.X[idx] = X[good]                                        # map points in reference frame
        keep = np.zeros(len(self.pts), bool)                         # keep only mapped tracks
        keep[idx] = True                                             # mapped
        f_ref = int(self.f0[idx[0]])                                 # reference frame index
        self.hist[f_ref] = np.eye(4)                                 # world = reference camera
        self.T_cw_prev2 = np.eye(4)                                  # velocity model start
        self.T_cw_prev = T_c2_c1.copy()                              # current camera pose
        self._keep(keep)                                             # drop unmapped tracks
        return True                                                  # initialised

    def _pnp(self, n_cur: np.ndarray):                               # pose from map points
        has = ~np.isnan(self.X[:, 0])                                # tracks with 3D
        if has.sum() < self.cfg.min_pnp_points:                      # too few
            return None                                              # fail
        guess = self.T_cw_prev                                       # previous pose
        if self.T_cw_prev2 is not None:                              # constant-velocity prediction
            guess = (self.T_cw_prev @ _inv(self.T_cw_prev2)) @ self.T_cw_prev   # D * T_prev
        rvec = cv2.Rodrigues(guess[:3, :3])[0]                       # guess rotation vector
        tvec = guess[:3, 3].reshape(3, 1).copy()                     # guess translation
        Xs = np.ascontiguousarray(self.X[has])                       # 3D points
        xs = np.ascontiguousarray(n_cur[has])                        # 2D normalized observations
        ok, rvec, tvec, inl = cv2.solvePnPRansac(Xs, xs, np.eye(3), np.zeros((4, 1)), rvec, tvec,
                                                 True, 200, self.thr, 0.995, flags=cv2.SOLVEPNP_ITERATIVE)  # robust PnP
        if not ok or inl is None or len(inl) < self.cfg.min_pnp_inliers:   # failure
            return None                                              # fail
        inl = inl.ravel()                                            # inlier indices
        rvec, tvec = cv2.solvePnPRefineLM(Xs[inl], xs[inl], np.eye(3), np.zeros((4, 1)), rvec, tvec)   # refine
        T_cw = _T(cv2.Rodrigues(rvec)[0], tvec)                      # world -> camera
        idx_has = np.where(has)[0]                                   # indices of tracks with 3D
        outl = np.ones(len(idx_has), bool)                           # start: all outliers
        outl[inl] = False                                            # inliers are fine
        return T_cw, idx_has[outl], len(inl)                         # pose, outlier tracks, inlier count

    def _new_points(self, idx: int, T_cw: np.ndarray):               # triangulate tracks that gained parallax
        cand = np.where(np.isnan(self.X[:, 0]))[0]                   # 2D-only tracks
        if len(cand) == 0:                                           # nothing to do
            return 0                                                 # none added
        n_cur = self.cam.undistort_to_normalized(self.pts[cand])     # current normalized
        mapped = ~np.isnan(self.X[:, 0])                             # tracks that already have 3D
        if mapped.sum() < 10:                                        # no reliable depth reference
            return 0                                                 # do not grow from nothing
        zref = np.median((T_cw[:3, :3] @ self.X[mapped].T + T_cw[:3, 3:4])[2])   # median depth of the map
        added = 0                                                    # counter
        for f in np.unique(self.f0[cand]):                           # group by first frame
            if f == idx or f not in self.hist:                       # no baseline or no pose
                continue                                             # skip
            sel = self.f0[cand] == f                                 # tracks first seen at f
            T_w_c0 = self.hist[int(f)]                               # pose at first observation
            T_c0_w = _inv(T_w_c0)                                    # world -> camera at f
            Xw = self._triangulate(self.n0[cand[sel]], n_cur[sel], T_c0_w, T_cw)   # triangulate
            z0, e0 = self._check(Xw, self.n0[cand[sel]], T_c0_w)     # check first view
            z1, e1 = self._check(Xw, n_cur[sel], T_cw)               # check current view
            c0 = T_w_c0[:3, 3]                                       # first camera centre
            c1 = _inv(T_cw)[:3, 3]                                   # current camera centre
            a = Xw - c0                                              # ray from first camera
            b = Xw - c1                                              # ray from current camera
            cosang = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)   # cosine
            ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))      # triangulation angle
            ok = (z0 > 0) & (z1 > 0) & (e0 < 2 * self.cfg.ransac_thr_px) & (e1 < 2 * self.cfg.ransac_thr_px) \
                 & (ang > self.cfg.min_tri_angle_deg) \
                 & (z1 < self.cfg.depth_ratio_max * zref) & (z1 > self.cfg.depth_ratio_min * zref)   # acceptance test
            ids = cand[sel][ok]                                      # accepted tracks
            self.X[ids] = Xw[ok]                                     # store 3D
            added += int(ok.sum())                                   # count
        return added                                                 # how many new map points

    def process(self, gray_u8: np.ndarray):                          # one frame -> (T_w_c|None, status, info)
        self.frame_idx += 1                                          # advance frame counter
        idx = self.frame_idx                                         # shorthand
        gray = self._prep(gray_u8)                                   # preprocessed image
        info = {"n_tracks": 0, "n_map": 0, "n_inl": 0, "n_new": 0, "map_id": self.map_id}   # diagnostics
        if self.prev_gray is None:                                   # very first frame
            self._add_tracks(self._detect(gray, self.cfg.max_corners), idx)   # seed tracks
            self.prev_gray = gray                                    # remember
            info["n_tracks"] = len(self.pts)                         # report
            return None, UNINIT, info                                # no pose yet
        self._lk(gray)                                               # track existing points
        if self.state == "init":                                     # still initialising
            n_cur = self.cam.undistort_to_normalized(self.pts) if len(self.pts) else np.zeros((0, 2))   # normalized
            par = np.median(np.linalg.norm(self.pts - self.p0, axis=1)) if len(self.pts) else 0.0   # median flow
            if len(self.pts) < self.cfg.init_min_tracks:             # tracks dying: restart from here
                self.pts = np.zeros((0, 2), np.float32)              # clear
                self.p0 = np.zeros((0, 2), np.float32)               # clear
                self.n0 = np.zeros((0, 2), np.float64)               # clear
                self.f0 = np.zeros((0,), np.int64)                   # clear
                self.X = np.zeros((0, 3), np.float64)                # clear
                self._add_tracks(self._detect(gray, self.cfg.max_corners), idx)   # new reference
            elif par >= self.cfg.init_parallax_px and self._try_init(n_cur):   # enough motion: try
                self.state = "track"                                 # initialised
                self.n_lost = 0                                      # reset
                T_w_c = _inv(self.T_cw_prev)                         # current camera pose in map frame
                self.hist[idx] = T_w_c                               # remember
                self.prev_gray = gray                                # update
                info.update(n_tracks=len(self.pts), n_map=int((~np.isnan(self.X[:, 0])).sum()), map_id=self.map_id)   # info
                return T_w_c, OK, info                               # first valid pose
            self.prev_gray = gray                                    # update
            info["n_tracks"] = len(self.pts)                         # report
            return None, UNINIT, info                                # still waiting
        n_cur = self.cam.undistort_to_normalized(self.pts) if len(self.pts) else np.zeros((0, 2))   # normalized
        res = self._pnp(n_cur) if len(self.pts) else None            # pose estimation
        if res is None:                                              # tracking lost this frame
            self.n_lost += 1                                         # count
            status = LOST                                            # flag
            guess = self.T_cw_prev if self.T_cw_prev2 is None else (self.T_cw_prev @ _inv(self.T_cw_prev2)) @ self.T_cw_prev   # prediction
            T_cw = guess                                             # keep predicted pose
            self.T_cw_prev2, self.T_cw_prev = None, T_cw             # drop velocity after a loss
            if self.n_lost > self.cfg.max_lost:                      # give up: re-initialise map
                self.map_id += 1                                     # new map segment
                T_last = _inv(T_cw)                                  # last predicted pose
                keep_hist = idx                                      # for clarity
                self._reset()                                        # clear tracks and map
                self.prev_gray = gray                                # keep image
                self._add_tracks(self._detect(gray, self.cfg.max_corners), keep_hist)   # restart tracks
                info.update(n_tracks=len(self.pts), map_id=self.map_id)   # info
                return None, UNINIT, info                            # wait for new init
        else:                                                        # tracking fine
            T_cw, outl, n_inl = res                                  # unpack
            self.n_lost = 0                                          # reset
            status = OK                                              # good frame
            keep = np.ones(len(self.pts), bool)                      # drop tracks with bad 3D
            keep[outl] = False                                       # outlier tracks
            self._keep(keep)                                         # apply
            self.T_cw_prev2, self.T_cw_prev = self.T_cw_prev, T_cw   # velocity model update
            info["n_inl"] = n_inl                                    # report
        T_w_c = _inv(T_cw)                                           # camera pose in map frame
        if status == OK:                                             # only trusted poses go in history
            self.hist[idx] = T_w_c                                   # store
            info["n_new"] = self._new_points(idx, T_cw)              # grow the map
        if len(self.pts) < self.cfg.min_tracks:                      # replenish tracks
            self._add_tracks(self._detect(gray, min(self.cfg.max_corners, self.cfg.max_tracks - len(self.pts))), idx)   # add
        self.prev_gray = gray                                        # update
        info.update(n_tracks=len(self.pts), n_map=int((~np.isnan(self.X[:, 0])).sum()), map_id=self.map_id)   # info
        return T_w_c, status, info                                   # result
