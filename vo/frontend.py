"""Monocular VO front end with descriptor features (ORB, SIFT optional), a persistent map and PnP.

Pipeline: detect (central circular mask) -> undistort keypoints with the dataset camera model ->
essential-matrix bootstrap -> triangulated map -> guided descriptor matching + solvePnPRansac ->
keyframe triangulation of new points -> pruning by reprojection error, cheirality and staleness.
Poses are in VO units (|t| of the initialisation pair = 1). Plain Python, no ROS.
The earlier LK-tracking version lives in vo/frontend_lk.py.
"""
from __future__ import annotations                                   # forward type hints
from dataclasses import dataclass                                    # config container
import numpy as np                                                   # arrays
import cv2                                                           # computer vision
from scipy.spatial import cKDTree                                    # radius search for guided matching

from vo.frontend_lk import OK, LOST, UNINIT, _inv, _T                # shared status codes and helpers

_POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)   # popcount table for Hamming distance


@dataclass                                                           # tunables in one place
class OrbVoConfig:
    feature: str = "orb"                                             # "orb" or "sift"
    n_features: int = 1500                                           # keypoints per frame
    mask_radius_px: float = 225.0                                    # central circular mask radius
    use_clahe: bool = True                                           # contrast boost (frames are dark)
    ratio: float = 0.8                                               # Lowe ratio test
    max_hamming: int = 80                                            # absolute ORB distance limit (of 256 bits)
    refresh_desc: bool = True                                        # replace a map descriptor by its latest observation
    max_l2: float = 250.0                                            # absolute SIFT distance limit
    ransac_thr_px: float = 1.5                                       # RANSAC / reprojection threshold
    init_min_matches: int = 80                                       # matches needed to try initialisation
    init_parallax_px: float = 15.0                                   # median disparity needed to initialise
    init_min_points: int = 50                                        # triangulated points to accept init
    init_min_angle_deg: float = 2.0                                  # median triangulation angle at init
    search_radius_px: float = 30.0                                   # guided matching radius
    min_pnp_points: int = 12                                         # matches needed for PnP
    min_pnp_inliers: int = 15                                        # inliers needed to accept a pose
    min_tri_angle_deg: float = 2.0                                   # parallax for new map points
    depth_ratio_max: float = 4.0                                     # new points deeper than this x median depth rejected
    depth_ratio_min: float = 0.1                                     # ... or nearer than this x median depth
    kf_min_gap: int = 3                                              # frames between keyframes at least
    kf_baseline_ratio: float = 0.08                                  # baseline / median depth to create a keyframe
    kf_inlier_ratio: float = 0.7                                     # new keyframe when inliers fall below this x last KF
    max_bad: int = 3                                                 # outlier count before a map point is removed
    max_unseen: int = 80                                             # frames unseen before a map point is removed
    max_map: int = 4000                                              # cap on map size
    max_lost: int = 10                                               # lost frames before re-initialising


VoConfig = OrbVoConfig                                               # name used by run_vo


class MonoVO:                                                        # streaming front end
    def __init__(self, camera, size, cfg: OrbVoConfig | None = None):   # camera model + image size
        self.cam = camera                                            # camera model (datasets.py)
        self.cfg = cfg or OrbVoConfig()                              # configuration
        self.fx = float(camera.K[0, 0])                              # focal length [px]
        self.thr = self.cfg.ransac_thr_px / self.fx                  # threshold in normalized units
        self.size = size                                             # (w, h)
        self.mask = np.zeros((size[1], size[0]), np.uint8)           # central mask image
        cv2.circle(self.mask, (int(camera.K[0, 2]), int(camera.K[1, 2])), int(self.cfg.mask_radius_px), 255, -1)   # disc
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))   # contrast equaliser
        if self.cfg.feature == "sift":                               # switchable detector
            self.det = cv2.SIFT_create(nfeatures=self.cfg.n_features, contrastThreshold=0.02)   # SIFT (float descriptors)
            self.norm, self.max_dist = cv2.NORM_L2, self.cfg.max_l2  # L2 matching
        else:                                                        # ORB (binary descriptors)
            self.det = cv2.ORB_create(nfeatures=self.cfg.n_features, scaleFactor=1.2, nlevels=8,
                                      edgeThreshold=15, fastThreshold=10)   # low FAST threshold: dark images
            self.norm, self.max_dist = cv2.NORM_HAMMING, self.cfg.max_hamming   # Hamming matching
        self.bf = cv2.BFMatcher(self.norm)                           # brute-force matcher
        self.frame_idx = -1                                          # last processed frame
        self.map_id = 0                                              # increments on re-initialisation
        self._reset()                                                # clear map and state

    # ------------------------------------------------------------------ state
    def _reset(self):                                                # forget map and state
        self.state = "init"                                          # waiting for initialisation
        self.ref = None                                              # reference frame for bootstrapping
        self.X = np.zeros((0, 3))                                    # map points (VO world)
        self.D = None                                                # map descriptors
        self.last_seen = np.zeros(0, np.int64)                       # frame of last observation
        self.bad = np.zeros(0, np.int32)                             # outlier counter
        self.kf = None                                               # last keyframe record
        self.n_lost = 0                                              # consecutive lost frames
        self.T_cw = None                                             # last valid world -> camera pose
        self.T_cw2 = None                                            # the one before (velocity model)

    # ------------------------------------------------------------------ features
    def _detect(self, gray: np.ndarray):                             # keypoints + descriptors
        g = self.clahe.apply(gray) if self.cfg.use_clahe else gray   # optional enhancement
        kps, desc = self.det.detectAndCompute(g, self.mask)          # masked detection
        if desc is None or len(kps) == 0:                            # nothing found
            return np.zeros((0, 2), np.float32), None, np.zeros((0, 2))   # empty
        px = np.array([k.pt for k in kps], np.float32)               # pixel coordinates
        return px, desc, self.cam.undistort_to_normalized(px)        # + normalized coordinates

    def _dist(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:     # (n,d) vs (m,d) -> (n,m) distances
        if self.norm == cv2.NORM_HAMMING:                            # binary descriptors
            return _POP[a[:, None, :] ^ b[None, :, :]].sum(-1).astype(np.float32)   # popcount of XOR
        d = a[:, None, :].astype(np.float32) - b[None, :, :].astype(np.float32)   # float descriptors
        return np.sqrt((d * d).sum(-1))                              # L2

    def _bf_match(self, d1: np.ndarray, d2: np.ndarray):             # ratio-test matching, returns index pairs
        if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:   # not enough
            return np.zeros((0, 2), int)                             # none
        out = []                                                     # accepted pairs
        for m in self.bf.knnMatch(d1, d2, k=2):                      # two nearest neighbours
            if len(m) == 2 and m[0].distance < self.cfg.ratio * m[1].distance and m[0].distance < self.max_dist:
                out.append((m[0].queryIdx, m[0].trainIdx))           # keep
        return np.array(out, int).reshape(-1, 2)                     # (K,2)

    def _guided(self, px_proj: np.ndarray, idx_map: np.ndarray, kp_px, kp_desc, radius: float):   # match by projection
        tree = cKDTree(kp_px)                                        # keypoint search tree
        neigh = tree.query_ball_point(px_proj, radius)               # candidates per map point
        best = {}                                                    # keypoint -> (dist, map idx)
        for mi, cand in zip(idx_map, neigh):                         # each projected map point
            if not cand:                                             # no keypoint nearby
                continue                                             # skip
            dd = self._dist(self.D[mi:mi + 1], kp_desc[cand])[0]     # distances to the candidates
            o = np.argsort(dd)                                       # nearest first
            if dd[o[0]] > self.max_dist:                             # too different
                continue                                             # skip
            if len(o) > 1 and dd[o[0]] > self.cfg.ratio * dd[o[1]]:  # ambiguous
                continue                                             # skip
            k = cand[o[0]]                                           # chosen keypoint
            if k not in best or dd[o[0]] < best[k][0]:               # one map point per keypoint
                best[k] = (dd[o[0]], mi)                             # keep the closer one
        return np.array([(mi, k) for k, (_, mi) in best.items()], int).reshape(-1, 2)   # (K,2) [map, kp]

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def _triangulate(n1, n2, T1, T2):                                # linear triangulation
        Xh = cv2.triangulatePoints(T1[:3], T2[:3], n1.T, n2.T)       # homogeneous (4,N)
        return (Xh[:3] / Xh[3]).T                                    # (N,3)

    def _check(self, X, n, T_cw):                                    # depth and reprojection error in pixels
        Xc = (T_cw[:3, :3] @ X.T + T_cw[:3, 3:4]).T                  # points in the camera frame
        z = Xc[:, 2]                                                 # depth
        pr = Xc[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]   # normalized projection
        return z, np.linalg.norm(pr - n, axis=1) * self.fx           # depth, pixel error

    def _tri_angle(self, X, C1, C2):                                 # triangulation angle [deg]
        a, b = X - C1, X - C2                                        # rays from both camera centres
        c = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)   # cosine
        return np.degrees(np.arccos(np.clip(c, -1, 1)))              # degrees

    def _add_points(self, X, D, idx):                                # append map points
        self.X = np.vstack([self.X, X])                              # coordinates
        self.D = D if self.D is None or len(self.D) == 0 else np.vstack([self.D, D])   # descriptors
        self.last_seen = np.concatenate([self.last_seen, np.full(len(X), idx, np.int64)])   # freshness
        self.bad = np.concatenate([self.bad, np.zeros(len(X), np.int32)])   # outlier counters

    def _drop_points(self, keep: np.ndarray):                        # remove map points by mask
        self.X, self.D = self.X[keep], self.D[keep]                  # coordinates and descriptors
        self.last_seen, self.bad = self.last_seen[keep], self.bad[keep]   # bookkeeping

    # ------------------------------------------------------------------ initialisation
    def _try_init(self, ref, cur, idx) -> bool:                      # essential-matrix bootstrap
        m = self._bf_match(ref["desc"], cur["desc"])                 # reference <-> current matches
        if len(m) < self.cfg.init_min_matches:                       # too few
            return False                                             # fail
        pr, pc = ref["px"][m[:, 0]], cur["px"][m[:, 1]]              # pixels
        if np.median(np.linalg.norm(pc - pr, axis=1)) < self.cfg.init_parallax_px:   # not enough motion yet
            return False                                             # wait
        nr, nc = ref["n"][m[:, 0]], cur["n"][m[:, 1]]                # normalized coordinates
        E, mk = cv2.findEssentialMat(nr, nc, np.eye(3), cv2.RANSAC, 0.999, self.thr)   # RANSAC essential matrix
        if E is None or E.shape[0] < 3:                              # degenerate
            return False                                             # fail
        E = E[:3]                                                    # first solution
        n_in, R, t, mp = cv2.recoverPose(E, nr, nc, np.eye(3), mask=mk)   # cheirality
        if n_in < self.cfg.init_min_points:                          # too few inliers
            return False                                             # fail
        T2 = _T(R, t)                                                # |t| = 1 VO unit
        sel = mp.ravel() > 0                                         # inliers after cheirality
        X = self._triangulate(nr[sel], nc[sel], np.eye(4), T2)       # points in the reference frame
        z1, e1 = self._check(X, nr[sel], np.eye(4))                  # reference view
        z2, e2 = self._check(X, nc[sel], T2)                         # current view
        ang = self._tri_angle(X, np.zeros(3), _inv(T2)[:3, 3])       # triangulation angles
        lim = 3 * self.cfg.ransac_thr_px                             # pixel limit
        good = (z1 > 0) & (z2 > 0) & (e1 < lim) & (e2 < lim)         # cheirality + reprojection
        if good.sum() < self.cfg.init_min_points or np.median(ang[good]) < self.cfg.init_min_angle_deg:
            return False                                             # not enough points or parallax
        kp_idx = m[sel][good, 1]                                     # keypoint indices in the current frame
        self._add_points(X[good], cur["desc"][kp_idx], idx)          # initial map
        used = np.zeros(len(cur["px"]), bool)                        # keypoints that own a map point
        used[kp_idx] = True                                          # mark
        self.T_cw2, self.T_cw = np.eye(4), T2                        # velocity model start
        self.kf = dict(T_cw=T2.copy(), px=cur["px"], n=cur["n"], desc=cur["desc"], used=used, idx=idx,
                       n_inl=int(good.sum()))                        # keyframe record
        return True                                                  # initialised

    # ------------------------------------------------------------------ tracking
    def _predict(self):                                              # constant-velocity guess
        if self.T_cw2 is None:                                       # no velocity yet
            return self.T_cw                                         # last pose
        return (self.T_cw @ _inv(self.T_cw2)) @ self.T_cw            # D * T_prev

    def _project_map(self, T_cw):                                    # visible map points under a pose
        Xc = (T_cw[:3, :3] @ self.X.T + T_cw[:3, 3:4]).T             # camera frame
        ok = Xc[:, 2] > 0.05                                         # in front
        idx = np.where(ok)[0]                                        # candidates
        if len(idx) == 0:                                            # nothing visible
            return idx, np.zeros((0, 2))                             # empty
        px = self.cam.project(Xc[idx])                               # pixels
        c = np.array([self.cam.K[0, 2], self.cam.K[1, 2]])           # principal point
        inside = np.linalg.norm(px - c, axis=1) < self.cfg.mask_radius_px + 15   # inside the usable disc
        return idx[inside], px[inside]                               # visible ids and pixels

    def _pnp(self, pairs, cur, guess):                               # pose from 2D-3D matches
        if len(pairs) < self.cfg.min_pnp_points:                     # too few
            return None                                              # fail
        Xs = np.ascontiguousarray(self.X[pairs[:, 0]], np.float64)   # 3D points
        xs = np.ascontiguousarray(cur["n"][pairs[:, 1]], np.float64)   # normalized observations
        if guess is not None:                                        # use the prediction
            rv, tv, use = cv2.Rodrigues(guess[:3, :3])[0], guess[:3, 3:4].copy(), True   # initial pose
        else:                                                        # no guess (relocalisation)
            rv, tv, use = np.zeros((3, 1)), np.zeros((3, 1)), False  # blank
        flags = cv2.SOLVEPNP_ITERATIVE if use else cv2.SOLVEPNP_EPNP   # solver
        ok, rv, tv, inl = cv2.solvePnPRansac(Xs, xs, np.eye(3), np.zeros((4, 1)), rv, tv, use, 200, self.thr, 0.995,
                                             flags=flags)           # robust PnP
        if not ok or inl is None or len(inl) < self.cfg.min_pnp_inliers:   # failure
            return None                                              # fail
        inl = inl.ravel()                                            # inlier indices
        rv, tv = cv2.solvePnPRefineLM(Xs[inl], xs[inl], np.eye(3), np.zeros((4, 1)), rv, tv)   # refine
        mask = np.zeros(len(pairs), bool)                            # inlier mask
        mask[inl] = True                                             # mark
        return _T(cv2.Rodrigues(rv)[0], tv), mask                    # pose, inlier mask

    def _new_keyframe(self, cur, T_cw, matched_kp, idx) -> int:      # triangulate new points against the last KF
        kf = self.kf                                                 # previous keyframe
        used_cur = np.zeros(len(cur["px"]), bool)                    # current keypoints already in the map
        used_cur[matched_kp] = True                                  # inlier matches
        a = np.where(~kf["used"])[0]                                 # free keypoints in the old KF
        b = np.where(~used_cur)[0]                                   # free keypoints in the current frame
        added = 0                                                    # counter
        if len(a) >= 8 and len(b) >= 8:                              # something to match
            m = self._bf_match(kf["desc"][a], cur["desc"][b])        # descriptor matches
            if len(m):                                               # any
                ia, ib = a[m[:, 0]], b[m[:, 1]]                      # keypoint indices
                T_rel = T_cw @ _inv(kf["T_cw"])                      # camera 1 -> camera 2
                tx = np.array([[0, -T_rel[2, 3], T_rel[1, 3]], [T_rel[2, 3], 0, -T_rel[0, 3]],
                               [-T_rel[1, 3], T_rel[0, 3], 0]])      # [t]x
                Em = tx @ T_rel[:3, :3]                              # essential matrix from the known motion
                h1 = np.hstack([kf["n"][ia], np.ones((len(ia), 1))]) # homogeneous points, view 1
                h2 = np.hstack([cur["n"][ib], np.ones((len(ib), 1))])   # view 2
                l2 = (Em @ h1.T).T                                   # epipolar lines in view 2
                epi = np.abs(np.sum(h2 * l2, 1)) / (np.linalg.norm(l2[:, :2], axis=1) + 1e-12)   # normalized distance
                keep = epi < 3 * self.thr                            # epipolar consistency
                ia, ib = ia[keep], ib[keep]                          # filter
                if len(ia):                                          # triangulate
                    Xw = self._triangulate(kf["n"][ia], cur["n"][ib], kf["T_cw"], T_cw)   # world points
                    z1, e1 = self._check(Xw, kf["n"][ia], kf["T_cw"])   # view 1
                    z2, e2 = self._check(Xw, cur["n"][ib], T_cw)     # view 2
                    ang = self._tri_angle(Xw, _inv(kf["T_cw"])[:3, 3], _inv(T_cw)[:3, 3])   # angles
                    vis = self.X[self.last_seen >= idx - 5]          # recently seen map points
                    zref = np.median((T_cw[:3, :3] @ vis.T + T_cw[:3, 3:4])[2]) if len(vis) >= 10 else None   # depth reference
                    ok = (z1 > 0) & (z2 > 0) & (e1 < 2 * self.cfg.ransac_thr_px) & (e2 < 2 * self.cfg.ransac_thr_px) \
                         & (ang > self.cfg.min_tri_angle_deg)        # basic checks
                    if zref is not None:                             # depth sanity relative to the map
                        ok &= (z2 < self.cfg.depth_ratio_max * zref) & (z2 > self.cfg.depth_ratio_min * zref)
                    if ok.any():                                     # add
                        self._add_points(Xw[ok], cur["desc"][ib[ok]], idx)   # new map points
                        used_cur[ib[ok]] = True                      # now own a map point
                        added = int(ok.sum())                        # count
        self.kf = dict(T_cw=T_cw.copy(), px=cur["px"], n=cur["n"], desc=cur["desc"], used=used_cur, idx=idx,
                       n_inl=int(used_cur.sum()))                    # becomes the keyframe
        return added                                                 # number of new points

    def _prune(self, idx):                                           # remove bad and stale map points
        keep = (self.bad < self.cfg.max_bad) & (idx - self.last_seen < self.cfg.max_unseen)   # health
        if len(self.X) > self.cfg.max_map:                           # cap size by freshness
            order = np.argsort(-self.last_seen)                      # newest first
            cap = np.zeros(len(self.X), bool)                        # selection
            cap[order[:self.cfg.max_map]] = True                     # keep the newest
            keep &= cap                                              # combine
        if not keep.all():                                           # something to remove
            self._drop_points(keep)                                  # remove

    # ------------------------------------------------------------------ main entry
    def process(self, gray_u8: np.ndarray):                          # one frame -> (T_w_c|None, status, info)
        self.frame_idx += 1                                          # advance
        idx = self.frame_idx                                         # shorthand
        px, desc, n = self._detect(gray_u8)                          # features
        cur = dict(px=px, desc=desc, n=n)                            # current frame record
        info = {"n_kp": len(px), "n_match": 0, "n_inl": 0, "n_map": len(self.X), "n_new": 0, "map_id": self.map_id}   # diagnostics
        if self.state == "init":                                     # bootstrapping
            if self.ref is None or desc is None:                     # first frame of an attempt
                self.ref = cur if desc is not None else None         # remember it
                return None, UNINIT, info                            # nothing yet
            if self._try_init(self.ref, cur, idx):                   # enough parallax and good geometry
                self.state, self.ref, self.n_lost = "track", None, 0   # tracking from now on
                info.update(n_map=len(self.X), n_inl=self.kf["n_inl"], n_match=self.kf["n_inl"])   # diagnostics
                return _inv(self.T_cw), OK, info                     # first pose
            if len(self._bf_match(self.ref["desc"], desc)) < self.cfg.init_min_matches // 2:   # reference went stale
                self.ref = cur                                       # restart from here
            return None, UNINIT, info                                # keep waiting
        guess = self._predict()                                      # pose prediction
        vis, proj = self._project_map(guess)                         # visible map points
        pairs = np.zeros((0, 2), int)                                # [map idx, kp idx]
        if desc is not None and len(vis):                            # guided matching
            for rad in (self.cfg.search_radius_px, 2.5 * self.cfg.search_radius_px):   # widen once if needed
                pairs = self._guided(proj, vis, px, desc, rad)       # matches
                if len(pairs) >= self.cfg.min_pnp_points:            # enough
                    break                                            # stop widening
        res = self._pnp(pairs, cur, guess) if len(pairs) else None   # pose from guided matches
        if res is None and desc is not None and len(self.X) >= self.cfg.min_pnp_points:   # relocalise globally
            ids = np.arange(len(self.X))                             # all map points
            m = self._bf_match(self.D, desc)                         # unguided matching
            if len(m):                                               # any
                pairs = np.stack([ids[m[:, 0]], m[:, 1]], 1)         # [map, kp]
                res = self._pnp(pairs, cur, None)                    # PnP without guess
        info["n_match"] = len(pairs)                                 # matches used
        if res is None:                                              # tracking lost
            self.n_lost += 1                                         # count
            self.T_cw2, self.T_cw = None, guess                      # coast on the prediction
            if self.n_lost > self.cfg.max_lost:                      # give up: new map
                self.map_id += 1                                     # new segment
                self._reset()                                        # clear
                self.ref = cur if desc is not None else None         # new bootstrap reference
                info.update(map_id=self.map_id, n_map=0)             # diagnostics
                return None, UNINIT, info                            # wait for initialisation
            info["n_map"] = len(self.X)                              # map size
            return _inv(guess), LOST, info                           # predicted pose
        T_cw, inl = res                                              # unpack
        self.n_lost = 0                                              # reset
        self.T_cw2, self.T_cw = self.T_cw, T_cw                      # velocity model
        mi, ki = pairs[inl, 0], pairs[inl, 1]                        # inlier map ids and keypoints
        self.last_seen[mi] = idx                                     # seen now
        if self.cfg.refresh_desc:                                    # follow gradual appearance change
            self.D[mi] = cur["desc"][ki]                             # newest descriptor
        mo = pairs[~inl, 0]                                          # outlier matches
        self.bad[mo] += 1                                            # strike
        zc, err = self._check(self.X[mi], cur["n"][ki], T_cw)        # reprojection of inliers
        self.bad[mi[(err > 2 * self.cfg.ransac_thr_px) | (zc <= 0)]] += 1   # cheirality / reprojection strikes
        info["n_inl"] = int(inl.sum())                               # inliers
        zmed = float(np.median(zc[zc > 0])) if (zc > 0).any() else 1.0   # scene depth in this frame
        C_cur, C_kf = _inv(T_cw)[:3, 3], _inv(self.kf["T_cw"])[:3, 3]   # camera centres
        base = float(np.linalg.norm(C_cur - C_kf))                   # baseline to the last keyframe
        if idx - self.kf["idx"] >= self.cfg.kf_min_gap and (
                base > self.cfg.kf_baseline_ratio * zmed or info["n_inl"] < self.cfg.kf_inlier_ratio * self.kf["n_inl"]):
            info["n_new"] = self._new_keyframe(cur, T_cw, ki, idx)   # keyframe + new points
        self._prune(idx)                                             # remove bad / stale points
        info["n_map"] = len(self.X)                                  # map size after pruning
        return _inv(T_cw), OK, info                                  # camera pose in the map frame
