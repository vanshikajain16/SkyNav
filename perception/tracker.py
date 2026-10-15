"""SORT-style multi-object tracker: constant-velocity Kalman filter + IoU Hungarian association."""
from __future__ import annotations                                   # forward type hints
from dataclasses import dataclass                                    # config container
import numpy as np                                                   # arrays
from scipy.optimize import linear_sum_assignment                     # Hungarian algorithm


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:          # IoU between two xyxy box sets
    if len(a) == 0 or len(b) == 0:                                   # nothing to compare
        return np.zeros((len(a), len(b)))                            # empty matrix
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])                    # intersection left
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])                    # intersection top
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])                    # intersection right
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])                    # intersection bottom
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)    # intersection area
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])                   # area of a
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])                   # area of b
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)        # IoU


def xyxy_to_z(b: np.ndarray) -> np.ndarray:                          # box -> [cx, cy, w, h]
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], dtype=np.float64)   # vector


def x_to_xyxy(x: np.ndarray) -> np.ndarray:                          # state -> box
    return np.array([x[0] - x[2] / 2, x[1] - x[3] / 2, x[0] + x[2] / 2, x[1] + x[3] / 2])   # xyxy


@dataclass                                                           # tracker tunables
class TrackerConfig:
    iou_min: float = 0.25                                            # association threshold
    max_age: int = 10                                                # frames a track may coast
    min_hits: int = 3                                                # hits before a track is reported
    q_pos: float = 1.0                                               # process noise on position [px]
    q_vel: float = 0.5                                               # process noise on velocity [px/frame]
    r_meas: float = 3.0                                              # measurement noise [px]


class Track:                                                         # one tracked object
    _next_id = 1                                                     # global id counter

    def __init__(self, box: np.ndarray, cfg: TrackerConfig, cls: int = -1):   # start from a detection
        self.cfg = cfg                                               # config
        self.id = Track._next_id                                     # unique id
        Track._next_id += 1                                          # advance
        self.x = np.zeros(8)                                         # [cx, cy, w, h, vx, vy, vw, vh]
        self.x[:4] = xyxy_to_z(box)                                  # initial box
        self.P = np.diag([10.0] * 4 + [100.0] * 4)                   # initial uncertainty (velocity unknown)
        self.hits, self.age, self.since_update, self.cls = 1, 0, 0, cls   # bookkeeping

    def predict(self):                                               # constant-velocity prediction
        F = np.eye(8)                                                # transition
        F[:4, 4:] = np.eye(4)                                        # position += velocity
        Q = np.diag([self.cfg.q_pos ** 2] * 4 + [self.cfg.q_vel ** 2] * 4)   # process noise
        self.x = F @ self.x                                          # state
        self.P = F @ self.P @ F.T + Q                                # covariance
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)                   # keep sizes positive
        self.age += 1                                                # frames alive
        self.since_update += 1                                       # frames since a detection

    def update(self, box: np.ndarray, cls: int = -1):                # fuse a detection
        H = np.zeros((4, 8))                                         # measurement matrix
        H[:, :4] = np.eye(4)                                         # observe box only
        R = np.eye(4) * self.cfg.r_meas ** 2                         # measurement noise
        S = H @ self.P @ H.T + R                                     # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)                          # gain
        self.x = self.x + K @ (xyxy_to_z(box) - H @ self.x)          # state update
        self.P = (np.eye(8) - K @ H) @ self.P                        # covariance update
        self.hits += 1                                               # hit count
        self.since_update = 0                                        # reset
        self.cls = cls                                               # latest class

    def box(self) -> np.ndarray:                                     # current box estimate
        return x_to_xyxy(self.x)                                     # xyxy


class SortTracker:                                                   # the tracker
    def __init__(self, cfg: TrackerConfig | None = None):           # configuration
        self.cfg = cfg or TrackerConfig()                            # config
        self.tracks: list[Track] = []                                # live tracks
        self.frame = 0                                               # frame counter

    def compensate(self, warp):                                      # shift track centres by an ego-motion warp
        for t in self.tracks:                                        # every live track
            c = warp(t.x[:2])                                        # where the old centre lands in the new image
            if c is not None and np.all(np.isfinite(c)):             # valid warp
                t.x[:2] = c                                          # move the centre (size and velocity kept)

    def update(self, dets: np.ndarray):                              # dets: (N, >=4) xyxy [+score, cls]
        self.frame += 1                                              # advance
        for t in self.tracks:                                        # predict all tracks
            t.predict()                                              # coast
        trk = np.array([t.box() for t in self.tracks]) if self.tracks else np.zeros((0, 4))   # predicted boxes
        d = dets[:, :4] if len(dets) else np.zeros((0, 4))           # detection boxes
        matched, un_d = set(), list(range(len(d)))                   # bookkeeping
        if len(trk) and len(d):                                      # something to associate
            cost = 1.0 - iou_matrix(trk, d)                          # cost = 1 - IoU
            ri, ci = linear_sum_assignment(cost)                     # optimal assignment
            for r, c in zip(ri, ci):                                 # accept good matches
                if 1.0 - cost[r, c] >= self.cfg.iou_min:             # IoU above threshold
                    cls = int(dets[c, 5]) if dets.shape[1] > 5 else -1   # class if present
                    self.tracks[r].update(d[c], cls)                 # fuse
                    matched.add(c)                                   # mark detection used
            un_d = [i for i in range(len(d)) if i not in matched]    # unmatched detections
        for i in un_d:                                               # births
            cls = int(dets[i, 5]) if dets.shape[1] > 5 else -1       # class if present
            self.tracks.append(Track(d[i], self.cfg, cls))           # new track
        self.tracks = [t for t in self.tracks if t.since_update <= self.cfg.max_age]   # deaths
        out = []                                                     # reported tracks
        for t in self.tracks:                                        # confirmed and recently seen
            if t.since_update == 0 and (t.hits >= self.cfg.min_hits or self.frame <= self.cfg.min_hits):
                out.append((t.id, *t.box(), t.cls))                  # (id, x1, y1, x2, y2, cls)
        return np.array(out, dtype=np.float64).reshape(-1, 6)        # (M, 6)
