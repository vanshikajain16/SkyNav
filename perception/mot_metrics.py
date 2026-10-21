"""MOT metrics with the motmetrics library: MOTA, IDF1 (and ID switches, recall, precision) with ignore-region handling.

Boxes are xyxy. Per frame: valid GT (evaluated categories), hypotheses (tracker output), and ignore boxes
(GT of non-evaluated categories and ignored regions). Hypotheses that are not matched to a valid GT but overlap an
ignore box (IoA > 0.5 of the hypothesis area) are dropped before scoring, as in common MOT benchmark practice.
"""
from __future__ import annotations                                   # forward type hints
import numpy as np                                                   # arrays
import motmetrics as mm                                              # MOTA / IDF1
from scipy.optimize import linear_sum_assignment                     # matching

from perception.tracker import iou_matrix                            # IoU of xyxy boxes


def ioa(a: np.ndarray, b: np.ndarray) -> np.ndarray:                 # intersection over area of a (rows of a vs rows of b)
    if len(a) == 0 or len(b) == 0:                                   # nothing to compare
        return np.zeros((len(a), len(b)))                            # empty
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])   # intersection corner 1
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])   # intersection corner 2
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)    # intersection area
    area = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]      # area of a
    return inter / np.maximum(area, 1e-9)                            # IoA


def score_sequence(gt: dict, hyp: dict, ignore: dict, iou_thr: float = 0.5, n_frames: int | None = None):
    """gt: frame -> (ids (n,), boxes (n,4)); hyp: frame -> (ids, boxes); ignore: frame -> boxes (m,4). Returns an accumulator."""
    acc = mm.MOTAccumulator(auto_id=False)                           # frame ids supplied explicitly
    frames = range(1, (n_frames or max(list(gt) + list(hyp) + [0])) + 1)   # all frames, 1-based
    for f in frames:                                                 # frame loop
        g_ids, g_box = gt.get(f, (np.zeros(0, int), np.zeros((0, 4))))   # valid GT
        h_ids, h_box = hyp.get(f, (np.zeros(0, int), np.zeros((0, 4))))   # hypotheses
        ig = ignore.get(f, np.zeros((0, 4)))                         # ignore boxes
        if len(h_ids) and len(ig):                                   # drop hypotheses inside ignore regions
            M = iou_matrix(g_box, h_box) if len(g_ids) else np.zeros((0, len(h_ids)))   # IoU with valid GT
            matched = np.zeros(len(h_ids), bool)                     # hypotheses matched to a valid GT
            if M.size:                                               # any valid GT
                r, c = linear_sum_assignment(1 - M)                  # assignment
                matched[c[M[r, c] >= iou_thr]] = True                # keep those with enough overlap
            drop = (ioa(h_box, ig).max(axis=1) > 0.5) & ~matched     # unmatched and inside an ignore box
            h_ids, h_box = h_ids[~drop], h_box[~drop]                # remove them
        D = 1.0 - iou_matrix(g_box, h_box) if len(g_ids) and len(h_ids) else np.zeros((len(g_ids), len(h_ids)))   # 1 - IoU
        D[D > 1.0 - iou_thr] = np.nan                                # pairs below the IoU threshold cannot match
        # (motmetrics.distances.iou_matrix is avoided: v1.4.0 calls np.asfarray, removed in NumPy 2)
        acc.update(list(g_ids), list(h_ids), D, frameid=f)           # register the frame
    return acc                                                       # accumulator


def xyxy_to_xywh(b: np.ndarray) -> np.ndarray:                       # motmetrics.iou_matrix expects (x, y, w, h)
    out = b.copy().astype(float)                                     # copy
    if len(out):                                                     # any boxes
        out[:, 2] = b[:, 2] - b[:, 0]                                # width
        out[:, 3] = b[:, 3] - b[:, 1]                                # height
    return out                                                       # xywh


def compute(accs: dict) -> "mm.pd.DataFrame":                        # summary over sequences (+ overall)
    mh = mm.metrics.create()                                         # metric host
    return mh.compute_many(list(accs.values()), metrics=["mota", "idf1", "num_switches", "num_misses", "num_false_positives",
                                                         "num_objects", "recall", "precision", "mostly_tracked", "num_fragmentations"],
                           names=list(accs.keys()), generate_overall=True)   # per sequence + OVERALL
