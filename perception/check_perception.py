"""Step 6 checkpoint: detector on real room1 frames, tracker on a synthetic multi-target scenario (MOT metrics)."""
from __future__ import annotations                                   # forward type hints
import json                                                          # metrics file
import time                                                          # timing
from collections import Counter                                      # class histogram
import numpy as np                                                   # arrays
import cv2                                                           # image output
from scipy.optimize import linear_sum_assignment                     # matching

from vo.datasets import Sequence, PROJECT_ROOT                       # data access
from perception.detector import YoloxDetector, COCO                  # detector
from perception.tracker import SortTracker, TrackerConfig, iou_matrix, Track   # tracker


def synth_scene(n_frames=600, n_targets=5, seed=0, size=512, miss=0.10, fp_rate=0.5, sigma=2.0):   # synthetic MOT data
    rng = np.random.default_rng(seed)                                # reproducible
    gt = {}                                                          # frame -> list of (id, box)
    dets = {}                                                        # frame -> detections (N,6)
    base = []                                                        # per-target motion parameters
    for i in range(n_targets):                                       # random smooth trajectories
        base.append(dict(c=rng.uniform(100, size - 100, 2), a=rng.uniform(60, 180, 2), w=rng.uniform(0.01, 0.04, 2),
                         ph=rng.uniform(0, 6.28, 2), wh=rng.uniform(30, 70, 2), life=(rng.integers(0, 80), n_frames - rng.integers(0, 80))))
    occl = {i: (rng.integers(100, 500), 0) for i in range(n_targets)}   # one occlusion window per target
    occl = {i: (s, s + int(rng.integers(3, 9))) for i, (s, _) in occl.items()}   # (start, end)
    for f in range(n_frames):                                        # simulate
        g, d = [], []                                                # this frame
        for i, b in enumerate(base):                                 # each target
            if not (b["life"][0] <= f < b["life"][1]):               # not alive
                continue                                             # skip
            cx = b["c"][0] + b["a"][0] * np.sin(b["w"][0] * f + b["ph"][0])   # x position
            cy = b["c"][1] + b["a"][1] * np.sin(b["w"][1] * f + b["ph"][1])   # y position
            w, h = b["wh"][0], b["wh"][1]                            # size
            box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])   # xyxy
            g.append((i, box))                                       # ground truth
            seen = rng.random() > miss and not (occl[i][0] <= f < occl[i][1])   # detector hit?
            if seen:                                                 # noisy detection
                nb = box + rng.normal(0, sigma, 4) + rng.normal(0, 0.03 * w, 4) * np.array([-1, -1, 1, 1])   # jitter
                d.append([*nb, rng.uniform(0.4, 0.95), 0])           # box, score, class
        for _ in range(rng.poisson(fp_rate)):                        # false positives
            c = rng.uniform(40, size - 40, 2)                        # random place
            s = rng.uniform(20, 60, 2)                               # random size
            d.append([c[0] - s[0] / 2, c[1] - s[1] / 2, c[0] + s[0] / 2, c[1] + s[1] / 2, rng.uniform(0.3, 0.6), 0])   # FP
        gt[f], dets[f] = g, np.array(d, dtype=np.float64).reshape(-1, 6)   # store
    return gt, dets                                                  # scene


def clear_mot(gt: dict, trk: dict, iou_thr: float = 0.5) -> dict:    # simplified CLEAR-MOT
    fn = fp = idsw = ngt = matches = 0                               # counters
    last = {}                                                        # gt id -> last matched track id
    prev = {}                                                        # gt id -> track id in previous frame
    tracked_frames, life = Counter(), Counter()                      # for mostly-tracked ratio
    for f in sorted(gt):                                             # frame loop
        g = gt[f]                                                    # GT list
        t = trk.get(f, np.zeros((0, 6)))                             # reported tracks
        ngt += len(g)                                                # count GT
        gids = [i for i, _ in g]                                     # GT ids
        gb = np.array([b for _, b in g]).reshape(-1, 4)              # GT boxes
        tb = t[:, 1:5] if len(t) else np.zeros((0, 4))               # track boxes
        M = iou_matrix(gb, tb)                                       # IoU
        pairs = {}                                                   # gt idx -> track idx
        used_t = set()                                               # tracks already used
        for gi, gid in enumerate(gids):                              # keep previous correspondence if still valid
            if gid in prev:                                          # had a match last frame
                ti = np.where(t[:, 0] == prev[gid])[0] if len(t) else []   # same track id still present?
                if len(ti) and M[gi, ti[0]] >= iou_thr:              # still overlapping
                    pairs[gi] = int(ti[0])                           # keep
                    used_t.add(int(ti[0]))                           # mark
        rest_g = [i for i in range(len(gids)) if i not in pairs]     # unmatched GT
        rest_t = [j for j in range(len(tb)) if j not in used_t]      # unmatched tracks
        if rest_g and rest_t:                                        # Hungarian on the rest
            sub = 1 - M[np.ix_(rest_g, rest_t)]                      # cost
            ri, ci = linear_sum_assignment(sub)                      # assign
            for r, c in zip(ri, ci):                                 # accept if IoU ok
                if 1 - sub[r, c] >= iou_thr:                         # threshold
                    pairs[rest_g[r]] = rest_t[c]                     # pair
        matches += len(pairs)                                        # TP count
        fn += len(gids) - len(pairs)                                 # missed
        fp += len(tb) - len(pairs)                                   # spurious
        prev = {}                                                    # rebuild
        for gi, tj in pairs.items():                                 # per match
            gid, tid = gids[gi], int(t[tj, 0])                       # ids
            if gid in last and last[gid] != tid:                     # identity changed
                idsw += 1                                            # switch
            last[gid] = tid                                          # remember
            prev[gid] = tid                                          # remember
            tracked_frames[gid] += 1                                 # tracked this frame
        for gid in gids:                                             # lifetime counter
            life[gid] += 1                                           # alive frames
    mt = np.mean([tracked_frames[g] / life[g] >= 0.8 for g in life]) if life else 0.0   # mostly tracked
    return {"MOTA": 1 - (fn + fp + idsw) / max(ngt, 1), "FN": fn, "FP": fp, "IDSW": idsw, "GT": ngt,
            "recall": matches / max(ngt, 1), "precision": matches / max(matches + fp, 1), "mostly_tracked": float(mt)}   # metrics


def main(name: str = "tumvi_room1"):                                 # run the checkpoint
    res = PROJECT_ROOT / "results"                                   # output folder
    out = {}                                                         # metrics
    # --- synthetic tracking benchmark ---
    for label, kw in (("clean", dict(miss=0.05, fp_rate=0.2, sigma=1.5)), ("hard", dict(miss=0.25, fp_rate=1.0, sigma=4.0))):   # two difficulties
        gt, dets = synth_scene(**kw)                                 # scene
        Track._next_id = 1                                           # reset ids
        tr = SortTracker(TrackerConfig())                            # tracker
        rep = {f: tr.update(dets[f]) for f in sorted(dets)}          # run
        out[f"synthetic_{label}_tracker"] = clear_mot(gt, rep)       # metrics with tracking
        raw = {f: np.hstack([np.arange(len(dets[f]))[:, None] + 1 + 10000 * f, dets[f][:, :4], dets[f][:, 5:6]]) for f in dets}   # fresh id every frame
        out[f"synthetic_{label}_raw_detections"] = clear_mot(gt, raw)   # baseline without tracking
    # --- real frames ---
    seq = Sequence(name)                                             # data
    det = YoloxDetector(PROJECT_ROOT / "data" / "models" / "yolox_nano.onnx", conf=0.3)   # detector
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))      # contrast boost
    stats = {}                                                       # per-variant stats
    best = (0.0, None, None)                                         # best detection for the figure
    for variant in ("raw", "clahe"):                                 # two preprocessing variants
        cnt, n_with, t0, nf = Counter(), 0, time.time(), 0           # counters
        for i in range(0, len(seq), 5):                              # every 5th frame (about 4 Hz)
            img = seq.frame_u8(i)                                    # 8-bit frame
            if variant == "clahe":                                   # optional enhancement
                img = clahe.apply(img)                               # CLAHE
            d = det.detect(img)                                      # detections
            nf += 1                                                  # frames processed
            n_with += int(len(d) > 0)                                # frames with any detection
            for row in d:                                            # per detection
                cnt[COCO[int(row[5])]] += 1                          # class histogram
                if row[4] > best[0]:                                 # keep the most confident one
                    best = (float(row[4]), i, d.copy())              # store
        stats[variant] = {"frames": nf, "frames_with_detection": n_with, "ms_per_frame": 1000 * (time.time() - t0) / nf,
                          "top_classes": cnt.most_common(5)}         # summary
    out["real_room1_detector"] = stats                               # store
    if best[1] is not None:                                          # save an example
        img = cv2.cvtColor(seq.frame_u8(best[1]), cv2.COLOR_GRAY2BGR)   # colour copy
        for x1, y1, x2, y2, s, c in best[2]:                         # draw
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)   # box
            cv2.putText(img, f"{COCO[int(c)]} {s:.2f}", (int(x1), max(int(y1) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)   # label
        cv2.imwrite(str(res / "detections_room1.png"), img)          # save
    with open(res / "metrics_step6.json", "w", encoding="utf-8") as f:   # save metrics
        json.dump(out, f, indent=2)                                  # write
    for k, v in out.items():                                         # print
        print(k, json.dumps(v, default=float))                       # line


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
