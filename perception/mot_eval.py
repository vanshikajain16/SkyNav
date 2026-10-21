"""Run detector + SORT tracker on VisDrone-MOT sequences, score MOTA / IDF1 with motmetrics, write a demo video.

usage: python -m perception.mot_eval --root data\\visdrone\\VisDrone2019-MOT-val --model yolox_s --provider dml
Expected layout (to be confirmed against the downloaded data): <root>\\sequences\\<seq>\\*.jpg and <root>\\annotations\\<seq>.txt
GT line: frame, id, left, top, width, height, score, category, truncation, occlusion
Evaluated categories (VisDrone-MOT challenge): pedestrian 1, car 4, van 5, truck 6, bus 9. Everything else in the GT
(including ignored regions and score 0) becomes an ignore region.
"""
from __future__ import annotations                                   # forward type hints
import argparse                                                      # command line
import json                                                          # metrics file
import time                                                          # timing
from pathlib import Path                                             # paths
import numpy as np                                                   # arrays
import cv2                                                           # images and video

from vo.datasets import PROJECT_ROOT                                 # project folder
from perception.detector import YoloxDetector                        # detector
from perception.tracker import SortTracker, TrackerConfig, Track     # tracker
from perception.mot_metrics import score_sequence, compute           # metrics

EVAL_CATS = {1, 4, 5, 6, 9}                                          # pedestrian, car, van, truck, bus
COCO_TO_VISDRONE = {0: 1, 2: 4, 5: 9, 7: 6}                          # COCO person/car/bus/truck -> VisDrone category
CAT_NAME = {1: "pedestrian", 4: "car", 5: "van", 6: "truck", 9: "bus"}   # for drawing


def read_gt(path: Path):                                             # parse one annotation file
    a = np.loadtxt(path, delimiter=",", ndmin=2)                     # (n, 10)
    gt, ign = {}, {}                                                 # per-frame dictionaries
    for f in np.unique(a[:, 0]).astype(int):                         # frame by frame
        r = a[a[:, 0] == f]                                          # rows of this frame
        box = np.stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]], 1)   # xyxy
        valid = np.isin(r[:, 7].astype(int), list(EVAL_CATS)) & (r[:, 6] != 0)   # evaluated and not score-0
        gt[f] = (r[valid, 1].astype(int), box[valid])                # ids and boxes
        ign[f] = box[~valid]                                         # everything else is an ignore region
    return gt, ign                                                   # dictionaries


def draw(img, tracks, frame_no, fps_text):                           # overlay boxes and ids
    rng = np.random.default_rng(0)                                   # fixed palette
    pal = rng.integers(60, 255, size=(512, 3))                       # colours by id
    for tid, x1, y1, x2, y2, cls in tracks:                          # each reported track
        c = tuple(int(v) for v in pal[int(tid) % 512])               # colour
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)   # box
        cv2.putText(img, f"{int(tid)} {CAT_NAME.get(int(cls), '')}", (int(x1), max(int(y1) - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)   # id and class
    cv2.putText(img, f"frame {frame_no}  {fps_text}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)   # header
    return img                                                       # annotated frame


def run_sequence(det, root: Path, seq: str, conf: float, max_frames, video_path: Path | None):
    files = sorted((root / "sequences" / seq).glob("*.jpg"))         # frames
    if max_frames:                                                   # optional limit
        files = files[:max_frames]                                   # first N
    tracker = SortTracker(TrackerConfig())                           # fresh tracker
    hyp, t_det, t_trk = {}, [], []                                   # outputs and timings
    vw = None                                                        # video writer
    for i, fpath in enumerate(files, start=1):                       # frame loop
        img = cv2.imread(str(fpath))                                 # colour frame
        d = det.detect(img)                                          # detections (x1,y1,x2,y2,score,coco_cls)
        t_det.append(det.last_times["total_ms"])                     # detector time
        keep = np.isin(d[:, 5].astype(int), list(COCO_TO_VISDRONE)) if len(d) else np.zeros(0, bool)   # mapped classes only
        d = d[keep]                                                  # filter
        if len(d):                                                   # map class ids
            d[:, 5] = [COCO_TO_VISDRONE[int(c)] for c in d[:, 5]]    # to VisDrone categories
        t0 = time.perf_counter()                                     # tracker timer
        tr = tracker.update(d.astype(np.float64))                    # tracks (id, x1, y1, x2, y2, cls)
        t_trk.append(1e3 * (time.perf_counter() - t0))               # tracker time
        hyp[i] = (tr[:, 0].astype(int), tr[:, 1:5]) if len(tr) else (np.zeros(0, int), np.zeros((0, 4)))   # for metrics
        if video_path is not None:                                   # write the demo video
            if vw is None:                                           # open lazily with the first frame's size
                h, w = img.shape[:2]; sc = min(1.0, 1280 / w)        # downscale wide frames
                vsize = (int(w * sc), int(h * sc))                   # writer frame size (kept, the writer cannot report it)
                video_path.parent.mkdir(parents=True, exist_ok=True) # ensure folder
                vw = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15, vsize)   # writer
            out = draw(img.copy(), tr, i, f"det {t_det[-1]:.0f} ms")   # annotated frame
            vw.write(cv2.resize(out, vsize))                         # write at the writer's size
    if vw is not None:                                               # finish
        vw.release()                                                 # close file
    return hyp, len(files), float(np.mean(t_det)), float(np.mean(t_trk))   # results


def main():                                                          # run the evaluation
    ap = argparse.ArgumentParser()                                   # parser
    ap.add_argument("--root", required=True)                         # dataset folder
    ap.add_argument("--model", default="yolox_s", choices=["yolox_nano", "yolox_tiny", "yolox_s"])   # detector
    ap.add_argument("--provider", default="dml", choices=["cpu", "dml"])   # execution provider
    ap.add_argument("--conf", type=float, default=0.3)               # detection threshold
    ap.add_argument("--max-frames", type=int, default=None)          # frames per sequence (for quick tests)
    ap.add_argument("--video-seq", default=None)                     # sequence to render (default: first)
    a = ap.parse_args()                                              # parse
    root = Path(a.root)                                              # dataset root
    seqs = sorted(p.name for p in (root / "sequences").iterdir() if p.is_dir())   # sequences
    det = YoloxDetector(PROJECT_ROOT / "data" / "models" / f"{a.model}.onnx", conf=a.conf, provider=a.provider)   # detector
    res = PROJECT_ROOT / "results"                                   # output folder
    tag = f"{a.model}_{a.provider}"                                  # file tag
    accs, timing = {}, {}                                            # outputs
    for s in seqs:                                                   # each sequence
        vp = res / f"tracking_{tag}_{s}.mp4" if s == (a.video_seq or seqs[0]) else None   # video for one sequence
        Track._next_id = 1                                           # ids restart per sequence
        hyp, n, td, tt = run_sequence(det, root, s, a.conf, a.max_frames, vp)   # detect + track
        gt, ign = read_gt(root / "annotations" / f"{s}.txt")         # ground truth
        accs[s] = score_sequence(gt, hyp, ign, n_frames=n)           # score
        timing[s] = {"frames": n, "det_ms": td, "tracker_ms": tt}    # timing
        print(f"{s}: {n} frames, detector {td:.1f} ms, tracker {tt:.2f} ms", flush=True)   # progress
    summ = compute(accs)                                             # MOTA / IDF1 table
    summ.to_csv(res / f"mot_metrics_{tag}.csv")                      # save
    cols = ["mota", "idf1", "num_switches", "num_misses", "num_false_positives", "num_objects", "recall", "precision"]   # shown columns
    L = [f"Model {a.model} ({a.provider}), conf {a.conf}, SORT tracker; categories scored: pedestrian, car, van, truck, bus; IoU 0.5.", "",
         "| sequence | MOTA | IDF1 | ID switches | misses | false positives | GT objects | recall | precision |", "|---|---|---|---|---|---|---|---|---|"]   # table
    for name, r in summ[cols].iterrows():                            # rows
        L.append(f"| {name} | {r['mota']:.3f} | {r['idf1']:.3f} | {int(r['num_switches'])} | {int(r['num_misses'])} | "
                 f"{int(r['num_false_positives'])} | {int(r['num_objects'])} | {r['recall']:.3f} | {r['precision']:.3f} |")   # row
    (res / f"mot_metrics_{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")   # save table
    json.dump(timing, open(res / f"mot_timing_{tag}.json", "w", encoding="utf-8"), indent=2)   # save timing
    print("\n".join(L))                                              # show


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
