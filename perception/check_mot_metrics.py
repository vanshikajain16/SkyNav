"""Cross-check: motmetrics-based MOTA / IDF1 vs the hand-written CLEAR-MOT on the synthetic scenes (validates the metric path)."""
from __future__ import annotations                                   # forward type hints
import numpy as np                                                   # arrays

from perception.check_perception import synth_scene, clear_mot      # synthetic data and the earlier metric
from perception.tracker import SortTracker, TrackerConfig, Track     # tracker
from perception.mot_metrics import score_sequence, compute           # motmetrics path


def main():                                                          # compare both implementations
    for label, kw in (("clean", dict(miss=0.05, fp_rate=0.2, sigma=1.5)), ("hard", dict(miss=0.25, fp_rate=1.0, sigma=4.0))):   # two scenes
        gt, dets = synth_scene(**kw)                                 # scene
        Track._next_id = 1                                           # reset ids
        tr = SortTracker(TrackerConfig())                            # tracker
        rep = {f: tr.update(dets[f]) for f in sorted(dets)}          # tracker output (id, x1, y1, x2, y2, cls)
        mine = clear_mot(gt, rep)                                    # earlier implementation
        g = {f: (np.array([i for i, _ in v], int), np.array([b for _, b in v]).reshape(-1, 4)) for f, v in gt.items()}   # GT for motmetrics
        h = {f: (rep[f][:, 0].astype(int), rep[f][:, 1:5]) for f in rep}   # hypotheses
        acc = score_sequence(g, h, {}, n_frames=len(gt))             # motmetrics accumulator
        s = compute({label: acc}).loc["OVERALL"]                     # summary
        print(f"{label}: motmetrics MOTA {s['mota']:.4f} IDF1 {s['idf1']:.4f} IDSW {int(s['num_switches'])} FN {int(s['num_misses'])} "
              f"FP {int(s['num_false_positives'])} | hand-written MOTA {mine['MOTA']:.4f} IDSW {mine['IDSW']} FN {mine['FN']} FP {mine['FP']}")   # compare


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
