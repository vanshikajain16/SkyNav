"""Benchmark detector FPS and latency: models x execution providers (CPU vs DirectML GPU).

usage: python -m perception.benchmark [--images DIR_OR_GLOB] [--n 200]
Latency is per frame for the whole detector call (preprocess + inference + decode/NMS), with a warm-up phase that is
excluded. Each frame is a real image; the same frames are used for every configuration.
"""
from __future__ import annotations                                   # forward type hints
import argparse                                                      # command line
import json                                                          # metrics file
import platform                                                      # machine description
from pathlib import Path                                             # paths
import numpy as np                                                   # arrays
import cv2                                                           # image loading
import onnxruntime as ort                                            # version info

from vo.datasets import PROJECT_ROOT, Sequence                       # default image source
from perception.detector import YoloxDetector                        # detector

MODELS = ["yolox_nano", "yolox_tiny", "yolox_s"]                     # available ONNX models (input 416 / 416 / 640)


def load_frames(src: str | None, n: int) -> list:                    # real images for the benchmark
    if src:                                                          # user-provided folder
        files = sorted(Path(src).glob("*.jpg")) + sorted(Path(src).glob("*.png"))   # images
        return [cv2.imread(str(f)) for f in files[:n]]               # colour frames
    seq = Sequence("tumvi_room1")                                    # default: room1 frames (grayscale fisheye)
    return [seq.frame_u8(i) for i in np.linspace(0, len(seq) - 1, n).astype(int)]   # spread over the sequence


def bench(model: str, provider: str, frames: list, warmup: int = 15) -> dict:   # one configuration
    det = YoloxDetector(PROJECT_ROOT / "data" / "models" / f"{model}.onnx", conf=0.3, provider=provider)   # detector
    for f in frames[:warmup]:                                        # warm-up (kernel compilation, caches)
        det.detect(f)                                                # discarded
    tot, pre, inf, post = [], [], [], []                             # per-frame timings
    for f in frames:                                                 # measured frames
        det.detect(f)                                                # run
        tm = det.last_times                                          # breakdown
        tot.append(tm["total_ms"]); pre.append(tm["pre_ms"]); inf.append(tm["infer_ms"]); post.append(tm["post_ms"])   # store
    tot = np.array(tot)                                              # array
    return {"model": model, "input": det.size, "provider_requested": provider, "provider_used": det.provider_used,
            "frames": len(frames), "mean_ms": float(tot.mean()), "p50_ms": float(np.percentile(tot, 50)),
            "p95_ms": float(np.percentile(tot, 95)), "fps_mean": float(1000.0 / tot.mean()),
            "pre_ms": float(np.mean(pre)), "infer_ms": float(np.mean(inf)), "post_ms": float(np.mean(post))}   # numbers


def main():                                                          # run the benchmark
    ap = argparse.ArgumentParser()                                   # parser
    ap.add_argument("--images", default=None)                        # folder of images (default: room1 frames)
    ap.add_argument("--n", type=int, default=200)                    # frames per configuration
    ap.add_argument("--tag", default="room1_frames")                 # output name suffix
    a = ap.parse_args()                                              # parse
    frames = load_frames(a.images, a.n)                              # images
    print(f"{len(frames)} frames, shape {frames[0].shape}")          # info
    rows = []                                                        # results
    for model in MODELS:                                             # each model
        for prov in ("cpu", "dml"):                                  # CPU then GPU (DirectML)
            r = bench(model, prov, frames)                           # measure
            rows.append(r)                                           # keep
            print(f"{model:11s} {prov:3s} used={r['provider_used']:22s} mean {r['mean_ms']:7.1f} ms  p95 {r['p95_ms']:7.1f} ms  "
                  f"{r['fps_mean']:6.1f} FPS  (pre {r['pre_ms']:.1f} / infer {r['infer_ms']:.1f} / post {r['post_ms']:.1f})")   # progress
    hw = {"cpu": platform.processor(), "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), DirectML", "onnxruntime": ort.__version__}   # machine
    L = [f"Machine: {hw['cpu']}; GPU path: {hw['gpu']}; ONNX Runtime {hw['onnxruntime']}; {len(frames)} frames ({a.tag}), batch 1.", "",
         "| model | input | provider | mean [ms] | p50 [ms] | p95 [ms] | FPS | pre / infer / post [ms] |", "|---|---|---|---|---|---|---|---|"]   # table
    for r in rows:                                                   # rows
        L.append(f"| {r['model']} | {r['input']} | {'GPU (DirectML)' if r['provider_used'].startswith('Dml') else 'CPU'} | {r['mean_ms']:.1f} | "
                 f"{r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['fps_mean']:.1f} | {r['pre_ms']:.1f} / {r['infer_ms']:.1f} / {r['post_ms']:.1f} |")   # row
    res = PROJECT_ROOT / "results"                                   # output folder
    (res / f"benchmark_{a.tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")   # save table
    json.dump({"hardware": hw, "rows": rows}, open(res / f"benchmark_{a.tag}.json", "w", encoding="utf-8"), indent=2)   # save raw
    print("\n".join(L))                                              # show


if __name__ == "__main__":                                           # required guard on Windows
    main()                                                           # go
