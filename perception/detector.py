"""YOLOX-nano object detector on ONNX Runtime (CPU). No torch, no ROS."""
from __future__ import annotations                                   # forward type hints
import time                                                          # timing
from pathlib import Path                                             # paths
import numpy as np                                                   # arrays
import cv2                                                           # resizing and NMS
import onnxruntime as ort                                            # inference

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]       # 80 COCO class names


class YoloxDetector:                                                 # detector wrapper
    def __init__(self, model_path: str | Path, conf: float = 0.3, nms_iou: float = 0.45, provider: str = "cpu"):   # load model
        so = ort.SessionOptions()                                    # session options
        if provider == "dml":                                        # DirectML needs these two settings (ONNX Runtime docs)
            so.enable_mem_pattern = False                            # memory pattern must be off
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL     # sequential execution
            provs = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]   # GPU first, CPU fallback
        elif provider == "cpu":                                      # plain CPU
            provs = ["CPUExecutionProvider"]                         # CPU only
        else:                                                        # unknown name
            raise ValueError(f"provider must be 'cpu' or 'dml', got {provider!r}")   # fail loudly
        self.sess = ort.InferenceSession(str(model_path), sess_options=so, providers=provs)   # session
        self.provider_used = self.sess.get_providers()[0]            # what ONNX Runtime really selected
        self.last_times = {}                                         # per-call timing breakdown [ms]
        self.inp = self.sess.get_inputs()[0].name                    # input tensor name
        self.size = int(self.sess.get_inputs()[0].shape[2])          # square input size (416)
        self.conf, self.nms_iou = conf, nms_iou                      # thresholds
        s = [8, 16, 32]                                              # YOLOX strides
        grids, strides = [], []                                      # decode tables
        for st in s:                                                 # per feature map
            n = self.size // st                                      # cells per side
            gx, gy = np.meshgrid(np.arange(n), np.arange(n))         # cell indices
            grids.append(np.stack([gx, gy], 2).reshape(-1, 2))       # (n*n, 2)
            strides.append(np.full((n * n, 1), st))                  # stride per cell
        self.grids = np.concatenate(grids).astype(np.float32)        # (3549, 2)
        self.strides = np.concatenate(strides).astype(np.float32)    # (3549, 1)

    def detect(self, img: np.ndarray) -> np.ndarray:                 # img: HxW or HxWx3, uint8
        t0 = time.perf_counter()                                     # start of preprocessing
        if img.ndim == 2:                                            # grayscale input
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)              # replicate to 3 channels
        h, w = img.shape[:2]                                         # original size
        r = min(self.size / h, self.size / w)                        # letterbox ratio
        nh, nw = int(round(h * r)), int(round(w * r))                # resized size
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)   # grey padding
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)   # place image top-left
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32)       # NCHW, 0..255 (YOLOX convention)
        t1 = time.perf_counter()                                     # preprocessing done
        out = self.sess.run(None, {self.inp: x})[0][0]               # (N, 85)
        t2 = time.perf_counter()                                     # inference done
        res = self._post(out, r, h, w)                               # decode + NMS
        t3 = time.perf_counter()                                     # postprocessing done
        self.last_times = {"pre_ms": 1e3 * (t1 - t0), "infer_ms": 1e3 * (t2 - t1), "post_ms": 1e3 * (t3 - t2),
                           "total_ms": 1e3 * (t3 - t0)}              # breakdown
        return res                                                   # (N, 6)

    def _post(self, out: np.ndarray, r: float, h: int, w: int) -> np.ndarray:   # decode raw outputs
        xy = (out[:, :2] + self.grids) * self.strides                # centres
        wh = np.exp(out[:, 2:4]) * self.strides                      # sizes
        scores_all = out[:, 4:5] * out[:, 5:]                        # objectness * class prob
        cls = scores_all.argmax(1)                                   # best class
        score = scores_all[np.arange(len(cls)), cls]                 # its score
        keep = score > self.conf                                     # confidence filter
        if not keep.any():                                           # nothing above threshold
            return np.zeros((0, 6), np.float32)                      # empty
        xy, wh, score, cls = xy[keep], wh[keep], score[keep], cls[keep]   # apply
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1) / r   # xyxy in original pixels
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)           # clip x
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)           # clip y
        xywh = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)   # NMS format
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), score.tolist(), self.conf, self.nms_iou)   # class-agnostic NMS
        idx = np.array(idx).reshape(-1)                              # indices
        return np.concatenate([boxes[idx], score[idx, None], cls[idx, None]], axis=1).astype(np.float32)   # (N,6)
