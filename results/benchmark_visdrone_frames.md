Machine: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD; GPU path: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), DirectML; ONNX Runtime 1.24.4; 200 frames (visdrone_frames), batch 1.

| model | input | provider | mean [ms] | p50 [ms] | p95 [ms] | FPS | pre / infer / post [ms] |
|---|---|---|---|---|---|---|---|
| yolox_nano | 416 | CPU | 11.0 | 11.1 | 12.2 | 90.6 | 1.0 / 9.1 / 0.9 |
| yolox_nano | 416 | GPU (DirectML) | 5.6 | 4.5 | 5.4 | 178.4 | 0.8 / 4.0 / 0.8 |
| yolox_tiny | 416 | CPU | 27.3 | 24.7 | 33.0 | 36.6 | 1.3 / 24.8 / 1.3 |
| yolox_tiny | 416 | GPU (DirectML) | 7.8 | 7.6 | 9.1 | 128.6 | 1.4 / 5.0 / 1.4 |
| yolox_s | 640 | CPU | 114.9 | 112.7 | 126.4 | 8.7 | 2.6 / 109.4 / 2.9 |
| yolox_s | 640 | GPU (DirectML) | 13.5 | 13.3 | 15.4 | 74.3 | 2.1 / 8.9 / 2.5 |
