Machine: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD; GPU path: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), DirectML; ONNX Runtime 1.24.4; 200 frames (room1_frames), batch 1.

| model | input | provider | mean [ms] | p50 [ms] | p95 [ms] | FPS | pre / infer / post [ms] |
|---|---|---|---|---|---|---|---|
| yolox_nano | 416 | CPU | 15.8 | 17.6 | 19.4 | 63.1 | 1.2 / 13.4 / 1.2 |
| yolox_nano | 416 | GPU (DirectML) | 6.8 | 6.8 | 7.7 | 146.6 | 1.6 / 4.0 / 1.2 |
| yolox_tiny | 416 | CPU | 37.5 | 37.5 | 39.3 | 26.7 | 1.5 / 34.5 / 1.5 |
| yolox_tiny | 416 | GPU (DirectML) | 8.1 | 8.0 | 8.8 | 123.8 | 1.7 / 5.1 / 1.3 |
| yolox_s | 640 | CPU | 112.7 | 125.8 | 135.0 | 8.9 | 2.8 / 107.2 / 2.7 |
| yolox_s | 640 | GPU (DirectML) | 28.1 | 29.1 | 40.8 | 35.5 | 3.6 / 21.0 / 3.5 |
