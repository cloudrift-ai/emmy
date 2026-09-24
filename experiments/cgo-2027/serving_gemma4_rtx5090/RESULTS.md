# Gemma 4 12B, one published image, stock vLLM against Emmy — results

## NVIDIA GeForce RTX 5090 x1 (`rtx5090x1`)

Two arms on one image,
`cloudriftai/vllm-emmy-gemma-4-12b-it@sha256:3a690e9f7859d46b969dd9eaaed36f52f92c25c5595dc112aee2adb781d26e28`: stock
vLLM's native model and Emmy's standard lane (FP32 accumulation). All 24 rows succeeded (4 points x 2 arms x 3
repeats) and every request completed. Mean of three repeats; throughput is output tokens per second, latencies in
milliseconds.

| Point | Arm | Output tok/s | Median TTFT | Mean TTFT | Median TPOT | Median ITL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4096/4096 c=1 | stock | 57.4 | 549 | 557 | 17.30 | 17.31 |
| 4096/4096 c=1 | emmy | 53.6 | 590 | 616 | 18.52 | 18.53 |
| 4096/4096 c=8 | stock | 384.7 | 1106 | 1252 | 20.53 | 19.78 |
| 4096/4096 c=8 | emmy | 381.2 | 1194 | 1418 | 20.68 | 19.87 |
| 8192/256 c=4 | stock | 112.8 | 2257 | 2496 | 26.62 | 18.20 |
| 8192/256 c=4 | emmy | 105.7 | 2612 | 2825 | 27.11 | 18.78 |
| 256/256 c=64 | stock | 1301.4 | 1796 | 3497 | 27.10 | 21.73 |
| 256/256 c=64 | emmy | 1303.9 | 1701 | 2147 | 31.63 | 23.39 |

| Item | Value |
| --- | --- |
| Run | `2026-09-23_15-58-35` (run ID `20260923T155835Z`), 24 rows, all `succeeded` |
| Source revision | `12d1a98604d5`, clean tree |
| Image | `cloudriftai/vllm-emmy-gemma-4-12b-it:0.23.0-806fef9a9@sha256:3a690e9f…d26e28`, vLLM build `91df0fad4dc9` |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.173.02 |
| CPU / memory | AMD Ryzen 9 9950X3D (16 cores, 32 threads), 64.9 GB |
| OS | Ubuntu 24.04.2 LTS, kernel 7.0.0-28-generic, Docker 29.5.0 |
| Toolchain | CUDA 13.0 (nvcc 13.0.88), cuBLAS 13.1.1.3 |
| Model | `google/gemma-4-12B-it` at `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, FP16 |
| Archive | `results_rtx5090x1.tar.gz` (Git LFS), root member `2026-09-23_15-58-35/` |
