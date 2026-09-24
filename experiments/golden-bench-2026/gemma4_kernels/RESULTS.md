# Gemma 4 12B kernels at sequence length 512 — results

The five projections of a Gemma 4 12B decoder layer and its sliding layers' causal attention, FP16, sequence length
512. Two lanes per card: standard (FP32 accumulation) and fast-math (`EMMY_FAST_MATH=1`, FP16 accumulation with
periodic promotion into FP32). Every Emmy row replays its golden from `golden/` under `--strict-evidence` and passes
the scaled correctness check against eager. Latency is the captured whole-program forward in microseconds. Ratios are
eager / Emmy within one task; above one favors Emmy.

## NVIDIA GeForce RTX 5090 x1 (`rtx5090x1`)

| Kernel | Eager | torch.compile | Emmy standard | Emmy fast-math | Standard | Fast-math |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_proj` 512x3840 @ 3840x4096 | 90.1 / 90.1 | 90.1 / 90.1 | 89.1 | 65.4 | 1.01x | 1.38x |
| `kv_proj` 512x3840 @ 3840x2048 | 47.1 / 47.1 | 47.1 / 47.1 | 49.2 | 36.9 | 0.96x | 1.28x |
| `o_proj` 512x4096 @ 4096x3840 | 96.2 / 95.3 | 82.1 / 82.5 | 84.3 | 67.8 | 1.14x | 1.41x |
| `mlp_gate_up` 512x3840 @ 3840x30720 | 631.2 / 630.1 | 606.0 / 607.1 | 630.6 | 402.2 | 1.00x | 1.57x |
| `mlp_down` 512x15360 @ 15360x3840 | 305.8 / 307.9 | 307.6 / 309.3 | 309.1 | 233.0 | 0.99x | 1.32x |
| `attention` causal (1, 16, 512, 256) | 35.0 / 35.0 | 34.9 / 34.9 | 41.2 | 37.6 | 0.85x | 0.93x |

Baselines are the standard-lane and the fast-math task's own measurements.

| Item | Value |
| --- | --- |
| Run | `2026-09-23_05-58-12` (run ID `20260923T055812Z`), 12 rows, all `succeeded` |
| Source revision | `dd7d5f6d3806ea2486ace5fe5a4adb8e57a09c28`, clean tree |
| Host | Ubuntu 24.04.2 LTS, kernel 7.0.0-28, AMD Ryzen 9 9950X3D |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.173.02, display attached |
| Toolchain | CUDA 13.0 (nvcc V13.0.88), PyTorch 2.13.0 |
| Archive | `results_rtx5090x1.tar.gz` (Git LFS), root member `2026-09-23_05-58-12/` |

## NVIDIA GeForce RTX 4090 x1 (`rtx4090x1`)

| Kernel | Eager | torch.compile | Emmy standard | Emmy fast-math | Standard | Fast-math |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_proj` 512x3840 @ 3840x4096 | 109.7 / 109.2 | 101.3 / 101.0 | 105.7 | 71.9 | 1.04x | 1.52x |
| `kv_proj` 512x3840 @ 3840x2048 | 52.2 / 51.3 | 51.0 / 50.9 | 59.1 | 43.9 | 0.88x | 1.17x |
| `o_proj` 512x4096 @ 4096x3840 | 120.1 / 120.0 | 106.8 / 107.1 | 110.9 | 77.2 | 1.08x | 1.55x |
| `mlp_gate_up` 512x3840 @ 3840x30720 | 848.7 / 836.1 | 730.1 / 733.7 | 766.1 | 546.8 | 1.11x | 1.53x |
| `mlp_down` 512x15360 @ 15360x3840 | 389.8 / 387.3 | 408.9 / 401.4 | 426.0 | 295.3 | 0.92x | 1.31x |
| `attention` causal (1, 16, 512, 256) | 42.4 / 42.5 | 42.3 / 42.4 | 41.1 | 37.9 | 1.03x | 1.12x |

| Item | Value |
| --- | --- |
| Run | `2026-09-17_23-41-08`, 12 rows, all `succeeded` |
| GPU | NVIDIA GeForce RTX 4090, driver 580.159.03, not display-attached |
| Toolchain | CUDA 13.3, PyTorch 2.13.0+cu130 |
| Archive | `results_rtx4090x1.tar.gz` (Git LFS) |
