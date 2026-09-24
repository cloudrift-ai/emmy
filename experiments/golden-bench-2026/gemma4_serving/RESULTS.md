# Gemma 4 12B end-to-end serving — results

## NVIDIA GeForce RTX 5090 x1 (`rtx5090x1`)

Three lanes: stock vLLM 0.23.0 (`vllm/vllm-openai:v0.23.0`), vLLM with the Emmy plugin, and the plugin's
`EMMY_FAST_MATH` fork (both `cloudriftai/vllm-emmy:0.23.0-76ff82c21`). The Emmy lanes boot under
`EMMY_STRICT_EVIDENCE=1`. All 18 rows succeeded and every request completed. Single-stream rows give the mean of three
repeats; batched points run once.

| Point | Lane | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| --- | --- | ---: | ---: | ---: |
| 256/256 c=1 | stock | 60.7 | 56.1 | 16.29 |
| 256/256 c=1 | emmy | 58.3 | 64.8 | 16.95 |
| 256/256 c=1 | emmy fast-math | 58.1 | 60.5 | 17.02 |
| 4096/4096 c=1 | stock | 57.2 | 565.0 | 17.36 |
| 4096/4096 c=1 | emmy | 55.1 | 576.3 | 18.01 |
| 4096/4096 c=1 | emmy fast-math | 55.0 | 448.4 | 18.07 |
| 4096/4096 c=4 | stock | 216.5 | 1082.5 | 18.22 |
| 4096/4096 c=4 | emmy | 210.8 | 1205.7 | 18.69 |
| 4096/4096 c=4 | emmy fast-math | 211.6 | 910.8 | 18.69 |
| 4096/4096 c=8 | stock | 383.9 | 1096.5 | 20.55 |
| 4096/4096 c=8 | emmy | 383.2 | 1190.8 | 20.59 |
| 4096/4096 c=8 | emmy fast-math | 386.4 | 930.0 | 20.48 |
| 8192/256 c=4 | stock | 112.8 | 2024.0 | 27.25 |
| 8192/256 c=4 | emmy | 107.2 | 2458.1 | 27.70 |
| 8192/256 c=4 | emmy fast-math | 120.4 | 1881.7 | 25.78 |
| 256/256 c=64 | stock | 1436.1 | 1686.9 | 27.72 |
| 256/256 c=64 | emmy | 1224.8 | 1823.5 | 30.40 |
| 256/256 c=64 | emmy fast-math | 1329.4 | 1644.5 | 27.40 |

| Item | Value |
| --- | --- |
| Run | `2026-09-22_13-25-38` (run ID `20260922T132538Z`), 18 rows, all `succeeded` |
| Source revision | `21fcb94a44d5d771b389ead75fd932fc01cf7ceb`, clean tree |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.173.02 |
| CPU / memory | AMD Ryzen 9 9950X3D (16 cores, 32 threads), 64.9 GB |
| OS | Ubuntu 24.04.2 LTS, kernel 7.0.0-28-generic, Docker 29.5.0 |
| Model | `google/gemma-4-12B-it` at `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, FP16 |
| Archive | `results_rtx5090x1.tar.gz` (Git LFS), root member `2026-09-22_13-25-38/` |

Row IDs, by point and lane (stock, emmy, fast-math) — 256/256 c=1: `c9e49861fb8c`, `af1f924afb43`, `fa406a490dde`;
4096/4096 c=1: `a45c51be8eff`, `6b35bdaaaf59`, `dbe64ae9822c`; 4096/4096 c=4: `f53020eecbc5`, `8b36549284d9`,
`b3a5630bf265`; 4096/4096 c=8: `127cb77a17dc`, `71320a273565`, `ef0e0564dc08`; 8192/256 c=4: `f763c1dd0350`,
`17d539c42127`, `5e831ca8e713`; 256/256 c=64: `c7dbaea21ffa`, `e10b538c22f2`, `ec83bae58318`.
