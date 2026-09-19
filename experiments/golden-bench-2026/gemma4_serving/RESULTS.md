# Gemma 4 12B end-to-end serving — results

## NVIDIA GeForce RTX 5090 x1 (`rtx5090x1`)

### Question

Does the stock vLLM column of the serving tables in "Outperforming vLLM and Llama.cpp on Gemma4-12B" (published
2026-08-01) still hold on this box? This run measures the non-Emmy lane only: stock vLLM 0.23.0 at the article's
workload points. The two Emmy lanes of the recipe (twelve rows) were not part of this run.

### Status

All 6 selected rows succeeded in one run (`2026-09-19_06-15-04`, run ID `20260919T061504Z`), selected with
`--filter 'engine.llm.vllm.image=vllm/vllm-openai:*'`. Every request of every repeat completed (0 failed). The
twelve Emmy rows were filtered out and are not in the archive.

### Protocol

`emmy bench experiments/golden-bench-2026/gemma4_serving --local --filter 'engine.llm.vllm.image=vllm/vllm-openai:*'`
on a pre-allocated RTX 5090, one server boot per row. Image `vllm/vllm-openai:v0.23.0`, `--dtype float16
--no-enable-prefix-caching`, context 16384, `--gpu-memory-utilization 0.96`. The client is `vllm bench serve` on
random prompts with seed 0, temperature 0 and `--ignore-eos`. The single-stream points repeat three times and the
batched points run once. Desktop applications were closed first so the 0.96 memory setting fits beside the display
server (185 MiB in use before the run).

### Measurements

Output token throughput and median latencies. The single-stream rows give the mean of three repeats.

| Tokens in | Tokens out | Concurrency | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 256 | 1 | 16 | 60.63 | 56.3 | 16.31 |
| 4096 | 4096 | 1 | 8 | 57.23 | 565.7 | 17.34 |
| 4096 | 4096 | 4 | 32 | 216.52 | 1085.0 | 18.22 |
| 4096 | 4096 | 8 | 64 | 384.21 | 1098.5 | 20.53 |
| 8192 | 256 | 4 | 16 | 112.62 | 2028.3 | 27.26 |
| 256 | 256 | 64 | 256 | 1434.07 | 1692.9 | 27.80 |

### Comparison with the article

| Point | Article tok/s | This run | Article TTFT / TPOT (ms) | This run |
| --- | ---: | ---: | ---: | ---: |
| 4096/4096 c=1 | 57.2 | 57.23 | 566 / 17.4 | 566 / 17.3 |
| 4096/4096 c=4 | 216.4 | 216.52 | 1088 / 18.2 | 1085 / 18.2 |
| 4096/4096 c=8 | 383.6 | 384.21 | 1100 / 20.6 | 1099 / 20.5 |
| 8192/256 c=4 | 112.0 | 112.62 | 2429 / 26.2 | 2028 / 27.3 |
| 256/256 c=64 | 1425.1 | 1434.07 | 1468 / 28.0 | 1693 / 27.8 |

The article does not publish the 256/256 c=1 point.

### What the numbers say

Stock vLLM reproduces the article. Throughput is within 0.7% at every published point, and TPOT is within 0.1 ms
at the three 4K/4K points. The two differences are both first-token latency, and neither is a change in the engine:

- **256/256 c=64 TTFT is a different measurement.** The article's 1468 ms comes from one wave of 64 requests
  (`--num-prompts 64`, its footnote 2). The recipe runs 256 prompts, so the median lands on queued second- and
  third-wave requests. The TPOT half of that cell uses the same protocol in both and agrees (27.8 against 28.0).
- **8192/256 c=4 TTFT moves between runs of the same image.** It measured 2028 ms here, 2382 ms in yesterday's run
  on this box (an unfinished run whose stock rows all completed) and 2429 ms in the article. TPOT moves the other
  way (27.3, 26.0, 26.2), and throughput stays at 112.6 to 112.7 in all three. The point runs once, with 16 prompts
  at four at a time. The order in which four 8K prefills and their decodes share a step is what moves; the engine
  does not. Compare this cell only within one run.

The stock baseline is stable enough to reuse. It is the same on the local 2026-08-05 run of the article-era recipe
(57.21, 216.48 and 383.38 tok/s at the 4K/4K points), in yesterday's run and in this one, to within 0.3%.

### Repeat variation

Across the three repeats of each single-stream point: throughput spread 0.33% at 256/256 (60.54 to 60.74 tok/s) and
0.09% at 4096/4096 (57.21 to 57.26 tok/s). Median TPOT spread was 0.03 ms and 0.01 ms, and median TTFT was 56.05 to
56.47 ms and 564.7 to 566.5 ms. The batched points run once, so their variation comes from comparing runs, as above.

### Limitations

- The Emmy lanes are not in this run, so this section makes no stock-versus-Emmy comparison. The recipe pins their
  image to `cloudriftai/vllm-emmy:0.23.0-73b8e5377`, built at a commit that no longer exists after the branch was
  rebased. An Emmy run needs an image rebuilt at the branch head.
- Single run on one box. The batched points and the RAG point have one repeat each.
- The comparison with the article is direct only where the protocol matches. The c=64 TTFT is not comparable (see
  above).

### System

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.173.02 |
| CPU / memory | AMD Ryzen 9 9950X3D (16 cores, 32 threads), 64.9 GB |
| OS | Ubuntu 24.04.2 LTS, kernel 7.0.0-28-generic, Docker 29.5.0 |
| Engine | `vllm/vllm-openai:v0.23.0` (vLLM 0.23.0) |
| Model | `google/gemma-4-12B-it` at `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, FP16 |
| Harness revision | `66f5c905c`, clean tree |
| Run window | 2026-09-19 06:15 to 07:23 UTC |

### Archive

`results_rtx5090x1.tar.gz` (Git LFS) holds `2026-09-19_06-15-04/` exactly as `emmy bench` wrote it: the run logs
`benchmark.log` and `benchmark_rtx5090_x_1.log`, and three files per row, an `.experiment.yaml`, a
`.benchmark.log` and a `.server.log`. Each row's files are named
`rtx5090x1_<point>_gmu0.96_ivllm-vllm-oai-v0.23.0_<row id>`, with row IDs `c9e49861fb8c` (256/256 c=1),
`a45c51be8eff` (4096/4096 c=1), `f53020eecbc5` (4096/4096 c=4), `127cb77a17dc` (4096/4096 c=8), `f763c1dd0350`
(8192/256 c=4) and `c7dbaea21ffa` (256/256 c=64).
