# Gemma 4 12B end-to-end serving — results

## NVIDIA GeForce RTX 5090 x1 (`rtx5090x1`)

### Question

Do the serving tables of "Outperforming vLLM and Llama.cpp on Gemma4-12B" (published 2026-08-01) still hold on the
current compiler, after the post-attention halves of the serving golden were re-recorded on the gate/up operand cut?
All three lanes of the recipe ran: stock vLLM 0.23.0, vLLM with the Emmy plugin, and the plugin's `EMMY_FAST_MATH`
fork.

### Status

All 18 rows succeeded in one run (`2026-09-20_23-25-14`, run ID `20260920T232514Z`). Every request of every repeat
completed. No lane logged an `EvidenceError`: with `EMMY_STRICT_EVIDENCE=1` every fork of every kernel the servers
compiled was decided by a golden row. The twelve Emmy rows cover eight distinct serving shapes; those eight compiled
from scratch (about 18 minutes each) and the remaining four hit the pack those compiles had just written, so nothing
in this run served a plan built from the rows the re-record replaced.

### Protocol

`emmy bench experiments/golden-bench-2026/gemma4_serving --local` on a pre-allocated RTX 5090, one server boot per
row. Stock lane: `vllm/vllm-openai:v0.23.0`. Emmy lanes: `cloudriftai/vllm-emmy:0.23.0-6328269d1`, the plain plugin
image built at the re-record commit, with a pack directory of its own. Every lane runs `--dtype float16
--no-enable-prefix-caching`, context 16384, `--gpu-memory-utilization 0.96`; the Emmy lanes set the decode bucket to
the concurrency and the 2048-token chunk quantum on the mixed points, as the article's recipes do. The client is
`vllm bench serve` on random prompts with seed 0, temperature 0 and `--ignore-eos`. Single-stream points repeat three
times, batched points run once. Desktop applications were closed first so the 0.96 memory setting fits beside the
display server.

### Measurements

Output token throughput and median latencies. Single-stream rows give the mean of three repeats.

| Point | Lane | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| --- | --- | ---: | ---: | ---: |
| 256/256 c=1 | stock | 60.9 | 56.1 | 16.3 |
| 256/256 c=1 | emmy | 53.9 | 67.5 | 18.4 |
| 256/256 c=1 | emmy fast-math | 54.6 | 65.3 | 18.1 |
| 4096/4096 c=1 | stock | 57.2 | 566.0 | 17.3 |
| 4096/4096 c=1 | emmy | 51.2 | 571.2 | 19.4 |
| 4096/4096 c=1 | emmy fast-math | 51.8 | 549.5 | 19.2 |
| 4096/4096 c=4 | stock | 216.4 | 1086.2 | 18.2 |
| 4096/4096 c=4 | emmy | 199.2 | 1207.8 | 19.8 |
| 4096/4096 c=4 | emmy fast-math | 198.8 | 1146.8 | 19.9 |
| 4096/4096 c=8 | stock | 383.6 | 1101.7 | 20.6 |
| 4096/4096 c=8 | emmy | 361.0 | 1260.0 | 21.9 |
| 4096/4096 c=8 | emmy fast-math | 360.4 | 1222.8 | 21.9 |
| 8192/256 c=4 | stock | 112.7 | 2030.0 | 27.3 |
| 8192/256 c=4 | emmy | 105.0 | 2375.7 | 28.8 |
| 8192/256 c=4 | emmy fast-math | 105.1 | 2791.6 | 26.4 |
| 256/256 c=64 | stock | 1435.6 | 1688.4 | 27.7 |
| 256/256 c=64 | emmy | 1164.1 | 2032.0 | 30.1 |
| 256/256 c=64 | emmy fast-math | 1184.4 | 1966.6 | 29.6 |

### Comparison with the article

The standard Emmy lane against the article's published Emmy column, with the same lane's numbers from the run before
the re-record (2026-09-19, the rows this change replaced) for the size of the move:

| Point | Article Emmy | Before | Now | Stock now |
| --- | ---: | ---: | ---: | ---: |
| 4096/4096 c=1 | 54.8, 628 / 18.1 | 49.5, 896 / 20.0 | 51.2, 571 / 19.4 | 57.2, 566 / 17.3 |
| 4096/4096 c=4 | 206.4, 1266 / 19.1 | 195.0, 1721 / 20.1 | 199.2, 1208 / 19.8 | 216.4, 1086 / 18.2 |
| 4096/4096 c=8 | 375.3, 1236 / 21.0 | 350.9, 1770 / 22.4 | 361.0, 1260 / 21.9 | 383.6, 1102 / 20.6 |
| 8192/256 c=4 | 101.7, 2655 / 29.2 | 86.8, 3330 / 32.9 | 105.0, 2376 / 28.8 | 112.7, 2030 / 27.3 |
| 256/256 c=64 | 1138.8, 1772 / 30.0 | 951.8, 3075 / 36.3 | 1164.1, 2032 / 30.1 | 1435.6, 1688 / 27.7 |

The article does not publish the 256/256 c=1 point.

### What the numbers say

**First-token latency is where the re-record lands.** Every point improved by 26 to 34%, and four of the five now
beat the article: 571 ms against 628 at the single-stream 4K point, where stock measures 566. The 4K point is at
parity with stock, which is new — the previous rows trailed it by 58%. The cause is arithmetic: prefill runs the
post-attention half at 2048 tokens per chunk, and that half went from 7223 to 3984 us per layer, which over 48 layers
and two chunks is 311 ms of the 325 ms the point actually gained.

**Throughput beats the article at two points.** The RAG point reaches 105.0 against 101.7, and the 64-stream point
1164 against 1139 — the latter up 22% from the previous rows, because its decode width of 64 gained 25% per layer
half and its prefill chunks gained the rest.

**The three decode-bound points are still 4 to 6% short of the article**, and all of that is per-token latency: 19.4
ms against 18.1 at c=1. An `nsys` trace of this image on this box (2026-09-22, 256/256 c=1, where the lane measures
18.4 ms) shows the decode step is GPU-bound: the host adds about 0.4 ms per step, as it does for stock. Of the Emmy
lane's 18.1 ms of kernel time, 0.9 ms is vLLM's native rotary path. The recipe starts the plain image with bare `vllm
serve`, so the plugin got that path where the article's image forced the fused kernel through its launch config; the
plugin now takes the fused kernel on its own, which brings the same lane to 17.6 ms. What remains against stock's
15.9 ms of kernel time is about 1 ms of small layer kernels (norm statistics and the cut's elementwise kernel, 5 to 8
us each, where stock's fused norms take 1 to 3) and 0.3 ms of projections. An earlier version of this paragraph
called the gap ~3.6 ms of plugin cost per step. That figure came from summing golden rows, which are timed with the
cache warm: the width-32 output projection is recorded at 14.2 us and runs at 22.3 in serving.

**Fast-math is no longer worth a lane at most points.** It wins 0.5 to 1.7% of throughput at five of six points and
matches the standard lane at the sixth. The one exception is the RAG point, where it takes per-token latency to 26.4
ms — below stock's 27.3 — while paying 416 ms of first-token latency for it. At prefill widths the standard lane's
gate/up now runs at 204 TFLOPS against the card's ~209 dense peak, so there is nothing for a lower-precision
accumulate to recover there.

**Stock reproduces itself.** Its six rows are within 0.3% of the stock-only run of 2026-09-19 on this box at every
point, which is what makes the Emmy comparison above a comparison and not a drift measurement.

### Limitations

- Single run on one box. The batched points and the RAG point have one repeat each; compare them only within a run.
- The 8192/256 c=4 first-token latency moves between runs of the same image (2030, 2028 and 2429 ms across three
  stock runs while throughput held at 112.6 to 112.7). The fast-math row's 2792 ms sits inside that spread.
- The 256/256 c=64 first-token latency is not comparable with the article's, which uses one wave of 64 requests
  where the recipe queues 256; the per-token half of that cell uses the same protocol in both.
- The Emmy lanes of this run ran vLLM's native rotary path (see above), about 4% of throughput at the single-stream
  points. The table has not been re-measured with the plugin fix.

### System

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.173.02 |
| CPU / memory | AMD Ryzen 9 9950X3D (16 cores, 32 threads), 64.9 GB |
| OS | Ubuntu 24.04.2 LTS, kernel 7.0.0-28-generic, Docker 29.5.0 |
| Engine | vLLM 0.23.0 (`vllm/vllm-openai:v0.23.0`, `cloudriftai/vllm-emmy:0.23.0-6328269d1`) |
| Model | `google/gemma-4-12B-it` at `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, FP16 |
| Harness revision | `27535aba2`, clean tree |
| Run window | 2026-09-20 23:25 to 2026-09-21 05:17 UTC |

### Archive

`results_rtx5090x1.tar.gz` (Git LFS) holds `2026-09-20_23-25-14/` exactly as `emmy bench` wrote it: the run logs
`benchmark.log` and `benchmark_rtx5090_x_1.log`, and three files per row, an `.experiment.yaml`, a `.benchmark.log`
and a `.server.log`. Row IDs, by point and lane — 256/256 c=1: `c9e49861fb8c` stock, `cd10898a4879` emmy,
`f1e9a385e636` fast-math; 4096/4096 c=1: `a45c51be8eff`, `0ba6ce0ffe68`, `fb1051410bd7`; 4096/4096 c=4:
`f53020eecbc5`, `dff7612efdc1`, `2627d992b007`; 4096/4096 c=8: `127cb77a17dc`, `4834e789ee92`, `6697d4658d95`;
8192/256 c=4: `f763c1dd0350`, `55834735532d`, `b2907048119b`; 256/256 c=64: `c7dbaea21ffa`, `e5b0ca460afc`,
`e9f3d25c9206`.
