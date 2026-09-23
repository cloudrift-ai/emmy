# Gemma 4 12B end-to-end serving — results

## NVIDIA GeForce RTX 5090 x1 (`rtx5090x1`)

### Question

Do the serving tables of "Outperforming vLLM and Llama.cpp on Gemma4-12B" (published 2026-08-01) still hold on the
current compiler, after the serving golden's targets were re-lifted, its fast-math prefill rows re-recorded on
two-channel TMA staging, and the single-stream points moved to decode bucket 8? All three lanes of the recipe ran:
stock vLLM 0.23.0, vLLM with the Emmy plugin, and the plugin's `EMMY_FAST_MATH` fork.

### Status

All 18 rows succeeded in one run (`2026-09-22_13-25-38`, run ID `20260922T132538Z`). Every request of every repeat
completed. No lane logged an `EvidenceError`: with `EMMY_STRICT_EVIDENCE=1` every fork of every kernel the servers
compiled was decided by a golden row. On `main` this lane could not have run at all — the audit deployed zero twins
from the golden's rows — so this is the first run of the article's protocol on a compiler where measured evidence
matches on the exact typed kernel identity.

### Protocol

`emmy bench experiments/golden-bench-2026/gemma4_serving --local` on a pre-allocated RTX 5090, one server boot per
row. Stock lane: `vllm/vllm-openai:v0.23.0`. Emmy lanes: `cloudriftai/vllm-emmy:0.23.0-76ff82c21`, the plain plugin
image built at this branch's head, with a pack directory of its own. Every lane runs `--dtype float16
--no-enable-prefix-caching`, context 16384, `--gpu-memory-utilization 0.96`. The Emmy lanes set the decode bucket to
the smallest recorded width that holds the concurrency — 8 at c=1, c=4 and c=8, 64 at c=64 — and the 2048-token chunk
quantum on the mixed points. The article's c=1 points used bucket 32; the width-8 twins are recorded now and run
about 12 us per layer faster. The client is `vllm bench serve` on random prompts with seed 0, temperature 0 and
`--ignore-eos`. Single-stream points repeat three times, batched points run once.

### Measurements

Output token throughput and median latencies. Single-stream rows give the mean of three repeats.

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

### Comparison with the article and with the run before this branch

Output tok/s and median TTFT / TPOT. "Before" is the 2026-09-20 run of this recipe, the rows this branch replaced.

| Point | Article Emmy | Before (std) | Now (std) | Before (fm) | Now (fm) | Stock now |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256/256 c=1 | — | 53.9, 68 / 18.4 | 58.3, 65 / 17.0 | 54.6, 65 / 18.1 | 58.1, 60 / 17.0 | 60.7, 56 / 16.3 |
| 4096/4096 c=1 | 54.8, 628 / 18.1 | 51.2, 571 / 19.4 | 55.1, 576 / 18.0 | 51.8, 550 / 19.2 | 55.0, 448 / 18.1 | 57.2, 565 / 17.4 |
| 4096/4096 c=4 | 206.4, 1266 / 19.1 | 199.2, 1208 / 19.8 | 210.8, 1206 / 18.7 | 198.8, 1147 / 19.9 | 211.6, 911 / 18.7 | 216.5, 1083 / 18.2 |
| 4096/4096 c=8 | 375.3, 1236 / 21.0 | 361.0, 1260 / 21.9 | 383.2, 1191 / 20.6 | 360.4, 1223 / 21.9 | 386.4, 930 / 20.5 | 383.9, 1096 / 20.6 |
| 8192/256 c=4 | 101.7, 2655 / 29.2 | 105.0, 2376 / 28.8 | 107.2, 2458 / 27.7 | 105.1, 2792 / 26.4 | 120.4, 1882 / 25.8 | 112.8, 2024 / 27.3 |
| 256/256 c=64 | 1138.8, 1772 / 30.0 | 1164.1, 2032 / 30.1 | 1224.8, 1823 / 30.4 | 1184.4, 1967 / 29.6 | 1329.4, 1645 / 27.4 | 1436.1, 1687 / 27.7 |

### What the numbers say

**The article's fast-math headline reproduces and is beaten.** Its claim was first-token latency below stock on the
long-prompt points; the previous run did not reproduce it (550 ms against stock's 566 at the single-stream 4K point).
Fast-math now measures 448 ms there against stock's 565 and the article's own 471, and it is ahead of stock at every
other long-prompt point: 911 against 1083 at c=4, 930 against 1096 at c=8, 1882 against 2024 on the RAG point, 1645
against 1687 at c=64. The cause is arithmetic: at the 4096-token chunk this point prefills in one
step, the fast-math post-attention half went from 6766 to 4822 us per layer and the pre half from 1149 to 1031, which
over 48 layers is 99 ms of the 102 ms the point gained.

**Fast-math beats stock on throughput at two points.** The RAG point reaches 120.4 tok/s against stock's 112.8 (+7%)
and the 8-stream point 386.4 against 383.9. Both are prefill-heavy, which is where the re-recorded rows land.

**Per-token latency is close to stock everywhere except the 64-stream point.** The standard lane trails stock by 4%
at 256/256 c=1 (16.95 against 16.29 ms), 3.7% at the 4K point, 2.6% at c=4 and 0.2% at c=8, where the previous run
trailed by 13%, 12%, 9% and 6%. An `nsys` trace of the c=4 point on this image puts a decode step at 17.63 ms of GPU
time against stock's 17.19: the projections are at parity once the weights are cold (about 283 us per layer averaged
over sliding and global layers against stock's 284), and what remains is the small norm kernels — Emmy runs about 22
us of them per layer where stock's fused norms take 13.5.

**The 64-stream point is the outlier.** Throughput trails stock by 15% in the standard lane and 7% in fast-math, with
median per-token latency 30.4 and 27.4 against 27.7 ms. Both lanes sit at ~99% KV usage with 40 to 50 requests
running, and median inter-token latency differs by 5%, so the gap is in how the steps are composed rather than in one
kernel. A measured A/B at c=4 and c=8 ruled out the leading hypothesis: running a full chunk step and its decode
riders as one symbolic pass instead of the chunk twin plus the decode twin changed nothing (195.2 against 195.1 tok/s
at c=4, 330.5 against 329.5 at c=8).

**The standard lane's global layers got slower, deliberately.** Three of its twins carried FP16-accumulate rows —
the fast-math precision — because a recorded row could name any tile. They are re-recorded on FP32 accumulation and
the post half at 4096 tokens went from 7135 to 8424 us per layer on the eight global layers; the sliding layers
gained 2%. The standard lane's first-token latency is within 1% of the previous run at the single-stream 4K point in
spite of it.

**Stock reproduces itself.** Its six rows are within 0.5% of the 2026-09-20 run at every point, which is what makes
the comparison above a comparison rather than a drift measurement.

### Limitations

- Single run on one box. The batched points and the RAG point have one repeat each; compare them only within a run.
- The 8192/256 c=4 first-token latency moves between runs of one image (2024, 2028, 2030 and 2429 ms across four
  stock runs while throughput held at 112.6 to 112.8), so the fast-math row's 1882 ms is a real move but a single
  sample inside a noisy cell.
- The 256/256 c=64 first-token latency is not comparable with the article's, which uses one wave of 64 requests where
  the recipe queues 256.
- Emmy and stock run different model code (`EmmyGenModel` against vLLM's native Gemma), so these rows are a
  system-level comparison, not an isolation of compiled kernels.
- The width-1 decode twins do not deploy from the golden's rows; the runner drops that tier at boot as slower than
  the bucket twins, so no row here ran them.

### System

| Item | Value |
| --- | --- |
| Run | `2026-09-22_13-25-38` (run ID `20260922T132538Z`), 18 rows, all `succeeded` |
| Source revision | `21fcb94a44d5d771b389ead75fd932fc01cf7ceb`, clean tree |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.173.02 |
| CPU / memory | AMD Ryzen 9 9950X3D (16 cores, 32 threads), 64.9 GB |
| OS | Ubuntu 24.04.2 LTS, kernel 7.0.0-28-generic, Docker 29.5.0 |
| Engine | vLLM 0.23.0 (`vllm/vllm-openai:v0.23.0`, `cloudriftai/vllm-emmy:0.23.0-76ff82c21`) |
| Model | `google/gemma-4-12B-it` at `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, FP16 |
| Run window | 2026-09-22 13:25 to 20:05 UTC |

### Archive

`results_rtx5090x1.tar.gz` (Git LFS) holds `2026-09-22_13-25-38/` exactly as `emmy bench` wrote it: the run logs
`benchmark.log` and `benchmark_rtx5090_x_1.log`, and three files per row — an `.experiment.yaml`, a `.benchmark.log`
and a `.server.log`. Row IDs, by point and lane (stock, emmy, fast-math) — 256/256 c=1: `c9e49861fb8c`,
`af1f924afb43`, `fa406a490dde`; 4096/4096 c=1: `a45c51be8eff`, `6b35bdaaaf59`, `dbe64ae9822c`; 4096/4096 c=4:
`f53020eecbc5`, `8b36549284d9`, `b3a5630bf265`; 4096/4096 c=8: `127cb77a17dc`, `71320a273565`, `ef0e0564dc08`;
8192/256 c=4: `f763c1dd0350`, `17d539c42127`, `5e831ca8e713`; 256/256 c=64: `c7dbaea21ffa`, `e10b538c22f2`,
`ec83bae58318`.
