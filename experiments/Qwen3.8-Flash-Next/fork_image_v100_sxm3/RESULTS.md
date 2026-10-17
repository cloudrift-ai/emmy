# Qwen3.8-Flash-Next on CloudRift's own 1Cat image, V100 SXM3: the step 2 baseline reproduced

**Question.** Does the 1Cat image built from CloudRift's fork branch serve this model as fast as the plain upstream
v1.5.1 release did in `fork_baseline_v100_sxm3`, within the spread of repeated runs? Only then can it replace the
release wheel as the fork that Emmy's serving image builds on.

**Image.** `cloudriftai/1cat-vllm-qwen3.8-flash-next:1.5.1-248d877c5`, built from
`docker/1cat-vllm-sm70/Dockerfile.v1.5.1` at the fork's `cloudrift/v1.5.1` branch (`248d877c549c`): upstream v1.5.1
(`589a5d1e1a9e`) plus the Poolside tool-call fix and the Laguna FP8 precision fix, with Transformers 5.19.0. The
branch also reverts the non-MTP Gated DeltaNet full-forward guard that CloudRift's older branch carried, which
upstream does not use.

## v100x16 (16x NVIDIA Tesla V100 SXM3 32GB, rows on cards 0-3 or all sixteen)

**Protocol.** Step 2's, unchanged: `emmy bench` with the recipe in this directory, the same rows, flags and seeds, one
deployment per row, random prompts, greedy, `ignore_eos`, four client repeats per row with fresh prompts (seed 380 +
repeat), and no warm-up request. Repeat 1 primes the server; repeats 2-4 are the steady numbers below, beside step 2's.

| Row | Time to first token | | Per output token | | Total output | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | this image | step 2 | this image | step 2 | this image | step 2 |
| 1 request, 1,024-token prompt, 512 out | 0.46-0.47 s | 0.49-0.54 s | 12.7-12.8 ms | 13.0-13.5 ms | | |
| 1 request, 8,192 / 512 | 1.56-1.57 s | 1.57 s | 12.9-13.0 ms | 13.0-13.1 ms | | |
| 1 request, 32,768 / 512 | 6.15-6.23 s | 6.12-6.22 s | 13.2 ms | 13.3-13.4 ms | | |
| 1 request, 130,000 / 256 (the full context) | 25.7-26.1 s | 25.9-26.2 s | 14.1-14.2 ms | 14.1-14.2 ms | | |
| 4 requests, 4,096 / 1,024 | 2.53-2.55 s mean | 2.54-2.55 s mean | 20.4-20.5 ms | 20.6-20.7 ms | 174.4-175.0 tok/s | 172.6-173.8 tok/s |
| 8 requests, 4,096 / 1,024 | 3.46-3.47 s mean | 3.45-3.55 s mean | 25.2-26.0 ms | 25.2-25.3 ms | 271.9-280.3 tok/s | 278.2-279.8 tok/s |
| 4 replicas on 16 cards, 32 requests through the load balancer | 7.5-8.3 s mean, 2.7-3.4 s median | 7.9-11.9 s mean, 2.8-4.0 s median | 26.3-26.8 ms | 26.3-26.4 ms | 808-863 tok/s | 775-811 tok/s |

**Repeat variation.** Repeats 2-4 agree within 1% on per-token time and throughput in every single-replica row,
except the eight-request row's repeat 4 (26.0 ms, 271.9 tok/s against 25.2-25.4 ms and 278-280 tok/s for repeats 2-3).
The four-replica row is the noisy one, as in step 2: its throughput spans 808-863 tok/s and its first-token median
2.7-3.4 s. Repeat 1 is slower everywhere, as in step 2 (first-token mean 1.1 s at 1K, 4.6 s at eight requests),
since the fork compiles kernels for each new batch size on first use.

**Comparisons.**

- Against step 2, row by row on the medians of repeats 2-4: per-token time and throughput are within 1% in every row
  except 1K and, for throughput, the four-replica row. Time to first token is within 1% from 8K to the full context
  and at four and eight requests.
- The 1K row is faster on this image: 0.46 s against 0.49-0.54 s to the first token, and 12.7 against 13.0-13.5 ms per
  token. Step 2's 1K row was its least steady single-request row (10.6% spread on the first token); the cause of the
  difference is not established.
- The four-replica row gives 1.09x step 2's throughput (median 858 against 789 tok/s). That row varies most between
  repeats (775-811 tok/s in step 2, 808-863 here), so this is not read as a real gain.
- A first run of this protocol on an earlier build of the branch, which still carried the Gated DeltaNet guard, gave
  first tokens 3-15% slower than step 2. An A/B of three servers at the same time on the same node traced the gap to
  the guard. With it on, first tokens took 610 ms / 1.75 s / 6.63 s at 1K / 8K / 32K; with it switched off, 480 ms /
  1.59 s / 6.40 s; the release wheel took 474 ms / 1.60 s / 6.26 s. That evidence is not archived here; this archive
  holds only the run without the guard.

**Conclusion.** The image reproduces step 2's numbers: decode, prefill and concurrent throughput all match within the
spread of repeated runs, and the 1K row is slightly faster. It can replace the release wheel as the fork that Emmy's
serving image builds on.

**Limitations.** Speed only. This run does not check output quality, offline boot, or whether a boot compiles
anything new. Every deployment here was a cold start: 553-559 s per four-card row and 638 s for the node, in line with
step 2's 560-637 s. The image is tagged locally on the test node and not published. The protocol limits are step 2's:
random prompts only, so the prefix cache contributes nothing, output length is fixed, and the four-replica row
measures the node through Emmy's nginx load balancer with one client. In the archived raw files, the test node's
address is replaced with `test-node` and the local run directory with its repository path; nothing else in them was
changed.

**Run.** 2026-10-16T07:09:52Z, run ID `20261016T070952Z`, Emmy revision `42d7c3d19`, 7 of 7 rows `succeeded`, no
failed requests. Host: the same VM as step 2, 16x Tesla V100-SXM3-32GB, driver 580.178.04, Ubuntu 24.04.1,
kernel 6.8.0-139, 80 Xeon Platinum 8168 vCPUs, one NUMA node. Image ID `sha256:09bf5dfc0b46…`: vLLM 1.5.1, torch
2.10.0+cu128, triton 3.6.0, Transformers 5.19.0, on `nvidia/cuda:12.8.1-devel-ubuntu24.04`.

**Archive.** `results_v100x16.tar.gz`, root `2026-10-16_07-09-52/`: seven `*.experiment.yaml` records (rows
`v100x4_mc1_np4_ril1024_rol512`, `v100x4_mc1_np4_ril8192_rol512`, `v100x4_mc1_np3_ril32768_rol512`,
`v100x4_mc1_np2_ril130000_rol256`, `v100x4_mc4_np16_ril4096_rol1024`, `v100x4_mc8_np32_ril4096_rol1024`,
`v100x16_mc32_np128_ril4096_rol1024`), their `*.benchmark.log` and `*.server.log`, and the run logs.
