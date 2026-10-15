# Qwen3.8-Flash-Next on the plain 1Cat fork, V100 SXM3: the baseline for Emmy

**Question.** How fast does the plain fork serve this model in the serving shape chosen for it on this node, so
that Emmy's kernels have a measured target to match?

**Shape.** RadixArk NVFP4 checkpoint at `7b719225242a`, TP4, MTP off, a 131,072-token context, the KV pool pinned at
2.5 GiB per card (194,016 tokens), a 4,096-token prefill step, up to eight sequences, prefix caching on, the n-gram
table in pinned host memory. It was chosen because it keeps 3 GiB per card free under the worst-case loads (one
full-context prompt, four 32K prompts at once, a 128K prefill among three decoding requests), the margin Emmy's
prefill is expected to need. The fork's own launcher defaults (an 8,192-token step, a 0.90 share, MTP on) leave less.

## v100x16 (16x NVIDIA Tesla V100 SXM3 32GB, rows on cards 0-3 or all sixteen)

**Protocol.** `emmy bench` with the recipe in this directory, one deployment per row, random prompts, greedy,
`ignore_eos`, four client repeats per row with fresh prompts (seed 380 + repeat). There is no warm-up request:
vllm bench's warm-up replays a measured prompt, which the prefix cache would then serve. Repeat 1 primes the server;
repeats 2-4 are the steady numbers below.

| Row | Time to first token | Per output token | Total output |
| --- | ---: | ---: | ---: |
| 1 request, 1,024-token prompt, 512 out | 0.49-0.54 s | 13.0-13.5 ms | |
| 1 request, 8,192 / 512 | 1.57 s | 13.0-13.1 ms | |
| 1 request, 32,768 / 512 | 6.12-6.22 s | 13.3-13.4 ms | |
| 1 request, 130,000 / 256 (the full context) | 25.9-26.2 s | 14.1-14.2 ms | |
| 4 requests, 4,096 / 1,024 | 2.54-2.55 s mean | 20.6-20.7 ms | 172.6-173.8 tok/s |
| 8 requests, 4,096 / 1,024 | 3.45-3.55 s mean | 25.2-25.3 ms | 278.2-279.8 tok/s |
| 4 replicas on 16 cards, 32 requests through the load balancer, 4,096 / 1,024 | 7.9-11.9 s mean, 2.8-4.0 s median | 26.3-26.4 ms | 775-811 tok/s |

**Repeat variation.** Repeats 2-4 agree within 1.5% on per-token time and within 1% on throughput, except the
four-replica row, whose throughput spans 775-811 tok/s. Repeat 1 is slower everywhere (first-token mean 1.1 s at
1K, 4.6 s at eight requests), as the fork compiles kernels for each new batch size on first use.

**Comparisons.**

- One request decodes at 74-77 tok/s; upstream reports 80.7 tok/s for this checkpoint without MTP on four SXM2
  cards with direct NVLink, at an 8K/512 shape (`upstream` numbers are 1Cat-vLLM's own reports).
- Prefill: about 5,200 tokens/s at 8K and 32K and 5,000 at the full context; upstream reports 7,056 at 8K. The
  4,096-token step this shape uses leaves single-request first tokens unchanged against the launcher's 8,192 (1.57 s
  against 1.50 s at 8K in a screen on the same node).
- Four replicas: per-token time stays at one replica's (26.3 against 25.3 ms), but the node gives 72% of four
  isolated replicas (793 against 4 x 279 tok/s). The load balancer spreads requests evenly overall (121-139 of 512
  per replica), and the loss shows as waiting: the mean first token is two to three times the median, as a replica
  that is handed a ninth request queues it behind its eight sequences. That reading is not measured directly.

**Per-phase profile.** Not part of the recipe. A torch-profiler trace of the same configuration on the same node
(one request, rank 0, a 1,024-token prompt and 64 decode steps, then one 4,096-token prompt) gives about 11 ms of GPU
work per decode step (10.9-11.3 ms on the four ranks), so roughly 2 ms of the 13 ms token is host time. Of the GPU
time per decode step: dense FP16 projections about 3.7 ms, exchanges between ranks (the fast all-reduce and the
fused hyper-connection gathers) about 2.7 ms, MoE (NVFP4 experts, router, shared expert) about 1.9 ms, hyper-
connection local work about 0.9 ms, sparse attention about 0.8 ms, Gated DeltaNet about 0.5 ms. A 3,920-token
prefill chunk takes about 0.6 s, led by dense GEMMs and the NVFP4 expert GEMMs, then attention, the hyper-connection
mixing, the NCCL ring all-reduce (prefill does not take the fast path) and Gated DeltaNet.

**Conclusion.** These rows are the numbers Emmy's serving image has to match or beat in this shape: 13.0-13.4 ms per
token at one request up to 32K tokens of context, 1.57 s / 6.2 s / 26 s to the first token at 8K / 32K / the full
context, and 279 tok/s at eight requests per replica.

**Limitations.** Random prompts only, so the prefix cache contributes nothing and output length is fixed; output
quality is not measured here. The image is the upstream v1.5.1 release wheel built locally on the node
(`local/1cat-vllm:1.5.1-wheel`), not a published image. The four-replica row measures the node through Emmy's nginx
load balancer with one client, not four independent clients. The test node's address is replaced with `test-node` in
the archived raw files; nothing else in them was changed.

**Run.** 2026-10-15T00:17:09Z, run ID `20261015T001709Z`, Emmy revision `454db8758`, 7 of 7 rows `succeeded`, no
failed requests. Host: 16x Tesla V100-SXM3-32GB, driver 580.178.04, NVLink on (NV6 between all pairs), Ubuntu 24.04.1,
kernel 6.8.0-139, 80 Xeon Platinum 8168 vCPUs, one NUMA node, 1.34 TB RAM. Image: 1Cat-vLLM v1.5.1 (tag commit
`589a5d1e1a9e`, wheel sha256 `757f2200ba6b…`), torch 2.10.0+cu128, triton 3.6.0, on
`nvidia/cuda:12.8.1-devel-ubuntu24.04` with `python3.12-dev`. Each deployment took 560-637 s (a cold start: the fork
compiles at boot).

**Archive.** `results_v100x16.tar.gz`, root `2026-10-15_00-17-09/`: seven `*.experiment.yaml` records (rows
`v100x4_mc1_np4_ril1024_rol512`, `v100x4_mc1_np4_ril8192_rol512`, `v100x4_mc1_np3_ril32768_rol512`,
`v100x4_mc1_np2_ril130000_rol256`, `v100x4_mc4_np16_ril4096_rol1024`, `v100x4_mc8_np32_ril4096_rol1024`,
`v100x16_mc32_np128_ril4096_rol1024`), their `*.benchmark.log` and `*.server.log`, and the run logs.
