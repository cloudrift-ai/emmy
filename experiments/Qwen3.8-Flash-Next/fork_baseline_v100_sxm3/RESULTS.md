# Qwen3.8-Flash-Next on the plain 1Cat fork, V100 SXM3: the baseline for Emmy

**Question.** How fast does the plain fork serve this model in the serving shape chosen for it on this node, so
that Emmy's kernels have a measured target to match?

**Shape.** RadixArk NVFP4 checkpoint at `7b719225242a`, TP4, MTP off, a 131,072-token context, the KV pool pinned at
2.5 GiB per card (194,016 tokens), a 4,096-token prefill step, up to eight sequences, prefix caching on, the n-gram
table in pinned host memory. It was chosen because it keeps 3 GiB per card free under the worst-case loads (one
full-context prompt, four 32K prompts at once, a 128K prefill among three decoding requests), the margin Emmy's
prefill is expected to need. The fork's own launcher defaults (an 8,192-token step, a 0.90 share, MTP on) leave less.
Since 2026-11-11 the shape runs with vLLM's async scheduling on, its default with MTP off: the host prepares the next
step while the GPU runs the current one.

## v100x16 (16x NVIDIA Tesla V100 SXM3 32GB, rows on cards 0-3 or all sixteen)

**Image.** `cloudriftai/1cat-vllm-qwen3.8-flash-next-nvfp4@sha256:bf976828fc00…`, CloudRift's v1.5.1 fork image
with its kernel caches baked in (`fork_serving_v100_sxm3`), the image Emmy's serving image builds on.

**Protocol.** `emmy bench` with the recipe in this directory, one deployment per row, random prompts, greedy,
`ignore_eos`, four client repeats per row with fresh prompts (seed 380 + repeat). There is no warm-up request:
vllm bench's warm-up replays a measured prompt, which the prefix cache would then serve. Repeat 1 primes the server;
repeats 2-4 are the steady numbers below. The last column is this experiment's first run (2026-10-15), with async
scheduling off on the upstream release wheel.

| Row | Time to first token | Per output token | Total output | First run, async off: first token / per token / total |
| --- | ---: | ---: | ---: | ---: |
| 1 request, 1,024-token prompt, 512 out | 0.47-0.52 s | 9.59 ms | | 0.49-0.54 s / 13.0-13.5 ms |
| 1 request, 8,192 / 512 | 1.59-1.60 s | 9.72-9.74 ms | | 1.57 s / 13.0-13.1 ms |
| 1 request, 32,768 / 512 | 6.13-6.21 s | 10.0 ms | | 6.12-6.22 s / 13.3-13.4 ms |
| 1 request, 130,000 / 256 (the full context) | 46.3-46.6 s | 10.9 ms | | 25.9-26.2 s / 14.1-14.2 ms |
| 4 requests, 4,096 / 1,024 | 2.51-2.53 s mean | 17.2-17.3 ms | 202.9-203.1 tok/s | 2.54-2.55 s / 20.6-20.7 ms / 172.6-173.8 tok/s |
| 8 requests, 4,096 / 1,024 | 3.73-3.90 s mean | 21.3-21.6 ms | 317.7-318.2 tok/s | 3.45-3.55 s / 25.2-25.3 ms / 278.2-279.8 tok/s |
| 4 replicas on 16 cards, 32 requests through the load balancer, 4,096 / 1,024 | 7.4-8.8 s mean, 3.3-3.4 s median | 22.1-22.4 ms | 897-1,008 tok/s | 7.9-11.9 s mean, 2.8-4.0 s median / 26.3-26.4 ms / 775-811 tok/s |

**Repeat variation.** Repeats 2-4 agree within 0.5% on per-token time and throughput in every single-replica row,
and within 1.5% on first tokens except at 1K (0.47-0.52 s) and the eight-request mean (3.73-3.90 s). The
four-replica row is the noisy one, as in the first run: 897-1,008 tok/s. Repeat 1 is no longer slower than the rest
(0.51 s at 1K, 3.71 s mean at eight requests): this image carries the kernels the release wheel compiled on the first
requests.

**Comparisons.** Two things changed against the first run, the image and async scheduling, and
`fork_image_v100_sxm3` measured the two images equal within the spread, so the differences are read as async's.

- Decode: one request takes 9.6-10.9 ms per token against 13.0-14.2, 3.2-3.9 ms less per step at every context
  length. Total output is 17% higher at four requests, 14% at eight, and 27% on the four-replica node (medians 1,005
  against 789 tok/s, the noisiest row).
- First token: unchanged up to 32K and at four requests; at eight requests the mean is 0.3 s later.
- The full context: 46.3-46.6 s against 25.9-26.2 s. In a separate A/B on the same node (not in this archive) the
  engine preempted each 130K prompt twice during its prefill and computed it again from the start, while 64K and 96K
  prompts took the same time in both modes with no preemption; with the KV pool raised to 3 GiB per card (232,523
  tokens) the async fork reached the first token in 26.1 s with no preemption. So async scheduling needs a larger pool
  than synchronous scheduling for a full-context prompt in this shape; why is not established.
- Deployment: 430-442 s to load and warm per four-card row (484 s for the node), of it 85-104 s compiling graphs.
  The image's baked caches were warmed with async off, and the flag is part of vLLM's compile key, so they miss;
  with them hitting, `fork_serving_v100_sxm3` booted in 303 s.

**Conclusion.** These rows are the numbers Emmy's serving image has to match or beat in the production shape since
2026-11-11: 9.6-10.9 ms per token at one request, 0.47 / 1.6 / 6.2 / 46 s to the first token at 1K / 8K / 32K / the
full context, and 203 and 318 tok/s at four and eight requests per replica.

**Limitations.** Random prompts only, so the prefix cache contributes nothing and output length is fixed; output
quality and memory against the 3 GiB margin are not measured here. The four-replica row measures the node through
Emmy's nginx load balancer with one client, not four independent clients. The first run's numbers are kept above as
history; its archive was replaced by this run's (run ID `20261015T001709Z`, Emmy `454db8758`, image
`local/1cat-vllm:1.5.1-wheel`, the upstream release wheel built on the node). In the archived raw files the test
node's address is replaced with `test-node`, and local paths with repository paths; nothing else in them was changed.

**Run.** 2026-11-11T20:55:13Z, run ID `20261111T205513Z`, Emmy revision `786969569` (the checkout also held plan
edits outside the harness), 7 of 7 rows `succeeded`, no failed requests. Host: the same VM as the first run, 16x Tesla
V100-SXM3-32GB, driver 580.178.04, NVLink on, Ubuntu 24.04.1, kernel 6.8.0-139, 80 Xeon Platinum 8168 vCPUs, CUDA
12.9 on the host. Image: 1Cat-vLLM v1.5.1 at the fork's `cloudrift/v1.5.1` branch (`248d877c5`), torch
2.10.0+cu128, triton 3.6.0, Transformers 5.19.0.

**Archive.** `results_v100x16.tar.gz`, root `2026-11-11_20-55-13/`: seven `*.experiment.yaml` records (rows
`v100x4_mc1_np4_ril1024_rol512`, `v100x4_mc1_np4_ril8192_rol512`, `v100x4_mc1_np3_ril32768_rol512`,
`v100x4_mc1_np2_ril130000_rol256`, `v100x4_mc4_np16_ril4096_rol1024`, `v100x4_mc8_np32_ril4096_rol1024`,
`v100x16_mc32_np128_ril4096_rol1024`), their `*.benchmark.log` and `*.server.log`, and the run logs.
