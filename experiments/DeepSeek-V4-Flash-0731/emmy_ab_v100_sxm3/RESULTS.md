# DeepSeek V4 Flash 0731 — the Emmy serving image against the 1Cat fork

An A/B of two images at one serving envelope, run per exact GPU platform. Each platform section below describes the
archive named in it and nothing else.

## NVIDIA Tesla V100 SXM3 32GB × 16 — `results_v100x16.tar.gz`

**Question.** How fast does the Emmy serving image serve DeepSeek V4 Flash 0731 on 16× V100 SXM3, compared with the
plain 1Cat fork it is built from, at the envelope the Emmy image was warmed at?

**Protocol.** `emmy bench experiments/DeepSeek-V4-Flash-0731/emmy_ab_v100_sxm3` against a pre-allocated host over SSH.
The arms differ only in their image:

- fork: `cloudriftai/1cat-vllm-deepseek-v4-flash-0731@sha256:276240257b224097876b5b6db8f0d32484dff6a6f168d6b03d6df188e5c65bc1`;
- Emmy: `cloudriftai/vllm-emmy-deepseek-v4-flash-0731:1.2.3-4781e138`, built FROM that digest at commit `4781e138`. The
  image is local to the host and not published, so the pull fails and `emmy bench` uses the local image.

Both arms get the same flags and environment: TP8 × PP2, fp16, fp8 KV cache, block size 256, context 4,096, 4,112
batched tokens, prefix caching off, `gpu_memory_utilization` 0.90. The fork captures CUDA graphs. Emmy runs eager.

There are two zipped points, with greedy decoding and `ignore_eos`:

- one request at a time: 4 prompts of 2,048 random tokens, 128 output tokens, 2 warm-up requests;
- 8 concurrent requests: 8 prompts of 1,024 random tokens, 64 output tokens, 8 warm-up requests.

The rounds run fork, Emmy, Emmy, fork, with seed 731 for the first two and 831 for the last two. Every row is a fresh
deployment, and the client runs 3 repeats against it. So each arm has 6 repeats per point, 3 per seed. Spreads below
are the sample standard deviation over those 6.

**Why commit `4781e138`.** It is the merge of #964, which keeps the router's expert-selection bias in float32. It also
predates #969. Images built from #969 on fail the golden gate and miscompile the Sinkhorn step-0 seed read of the stream
mixing. PR #978 fixes both and was still open when this ran.

**Run.** Timestamp `2026-09-29T21:58:00Z`, run ID `20260929T215800Z`, repository revision `3ee0db80` (clean tree).
All 8 rows `succeeded`. Every request completed: 12 per one-request row and 24 per concurrent row, 0 failed.

| Round | Arm | One request (row id) | 8 concurrent (row id) |
| --- | --- | --- | --- |
| 1, seed 731 | fork | `a143a1359425` | `178622fbeb7a` |
| 2, seed 731 | Emmy | `f52b77414ceb` | `e8cd752b2a4a` |
| 3, seed 831 | Emmy | `fa6018379c88` | `4fcf87c364f0` |
| 4, seed 831 | fork | `8477cc2ebf83` | `5d1e314d7ee5` |

Archive members sit under `2026-09-29_21-58-00/`:

- `benchmark.log` and `benchmark_v100_x_16.log`;
- per row, `<variant>_<row id>.{experiment.yaml,benchmark.log,server.log}`, 24 files.

The variant names follow `v100x16_mc{1,8}_np{4,8}_nw{2,8}_ril{2048,1024}_rol{128,64}_s{731,831}_ic-<image>`. In the
archive copy the host address is replaced with `v100-host`.

An earlier invocation on 2026-09-28 lost its Emmy rows to the bench client's tokenizer lookup inside the offline
image. That lookup is fixed in the same PR, and the earlier run is not archived.

**Machine and software.** Ubuntu 24.04.1, kernel 6.8.0-124-generic, Intel Xeon Platinum 8168 (80 logical CPUs),
1,338 GiB RAM, 16× Tesla V100-SXM3-32GB behind NVSwitches, driver 580.159.03, nvcc 12.9.86. Both arms run vLLM
`1.2.3.dev87+gd76126608`. Model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

**Result.**

| Point | Metric | Fork | Emmy | Emmy / fork |
| --- | --- | ---: | ---: | ---: |
| one request, 2,048 in / 128 out | Mean TTFT | 3,767 ± 14 ms | 11,074 ± 28 ms | 2.94× |
| | Mean TPOT | 148.02 ± 0.09 ms | 316.7 ± 2.1 ms | 2.14× |
| | Output throughput | 5.67 ± 0.00 tok/s | 2.50 ± 0.01 tok/s | 0.44× |
| 8 concurrent, 1,024 in / 64 out | Output throughput | 20.97 ± 0.68 tok/s | 6.82 ± 0.20 tok/s | 0.33× |
| | Mean TTFT | 9,208 ± 186 ms | 29,105 ± 757 ms | 3.16× |
| | Mean TPOT | 238.8 ± 8.9 ms | 718.7 ± 22.5 ms | 3.01× |
| | Median ITL | 174.3 ± 3.7 ms | 513.3 ± 19.8 ms | 2.95× |

Start-up differs too. The fork's model load and warm-up takes 155 s, 26 s of it loading weights. Emmy's takes
432–446 s, 296–300 s of it loading weights.

**Repeat variation.** Both arms are stable, and the result does not depend on round order.

- One request: the fork's per-token time moves by 0.1 ms. Emmy's moves by under 1%.
- Seeds agree: the fork's per-token time is 147.94 ms at seed 731 and 148.10 ms at seed 831; Emmy's is 317.0 ms and
  316.3 ms.
- 8 concurrent: the spread is about 3% in both arms.
- The fork's first and last rounds agree (20.67 vs 21.28 tok/s), so the host did not drift over the three hours.
- The median TTFT at 8 concurrent swings more (±14–15%) because the 8 requests queue behind each other's prefill. The
  mean is the stable figure.

**Comparison.** This is a direct comparison: same host, same flags, interleaved rounds.

- One request: Emmy takes 2.1× the fork's time per output token and 2.9× its time to first token.
- 8 concurrent: Emmy delivers a third of the fork's throughput.

Two differences are part of what is measured, not controlled for:

- The fork replays CUDA graphs. Emmy runs eager because its hyper-connection MoE syncs with the host every decode step.
- The Emmy image's M=1 expert tier is off, as in the verified release. At start-up every worker logs that its
  single-row and 256-row expert programs have no measured row under strict evidence and ride a wider tier.

Every Emmy worker also logs its 4,096-token prefill programs as far off their floor:

- post: 20.4 ms per layer, 10× the floor;
- pre: 3.1 ms per layer, 104× the floor.

That fits the prefill gap. Each worker also logs that one 17-token step fell to the symbolic path. That step is the
smoke-test request before the benchmark. The benchmark's own steps are either 8 rows or fewer, or long prefill chunks.

**Quality context (measured outside this recipe; not in this archive).** GSM8K was scored on the same host through
`scripts/run_lmeval_gate.py --chat` (200 questions, seed 0, this recipe's serving flags):

| Server | Strict match | Flexible extract |
| --- | ---: | ---: |
| Fork as shipped | 0.91 | 0.975 |
| Emmy with the #964 router fix | 0.71 | 0.96 |
| Fork with its prefill square computed in float32 | 0.755 | 0.96 |

The shipped fork's higher strict match comes from a bug in its prefill mHC prenorm kernels:

- They square fp16 inputs in fp16, which overflows to infinity once a value reaches 256.
- The overflowing row's stream mixing then falls back to values derived from the bias alone.
- The first token's row crosses 256 at layer 11. So in every prefill longer than 16 tokens, the fork's first-token row
  departs from the causal result from layer 12 on.

With the square in float32:

- the fork's prompt likelihood equals Emmy's, a difference of +0.0006 nats per token;
- flexible extract ties at 0.96;
- strict match differs on 14 fork-only against 5 Emmy-only questions, mostly a choice of answer format.

Emmy computes these rows correctly and should not copy the overflow.

**Conclusion.** At this envelope the Emmy serving image is 2.1× slower than the fork per output token for a single
request. At 8 concurrent requests its throughput is a third of the fork's. It also takes about 3× as long to the first
token and about 2.8× as long to start. Its answers match a correct fork. The shipped fork's quality edge is its own
overflow bug.

**Limitations.**

- This is one envelope (context 4,096) at two points, with random-token prompts.
- Every row is a fresh deployment, so the numbers are warmed but not sustained load.
- The Emmy image is unpublished and was built from `4781e138`, not from current main.
- The GSM8K and likelihood figures come from separate host runs, not from this archive. The Emmy GSM8K lane ran an
  image with the router fix applied, not this exact release image.
- Eager execution against CUDA graphs is inherent to the two images, not a controlled variable.
