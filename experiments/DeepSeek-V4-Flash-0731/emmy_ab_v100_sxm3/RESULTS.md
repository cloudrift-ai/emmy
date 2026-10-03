# DeepSeek V4 Flash 0731 — the Emmy serving image against the 1Cat fork

An A/B of two images at one serving envelope, run per exact GPU platform. Each platform section below describes the
archive named in it and nothing else.

## NVIDIA Tesla V100 SXM3 32GB × 16 — `results_v100x16.tar.gz`

**Question.** How fast does the Emmy serving image serve DeepSeek V4 Flash 0731 on 16× V100 SXM3, compared with the
plain 1Cat fork it is built from, at the envelope the Emmy image was warmed at?

**Protocol.** `emmy bench experiments/DeepSeek-V4-Flash-0731/emmy_ab_v100_sxm3 --ssh <host>` against a pre-allocated
host. The arms differ only in their image:

- fork: `cloudriftai/1cat-vllm-deepseek-v4-flash-0731@sha256:276240257b224097876b5b6db8f0d32484dff6a6f168d6b03d6df188e5c65bc1`;
- Emmy: `cloudriftai/vllm-emmy-deepseek-v4-flash-0731:1.2.3-3eb58b19` (image ID `597855398e81`), built FROM that digest
  at commit `3eb58b19f` by the release workflow: golden gate passed inside the image (115 measured rows, all 9 serving
  programs), warmed, baked, and verified (offline start, no new kernel or Triton compile, a pack hit on all 16
  workers). The image is local to the host and not published, so the pull fails and `emmy bench` uses the local image.

Both arms get the same flags and environment: TP8 × PP2, fp16, fp8 KV cache, block size 256, context 4,096, 4,112
batched tokens, prefix caching off, `gpu_memory_utilization` 0.90, at most 8 sequences. Each image keeps its own
execution mode and switches, which are part of what is measured:

- the fork captures CUDA graphs at its default sizes and keeps its fp8 small-shape kernel tuning at warm-up;
- the Emmy image captures decode at sizes 1, 2, 4 and 8 (its ladder runs to 16; the 8-sequence cap stops it at 8), and
  its baked environment turns the fork runtime's fp8 small-shape tuning off, which makes its results identical from
  boot to boot.

There are two zipped points, with greedy decoding and `ignore_eos`:

- one request at a time: 4 prompts of 2,048 random tokens, 128 output tokens, 2 warm-up requests;
- 8 concurrent requests: 8 prompts of 1,024 random tokens, 64 output tokens, 8 warm-up requests.

The rounds run fork, Emmy, Emmy, fork, with seed 731 for the first two and 831 for the last two. Every row is a fresh
deployment, and the client runs 3 repeats against it. So each arm has 6 repeats per point, 3 per seed. Spreads below
are the sample standard deviation over those 6.

**Why commit `3eb58b19f`.** It is `main` with the DeepSeek serving changes since the previous run (2026-09-29, image
`4781e138`): no stream drain per expert weight swap and no device queries in the routed decode (#1002); every routed
expert sliced across the tensor-parallel ranks, with single-token decode captured (#1006); each program's runtime
layout kept per environment, which removed most of the prefill's host time, and the fp8 tuning switch (#1014); and
decode batches captured up to the decode bucket (#1020).

**Run.** Timestamp `2026-10-03T02:29:44Z`, run ID `20261003T022944Z`, repository revision `0ab9c3e1` with this
recipe's Emmy image tag and protocol note edited (committed with this report). All 8 rows `succeeded`. Every request
completed: 12 per one-request row and 24 per concurrent row, 0 failed.

| Round | Arm | One request (row id) | 8 concurrent (row id) |
| --- | --- | --- | --- |
| 1, seed 731 | fork | `a143a1359425` | `178622fbeb7a` |
| 2, seed 731 | Emmy | `6ead48f02315` | `d6fe8a219da6` |
| 3, seed 831 | Emmy | `a79e8e2f6173` | `fca12f0e99bf` |
| 4, seed 831 | fork | `8477cc2ebf83` | `5d1e314d7ee5` |

Archive members sit under `2026-10-03_02-29-44/`:

- `benchmark.log` and `benchmark_v100_x_16.log`;
- per row, `<variant>_<row id>.{experiment.yaml,benchmark.log,server.log}`, 24 files.

The variant names follow `v100x16_mc{1,8}_np{4,8}_nw{2,8}_ril{2048,1024}_rol{128,64}_s{731,831}_ic-<image>`. In the
archive copy the host address is replaced with `v100-host`.

**Machine and software.** Ubuntu 24.04.1, kernel 6.8.0-124-generic, Intel Xeon Platinum 8168 (80 logical CPUs),
1,338 GiB RAM, 16× Tesla V100-SXM3-32GB behind NVSwitches, driver 580.159.03, nvcc 12.9.86. Both arms run vLLM
`1.2.3.dev87+gd76126608`. Model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

**Result.**

| Point | Metric | Fork | Emmy | Emmy / fork |
| --- | --- | ---: | ---: | ---: |
| one request, 2,048 in / 128 out | Mean TTFT | 3,768 ± 10 ms | 4,091 ± 58 ms | 1.09× |
| | Mean TPOT | 148.03 ± 0.08 ms | 119.10 ± 0.04 ms | 0.80× |
| | Output throughput | 5.67 ± 0.00 tok/s | 6.66 ± 0.02 tok/s | 1.17× |
| 8 concurrent, 1,024 in / 64 out | Output throughput | 21.22 ± 0.82 tok/s | 18.27 ± 0.54 tok/s | 0.86× |
| | Mean TTFT | 9,078 ± 320 ms | 11,242 ± 391 ms | 1.24× |
| | Mean TPOT | 236.8 ± 8.9 ms | 264.4 ± 7.2 ms | 1.12× |
| | Median ITL | 175.2 ± 3.5 ms | 190.35 ± 0.07 ms | 1.09× |

Start-up differs too. The fork's model load and warm-up takes about 159 s, 26–28 s of it loading weights (262 s and 56
s in the first row, the run's first deployment). Emmy's takes 338–342 s, 197–207 s of it loading weights, against
432–446 s and 296–300 s on 2026-09-29.

**Repeat variation.** Both arms are stable, and the result does not depend on round order.

- One request: the fork's per-token time moves by 0.08 ms, Emmy's by 0.04 ms. Emmy's time to first token moves by 1.4%.
- Seeds agree: the fork's per-token time is 147.95 ms at seed 731 and 148.10 ms at seed 831; Emmy's is 119.06 ms and
  119.13 ms.
- 8 concurrent: throughput spreads about 4% in the fork and 3% in Emmy. The fork's first and last rounds agree (20.69,
  21.99, 20.68 against 20.76, 22.53, 20.67 tok/s), so the host did not drift over the run. Emmy's two rounds give
  17.80, 18.92, 17.68 and 18.04, 18.25, 18.90.
- Emmy's median time between tokens at 8 concurrent varies by 0.07 ms: its decode step is one captured graph.

**Comparison.** This is a direct comparison: same host, same flags, interleaved rounds.

- One request: Emmy takes 0.80× the fork's time per output token, so it delivers 17% more output tokens per second;
  it reaches the first token 9% later.
- 8 concurrent: Emmy delivers 86% of the fork's throughput, with each decode step (median time between tokens) 9%
  slower and the mean time to first token 24% later.

Against the previous run on this host (2026-09-29, image `4781e138`): per output token for one request 316.7 → 119.1 ms,
time to first token 11.07 → 4.09 s, throughput at 8 concurrent 6.82 → 18.27 tok/s; the fork arm reproduced its numbers
within its spread.

What each Emmy worker logs, read from the server logs in this archive:

- the 256-row expert program has no measured row under strict evidence, so that width rides the symbolic expert
  program;
- its 4,096-token prefill programs are far off their roofline floor: post 20.4 ms per layer, 10× the floor; pre 2.85
  ms per layer, 95× the floor;
- one 17-token step fell to the symbolic path: the smoke-test request before the benchmark.

The prefill programs and the 8-row decode step, which runs each row's six expert picks as six single-row launches, are
where the remaining gap at 8 concurrent sits.

**Quality context (measured outside this recipe; not in this archive).** GSM8K was scored on the same host through
`scripts/run_lmeval_gate.py --chat` (200 questions, seed 0, this recipe's serving flags):

| Server | Strict match | Flexible extract |
| --- | ---: | ---: |
| Emmy at `3eb58b19f` (the release workflow's correctness boot of its base image) | 0.715 | 0.96 |
| Emmy at `4781e138` with the #964 router fix (2026-09-29) | 0.71 | 0.96 |
| Fork as shipped (2026-09-29) | 0.91 | 0.975 |
| Fork with its prefill square computed in float32 (2026-09-29) | 0.755 | 0.96 |

The shipped fork's higher strict match comes from a bug in its prefill mHC prenorm kernels:

- They square fp16 inputs in fp16, which overflows to infinity once a value reaches 256.
- The overflowing row's stream mixing then falls back to values derived from the bias alone.
- The first token's row crosses 256 at layer 11. So in every prefill longer than 16 tokens, the fork's first-token row
  departs from the causal result from layer 12 on.

With the square in float32, the fork's prompt likelihood equals Emmy's (a difference of +0.0006 nats per token),
flexible extract ties at 0.96, and strict match differs mostly in answer format. Emmy computes these rows correctly
and should not copy the overflow.

**Conclusion.** At this envelope the Emmy serving image now serves a single request faster than the fork: 0.80× its
time per output token and 17% more tokens per second, at 1.09× its time to first token. At 8 concurrent requests it
delivers 86% of the fork's throughput and takes 1.24× as long to the first token. It still takes about 2.1× as long to
start. Its answers match a correct fork.

**Limitations.**

- This is one envelope (context 4,096) at two points, with random-token prompts.
- Every row is a fresh deployment, so the numbers are warmed but not sustained load.
- The Emmy image is unpublished. It was built from `3eb58b19f`; the run's repository revision `0ab9c3e1` differs from it
  only outside the image (the recipe and later `main` commits).
- The GSM8K and likelihood figures come from separate host runs, not from this archive. The `3eb58b19f` GSM8K score was
  measured on the release's base image before the warm and bake, which run the same code.
- The capture ladder and the fp8 tuning switch differ between the arms by design; they are part of each image, not
  controlled variables.
