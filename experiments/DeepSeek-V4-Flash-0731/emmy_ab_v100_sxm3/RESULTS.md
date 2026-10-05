# DeepSeek V4 Flash 0731 — the Emmy serving image against the 1Cat fork

An A/B of two images at one serving envelope, run per exact GPU platform. Each platform section below describes the
archive named in it and nothing else.

## NVIDIA Tesla V100 SXM3 32GB × 16 — `results_v100x16.tar.gz`

**Question.** How fast does the Emmy serving image serve DeepSeek V4 Flash 0731 on 16× V100 SXM3, compared with the
plain 1Cat fork it is built from, at the envelope the Emmy image was warmed at?

**Protocol.** `emmy bench experiments/DeepSeek-V4-Flash-0731/emmy_ab_v100_sxm3 --ssh <host>` against a pre-allocated
host. The arms differ only in their image:

- fork: `cloudriftai/1cat-vllm-deepseek-v4-flash-0731@sha256:276240257b224097876b5b6db8f0d32484dff6a6f168d6b03d6df188e5c65bc1`;
- Emmy: `cloudriftai/vllm-emmy-deepseek-v4-flash-0731:1.2.3-992de5c8` (image ID `52a11f6c297d`), built FROM that digest
  at commit `992de5c80` by the release workflow: golden gate passed inside the image (115 measured rows, all 10
  serving programs), warmed, baked, and verified (offline start, no new kernel or Triton compile, a pack hit on all 16
  workers). During the run the image was local to the host, so the pull failed and `emmy bench` used the local
  image. It was published unchanged on 2026-10-05, at digest
  `sha256:52a11f6c297d200d12f9c2262b9684792797b71aa509dff5bdd4ca94b2b9424b`.

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

**Why commit `992de5c80`.** It is `main` with the DeepSeek serving changes since the previous run (2026-10-03, image
`3eb58b19`): a routed layer's experts run in one runtime call instead of a Python loop over the experts (#1039), which
is where the prefill waited on the host; and the 4,096-token pre program reduces cooperatively (#1049).

**Run.** Timestamp `2026-10-04T21:06:55Z`, run ID `20261004T210655Z`, repository revision `992de5c80` with this
recipe's Emmy image tag edited (committed with this report). All 8 rows `succeeded`. Every request completed: 12 per
one-request row and 24 per concurrent row, 0 failed.

| Round | Arm | One request (row id) | 8 concurrent (row id) |
| --- | --- | --- | --- |
| 1, seed 731 | fork | `a143a1359425` | `178622fbeb7a` |
| 2, seed 731 | Emmy | `44b1564659c1` | `3c7bc565e648` |
| 3, seed 831 | Emmy | `3c787d0fb83b` | `0543202d7df3` |
| 4, seed 831 | fork | `8477cc2ebf83` | `5d1e314d7ee5` |

Archive members sit under `2026-10-04_21-06-55/`:

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
| one request, 2,048 in / 128 out | Mean TTFT | 3,772 ± 11 ms | 3,035 ± 7 ms | 0.80× |
| | Mean TPOT | 148.13 ± 0.20 ms | 119.39 ± 0.03 ms | 0.81× |
| | Output throughput | 5.67 ± 0.01 tok/s | 7.04 ± 0.01 tok/s | 1.24× |
| 8 concurrent, 1,024 in / 64 out | Output throughput | 21.46 ± 0.64 tok/s | 21.50 ± 0.26 tok/s | 1.00× |
| | Mean TTFT | 9,034 ± 271 ms | 8,220 ± 192 ms | 0.91× |
| | Mean TPOT | 232.8 ± 6.7 ms | 245.0 ± 4.9 ms | 1.05× |
| | Median ITL | 175.3 ± 3.5 ms | 190.98 ± 0.23 ms | 1.09× |

Start-up still differs. The fork's model load and warm-up takes 158–159 s, 26–29 s of it loading weights. Emmy's
takes 339–354 s, 205–225 s of it loading weights, as on 2026-10-03 (338–342 s and 197–207 s).

**Repeat variation.**

- One request: the fork's per-token time moves by 0.20 ms, Emmy's by 0.03 ms. The time to first token moves by 0.3%
  in the fork and 0.2% in Emmy.
- Seeds agree at one request: the fork's per-token time is 147.95 ms at seed 731 and 148.31 ms at seed 831; Emmy's is
  119.37 ms and 119.41 ms.
- 8 concurrent, Emmy: throughput spreads 1.2% (21.41, 21.69, 21.72 and 21.16, 21.25, 21.76 tok/s).
- 8 concurrent, fork: throughput spreads 3.0%, nearly all of it between its two rounds: 21.96, 22.17, 21.97 in round 1
  against 20.74, 21.13, 20.81 in round 4. Its mean time to first token moves the same way (8.79 s against 9.28 s),
  while its decode step does not slow down (median time between tokens 178.0 ms against 172.7 ms). So the difference
  is in its prefill. The fork has seed 731 only in round 1 and seed 831 only in round 4, so this run cannot tell a
  seed effect from drift. On 2026-10-03 its two rounds agreed (21.12 and 21.32 tok/s).
- Emmy's median time between tokens at 8 concurrent varies by 0.23 ms: its decode step is one captured graph.

**Comparison.** This is a direct comparison: same host, same flags, interleaved rounds.

- One request: Emmy reaches the first token in 0.80× the fork's time and takes 0.81× its time per output token, so it
  delivers 24% more output tokens per second.
- 8 concurrent: the two deliver the same throughput, 21.50 against 21.46 tok/s, a difference far inside either
  spread. Emmy trails in the seed-731 rounds (21.61 against 22.03) and leads in the seed-831 rounds (21.39 against
  20.89). Its mean time to first token is 9% shorter, in both seeds (8.27 against 8.79 s, 8.17 against 9.28 s). Each
  of its decode steps (median time between tokens) is 9% longer.

Against the previous run on this host (2026-10-03, image `3eb58b19`):

- Emmy's time to first token for one request went from 4.09 to 3.04 s;
- its throughput at 8 concurrent from 18.27 to 21.50 tok/s, and its mean time to first token there from 11.24 to
  8.22 s;
- its per-token times are 0.2–0.3% higher: 119.10 → 119.39 ms for one request, 190.35 → 190.98 ms between tokens at 8
  concurrent;
- the fork arm reproduced its numbers within its spread (3,768 → 3,772 ms to first token, 148.03 → 148.13 ms per
  token, 21.22 → 21.46 tok/s).

What each Emmy worker logs, read from the server logs in this archive:

- the 256-row expert program has no measured row under strict evidence, so that width rides the symbolic expert
  program;
- its 4,096-token post program takes 20.3 ms per layer, 10× its roofline floor; the pre program takes 0.85 ms per
  layer, 28× its floor (2.85 ms and 95× on 2026-10-03);
- one 17-token step fell to the symbolic path: the smoke-test request before the benchmark.

One request from an outside address (`GET /`, answered 404) reached the API server during the first Emmy row. That
row's numbers agree with the other Emmy round's.

What is left against the fork: the 8-row decode step, which runs each row's six expert picks as six single-row
launches; the 4,096-token post program; and start-up.

**Quality context (measured outside this recipe; not in this archive).** GSM8K was scored on the same host through
`scripts/run_lmeval_gate.py --chat` (200 questions, seed 0, this recipe's serving flags):

| Server | Strict match | Flexible extract |
| --- | ---: | ---: |
| Emmy at `992de5c80` (the release workflow's correctness boot of its base image) | 0.73 | 0.955 |
| Emmy at `3eb58b19f` (the same boot of the previous release, 2026-10-03) | 0.715 | 0.96 |
| Fork as shipped (2026-09-29) | 0.91 | 0.975 |
| Fork with its prefill square computed in float32 (2026-09-29) | 0.755 | 0.96 |

Between the two Emmy releases, flexible extract differs by one question of 200 and strict match by three.

The shipped fork's higher strict match comes from a bug in its prefill mHC prenorm kernels:

- They square fp16 inputs in fp16, which overflows to infinity once a value reaches 256.
- The overflowing row's stream mixing then falls back to values derived from the bias alone.
- The first token's row crosses 256 at layer 11. So in every prefill longer than 16 tokens, the fork's first-token row
  departs from the causal result from layer 12 on.

With the square in float32, the fork's prompt likelihood equals Emmy's (a difference of +0.0006 nats per token),
flexible extract ties at 0.96, and strict match differs mostly in answer format. Emmy computes these rows correctly
and should not copy the overflow.

**Conclusion.** At this envelope the Emmy serving image serves a single request faster than the fork on both
measures: 0.80× its time to first token and 0.81× its time per output token, 24% more tokens per second. At 8
concurrent requests it matches the fork's throughput, reaches the first token 9% sooner, and runs each decode step 9%
slower. It still takes about 2.2× as long to start. Its answers match a correct fork.

**Limitations.**

- This is one envelope (context 4,096) at two points, with random-token prompts.
- Every row is a fresh deployment, so the numbers are warmed but not sustained load.
- The tie at 8 concurrent holds to about 3%, the fork's spread. The fork's two rounds differ by 5% and this run cannot
  say why.
- The Emmy image was built from `992de5c80`, which is also the run's repository revision.
- The GSM8K and likelihood figures come from separate host runs, not from this archive. The `992de5c80` GSM8K score
  was measured on the release's base image before the warm and bake, which run the same code.
- The capture ladder and the fp8 tuning switch differ between the arms by design; they are part of each image, not
  controlled variables.
