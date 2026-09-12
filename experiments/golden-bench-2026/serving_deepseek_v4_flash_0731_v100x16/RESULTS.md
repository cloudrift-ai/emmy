# DeepSeek V4 Flash 0731 serving on 16× V100 SXM3

## Conclusion

Both arms now serve this checkpoint and both produced serving numbers. The fork is roughly **38× faster per output
token** and **23× faster to first token**. The two arms were measured in separate invocations rather than one
alternating run, so this is a directional comparison, not the balanced A/B described under "What this run does not
establish".

The Emmy arm completes requests only with its static M=1 decode tier disabled (`EMMY_GEN_M1_TIER=0`). With that
tier enabled — the default — no generation finished: a single-token decode forward advances at a flat ~61 s per
layer and takes roughly 2,700 s, which overruns both vLLM's 300 s `sample_tokens` deadline and the 600 s NCCL collective
watchdog. Routing single-token decode through the M=16 bucket twins instead costs 5.6 s per token, so the tier itself
carries a factor of about 485×. That is an Emmy defect, recorded below and not yet fixed.

Three programs were fixed earlier in this work — `pre4096` by 594×, and the decode program's two hot kernels by 4.3×
and 5.6× — all confirmed by the deployed boot's own audit. Those fixes hold, and they are why the arm serves at all.
The two programs that now dominate were never recorded: `post.decode.m16` at 1,149× its roofline floor is most of the
per-token cost, and `post.chunk.m4096` at 317× is most of the time to first token.

At a 4,096-token context the fork serves this checkpoint cleanly: 480 requests across fifteen rows, zero failures,
and three workload shapes that differ far more in repeat stability than the shapes themselves suggest. Single-stream
decode is the steadiest measurement on this stack by a wide margin — 6.46 tok/s on all five repeats, with mean time
per output token inside a 0.2 ms band. The eight-way concurrent shape is the least steady, spanning 29.03 to
38.31 tok/s across its five repeats; a comparison built on that row alone could show a 30% difference from run-to-run
noise and nothing else.

This also supersedes the earlier 30.79 tok/s figure as a reference point. That number was measured at a 1,048,576-token
context, which the Emmy arm cannot hold, so it was never a legitimate baseline for a compiler comparison. The rows
below are at the envelope both arms can share.

## Measurements

| Concurrency | Input → output | Repeats | Output tok/s, mean ± SD | Range | Mean TPOT | Mean TTFT | Failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 128 → 128 | 5 | 33.71 ± 3.24 | 29.03 – 38.31 | 221.6 ms | 2.50 s | 0 |
| 4 | 1024 → 256 | 5 | 19.99 ± 0.39 | 19.32 – 20.41 | 179.7 ms | 5.41 s | 0 |
| 1 | 2048 → 512 | 5 | 6.46 ± 0.00 | 6.46 – 6.47 | 147.7 ms | 3.77 s | 0 |

Spread is the population standard deviation over the five repeats. Per-repeat output throughput:

- Concurrency 8: 38.31, 29.03, 33.71, 31.58, 35.94 tok/s
- Concurrency 4: 19.88, 20.01, 20.32, 19.32, 20.41 tok/s
- Concurrency 1: 6.46, 6.46, 6.46, 6.47, 6.46 tok/s

Every repeat redeploys the server, so each row carries its own model load and warm-up; the stability above is therefore
across fresh processes, not across clients hitting one long-lived server.

The single-stream row is the one worth building the Emmy comparison on. Its inter-token latency and time per output
token agree to two decimal places within a repeat and to 0.2 ms across repeats, so a real difference between arms would
be visible far below the noise floor of the other two shapes.

## The Emmy arm

The Emmy arm serves. The numbers below come from a serving boot on the same host driven by direct HTTP requests, not
from `emmy bench`, so they carry no experiment records and no archive. The compiler measurements that follow them are
kept because they explain where the time goes.

### Serving measurements

Boot on 2026-09-12, `--strict-evidence`, `EMMY_GEN_M1_TIER=0`, `gpu_memory_utilization` 0.90, prefix caching on.
Greedy decoding, single stream, streamed responses timestamped per chunk.

| Shape | TTFT | TPOT mean | TPOT range | Output tok/s | Decode steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 in → 33 out | 5.94 s | 5.565 s | 5.553 – 5.594 | 0.180 | 32 |
| 2,405 in → 9 out | 88.22 s | 5.610 s | 5.598 – 5.665 | 0.178 | 8 |

Against the fork's single-stream row, which is the steadiest measurement on this stack:

| | 1Cat fork | Emmy | Ratio |
| --- | ---: | ---: | ---: |
| TTFT | 3.77 s (2,048 in) | 88.22 s (2,405 in) | 23× |
| Mean time per output token | 147.7 ms | 5.610 s | 38× |
| Output tok/s, decode only | 6.77 | 0.178 | 38× |

Emmy's inter-token latency is the more stable of the two in relative terms: 40 decode steps across the two shapes
span 5.553 s to 5.665 s, a 2% spread, and the two shapes' means differ by 0.8% despite a 480× difference in context.
A repeat of the 2,405-token request returned in 56.29 s against a warm prefix cache, which is nine decode steps and no
prefill — an independent confirmation of the per-token figure.

The per-token cost is accounted for by one program. `post.decode.m16` runs at 1,149× its ~60 µs roofline floor, or
about 68 ms per layer, which is roughly 3.0 s of the 5.6 s across 44 layers. `post.chunk.m4096` at 317× its ~1,955 µs
floor is about 620 ms per layer and dominates the 88 s time to first token. Neither has ever been recorded — only the
`m1` shapes were.

### The static M=1 decode tier is a defect

With the M=1 tier enabled, which is the default, no generation request ever completed. The failure is not prefill: a
one-token completion returns in 5.8 s, and a one-token completion runs zero decode forwards because prefill produces
the first token's logits. vLLM's own scheduler dump at the failure names the step —
`total_num_scheduled_tokens=1`, `num_computed_tokens=[5]`, `num_output_tokens=[1]` — the first true decode forward.

Sampling one worker's layer index through a complete 22-layer pipeline stage gives a flat rate with no decay:

| Layer transition | 0→1 | 1→2 | 5→6 | 10→11 | 15→16 | 20→21 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seconds | 60 | 61 | 60 | 60 | 61 | 60 |

Twenty-two layers in 1,379 s, 62.7 s each, constant from the first layer to the last. That flatness is what rules out
first-use cost: a cost paid once per process would fall away after the first layers. A whole forward at that rate is
about 2,700 s, which is why it overruns the 300 s RPC deadline. Raising that deadline does not help, because the 600 s
NCCL collective watchdog then tears the workers down, and under two-stage pipelining the second stage blocks on the
first for the entire traversal. That watchdog timeout is a compiled-in constant with no environment override.

The same request through the M=16 bucket twins takes 5.6 s per token, so the tier carries a factor of about 485×.

What makes this hard to see from the boot audit: the audit times `post.decode.m1` at 17 ms per layer, and it does so
through raw launches, while serving drives the same program through whole-program CUDA graph capture and replay. The
M=1 and M=16 post programs have identical 36-kernel launch lists, and the M=16 tier performs the same per-layer
captures during prefill at about 0.16 s per layer. So the cost is in the M=1 tier's capture-and-replay path rather
than in the program, and the audit cannot reach it.

Seven other explanations were checked against the code and eliminated: the fixed-slot expert combine (DeepSeek's
hyper-connection branch returns before that check, and tensor-parallel expert sharding excludes it independently), the
refused expert `m256` twin (that tier applies only between the decode bucket and 256 routed rows; at one row the
dispatch selects the `one` tier, which built on all sixteen workers), the Emmy programs themselves, graph-cache
eviction, a launch-heavy M=1 program, ordinary first-use cost, and a straggler worker — all eight workers of a stage
sit at the same layer with every GPU at 100%.

### The boot reaches a serving state

On 2026-09-11 a boot with `--strict-evidence` came up: `/health` returned 200, `/v1/models` listed the checkpoint, and
vLLM logged `Application startup complete`. Engine init — profile, KV-cache creation and model warm-up — took 949 s.

Strict evidence is what made the boot possible, and the mechanism is worth recording. An earlier boot without the flag
logged 16 prior-clip warnings — one per worker — reporting a latency-proxy exponent of 996 against a shipped
artifact that peaks near 28. Past that clip every candidate scores identically, so the ranking degenerates to
enumeration order. The warning fires once per process, so it appeared once early and then held silently for the
whole run. Under
`--strict-evidence` that boot logged **zero** prior clips: every kernel that ran was decided by a measured row.

Strict evidence refused exactly one fork, identically on all 16 workers — the expert `m256` twin, on its `030_cut`
fork, with no measured row spelling a kernel-set arm. That width falls back to a wider tier and the boot continues; the
`m256` expert program is absent from the golden entirely, because the runner's expert prefill tier is a hardcoded
constant that the pinned serving config has no field to declare.

### Where the time goes

The boot's own roofline audit, consistent between the first and last layer:

| Program | Over roofline floor | Floor |
| --- | ---: | ---: |
| `pre.chunk.m4096` | 64,604× | ~30 µs |
| `post.decode.m1` | 1,273× | ~59 µs |
| `post.decode.m16` | 1,154× | ~59 µs |
| `post.chunk.m4096` | 343× | ~1,955 µs |

The recorded golden the boot deployed from carries 903 realizations, 295 of them measured. Summing the measured
per-kernel latencies within each program:

| Program | Measured sum | Worst single kernel |
| --- | ---: | ---: |
| `pre4096` | 43.2 s | 19.6 s |
| `pre1` | 29.7 s | 29.7 s |
| `pre16` | 17.2 s | 17.2 s |
| `expert-sym@mxfp4` | 1.61 s | 1.38 s |
| `expert1@mxfp4` | 1.33 s | 1.33 s |
| `post4096` | 0.82 s | 0.45 s |
| `post1` | 0.12 s | 0.045 s |
| `post16` | 0.086 s | 0.047 s |

The `pre` family dominates by three orders of magnitude, and every one of those programs is the same
`k_linear_mean_reduce` kernel. `pre1` and `pre16` are the decode path, which is what the RPC deadline measures.

### Three programs were recomputing work inside sweeps

Benched end to end on one card, `pre4096`'s greedy pick was a two-kernel placement split totalling 1,923,598 µs —
whole-program 1,923,912 µs, agreeing with the boot audit's 64,604× against a ~30 µs floor. A 4,096-row mean-reduce
costing 1.92 s is not a search problem, so we decoded what it computes.

It computes, per token, an RMS statistic over 16,384 hidden values and **four length-16,384 dot products**, then
sigmoid-mixes four 4,096-wide streams using those four coefficients. The dots depend on token and stream only. The
generated code recomputed them for **every one of 4,096 output channels** — 1.10 trillion dot-product terms per kernel
against 268 million if computed once per token, or **8,192× redundant work**.

Four placement cuts hoist them out of the channel sweeps. The same shape appears in the decode program's two hot
kernels: `9e578e` ran sixteen long dot products on a **single thread** (`if (_gid < 1)`), and `4e26cc` evaluated its
sixteen mixing logits 352 times across four threads.

| Target | Greedy pick | With cuts | Factor | Correctness vs greedy |
| --- | ---: | ---: | ---: | --- |
| `pre4096` | 1,923,598 µs (2 kernels) | **3,238 µs** (5) | **594×** | pass, max abs 1.2e-4 (fp16 level) |
| `post1` `9e578e` @ m1 | 42,278 µs (20 kernels) | **9,866 µs** (22) | **4.3×** | pass, exact (0.0 / 0.0) |
| `post1` `4e26cc` @ m1 | 30,016 µs (1 kernel) | **5,347 µs** (7) | **5.6×** | pass, exact (0.0 / 0.0) |

No new compiler capability was needed — the better structure was already expressible. The cuts move where work happens
rather than reassociating reductions, which is why the two decode results are bit-exact; `pre4096` differs only at fp16
rounding, and the golden's own alternative schedule for it shows the same magnitude.

### The election picks them up unaided

Recording each pick with `emmy run --golden … --bench --record-greedy` under the full pin list, merging those rows
into the serving golden, and re-booting with `--strict-evidence` moves the boot's own audit:

| Program | Before | After |
| --- | ---: | ---: |
| `pre.chunk.m4096` | 64,746× | **107×** |
| `post.decode.m1` | 1,275× | **294×** |
| `post.chunk.m4096` | 320× | 321× |
| `post.decode.m16` | 1,156× | 1,156× |

The last two are untouched because only the `m1` shapes were recorded. The first two match the bench measurements
(605× and 4.35×) closely enough to corroborate them independently. Nothing was hand-pinned in the serving boot: the
recorded rows win the election on price.

### Four things that cost a day, recorded so they cost nobody else one

- **A `perf` row times one CUDA op.** When the schedule under test splits a kernel, the row is a fragment. Reading it
  as the program's latency overstates the result by orders of magnitude. The check that catches it is whether the
  boot's roofline ratio moves.
- **A `PLACE` pin replaces the entire placement decision**; it does not add to it. Pinning 2 of a program's 21 cuts
  silently discards the other 19 and produced a hanging 3-kernel program with no diagnostic saying so.
- **`emmy tune` defaults to a 2 s cumulative-GPU-time bench budget.** On a kernel whose baseline is ~1 s per launch,
  warm-up plus one measured iteration exhausts it, and a run records **zero** valid latencies while appearing to search
  normally. `EMMY_BENCH_RUN_TIMEOUT_S` raises it.
- **A replay must use the golden the boot uses.** Benching one of these kernels against a golden with no rows for it
  produced a 60-second hang for a program the boot runs at 42 ms.

## What this run does not establish

- **It is not a balanced A/B.** The two arms were measured in separate invocations, not in one run with their order
  alternated inside each repeat the way the RTX 5090 gemma-4 experiment balances time and thermal drift. The gap is
  large enough that ordering cannot explain it, but the numbers are directional, not a controlled comparison. There is
  also no baked Emmy image for this model, so the single-image, two-entrypoint mechanism that A/B uses does not exist
  here yet.
- **The two arms did not run the same envelope.** The fork rows are at `gpu_memory_utilization` 0.80 with prefix
  caching disabled; the Emmy boot needed 0.90 and ran with prefix caching on. The Emmy shapes are 2,405 and 5 input
  tokens against the fork's 2,048, and 9 and 33 output tokens against its 512. Whichever values a joint recipe adopts,
  both arms must share them.
- **The Emmy rows are one repeat each.** The fork rows are five repeats with a reported spread; the Emmy rows are
  single runs, and their stability claim rests on the spread of decode steps within a run, not across runs.
- **The Emmy arm ran with a non-default configuration.** `EMMY_GEN_M1_TIER=0` is required for it to answer at all.
  Any comparison including the default configuration would show no completions from the Emmy arm.
- **The Emmy rows came from direct HTTP requests**, not from `emmy bench`, so they have no experiment records and are
  not in the archive.
- **It is not a regression check against the August run.** That run used different prompt shapes, a different context
  length and four repeats, so the two are not comparable row for row.
- **Correctness was checked only by the deployment smoke test**, which asks one arithmetic question and reads one
  token back. No token-level or numerical agreement was measured.

## Protocol

`emmy bench experiments/golden-bench-2026/serving_deepseek_v4_flash_0731_v100x16 --ssh <host> --no-teardown` against a
pre-allocated 16× V100 SXM3 host. Fifteen rows: five repeats crossed with three zipped workload shapes. Each row
deploys the server, runs the deployment smoke test, then runs one benchmark pass with greedy decoding
(`temperature: 0`, `seed: 0`) and `ignore_eos`, so every request produces exactly its requested output length. The
three shapes send 64 prompts at concurrency 8, 24 at concurrency 4, and 8 at concurrency 1.

Engine: TP8 × PP2, `gpu_memory_utilization` 0.80, `max_model_len` 4096, `max_num_batched_tokens` 4096, block size 256,
`dtype` float16, `deepseek_v4_fp8` weight quantization and an FP8 KV cache, prefix caching disabled. The engine
allocated 57,594 tokens of GPU KV cache. Model load and warm-up took about 151 s per row, of which weight loading was
about 25 s; benchmark time was 242–345 s for the concurrent shapes and about 661 s for the single-stream shape.

## Machine and software

Ubuntu 24.04.1 LTS, kernel 6.8.0-124-generic, two Intel Xeon Platinum 8168 (80 logical CPUs), 1,437 GB RAM, sixteen
Tesla V100-SXM3-32GB behind twelve NVSwitches, driver 580.159.03, nvcc 12.9.86, cuBLAS 12.9.2.10, Docker 29.5.3.
Engine image `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608`, resolved digest
`sha256:276240257b224097876b5b6db8f0d32484dff6a6f168d6b03d6df188e5c65bc1`, vLLM `1.2.3.dev87+gd76126608`. Model
revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

The experiment records report repository revision `f38b3a3b`, which is the checkout the `emmy` console script imports
from. The run was driven from a worktree at `29ec3b40`; the recipe used is byte-identical to the one on `main`, and
`emmy/benchmark`, `emmy/provisioning` and `emmy/deployment` are byte-identical between the two revisions, so the
orchestration that executed is the code the records name.

## Run and archive

Timestamp `2026-09-10T17:41:53Z`, run ID `20260910T174153Z`, fifteen rows, all `succeeded`, one run ID across every
record. Archive `results_v100x16.tar.gz`, root member `2026-09-10_17-41-53/`, 48 members: one `*.experiment.yaml`,
one `*.benchmark.log` and one `*.server.log` per row, plus the two orchestration logs. Row names carry their shape,
for example `v100x16_mc1_np8_ril2048_rol512_r0_ffaecb951bfc`.

The host is pre-allocated and privately owned, so its address and login were replaced with `<redacted-host>` and
`<redacted-user>` throughout the archived records and logs before archiving. That substitution is the only edit made
to what `emmy bench` produced; every timing, count and configuration value is untouched, and the records still parse
and still carry one run ID and a terminal status each.
