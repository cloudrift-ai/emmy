# DeepSeek V4 Flash 0731 serving on 16× V100 SXM3

## Conclusion

This run characterizes the pinned 1Cat fork at the short serving envelope. It is **not** the Emmy-versus-fork A/B, and
nothing here compares a compiler. Only the fork arm produced serving numbers.

The Emmy arm now boots and serves — that changed on 2026-09-11 and is recorded under "The Emmy arm" below — but it
cannot complete a request: the first generation exceeds vLLM's `sample_tokens` RPC deadline and kills the engine. Its
numbers here are per-kernel and per-program compiler measurements, not serving measurements, and they do not belong in
the same table as the fork's throughput.

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

These are compiler measurements, not serving measurements. They come from a serving boot and a kernel tuning run on the
same host, not from `emmy bench`, so they have no experiment records and no archive. They are recorded here because they
are the first measured Emmy evidence for this model on this GPU, and because they say precisely why the serving column
is still empty.

### The boot reaches a serving state

On 2026-09-11 a boot with `--strict-evidence` came up: `/health` returned 200, `/v1/models` listed the checkpoint, and
vLLM logged `Application startup complete`. Engine init — profile, KV-cache creation and model warm-up — took 949 s.

The first chat completion then killed it. One 24-token greedy request returned HTTP 500 after 173.1 s with
`TimeoutError: RPC call to sample_tokens timed out`, followed by `EngineDeadError`. The server is reachable and
correctly configured; a forward pass simply does not fit inside the engine's RPC deadline.

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

### Tuning finds real headroom in that kernel

A tuning run against freshly captured serving twins, scoped to `pre4096` and spread over all 16 cards, measured 24
distinct schedules of its dominant kernel. They span **114 ms to 1,088 ms — a 9.5× spread**, with the fastest setting
fewer schedule knobs than the slowest. None uses a placement cut.

So the schedule search bites on this kernel family, and a large part of the gap is a pick problem rather than a
compiler limit. It does not follow that serving is close: at 114 ms across 22 layers this one kernel still accounts for
seconds of prefill against a ~30 µs floor, and the decode-path programs `pre1` and `pre16` were outside this run's
scope.

One measurement note. `emmy tune` defaults to a 2 s cumulative-GPU-time bench budget for fast-fail sweeps. This
kernel's baseline is about 1 s per launch, so warm-up plus one measured iteration exhausted it and the first attempt
recorded **zero** valid latencies in eleven minutes — 70 variants failed on the budget and 28 on genuine hangs. Every
number above comes from a re-run at `EMMY_BENCH_RUN_TIMEOUT_S=30`. A tuning result on a kernel this slow is not
trustworthy without checking that the budget admitted it.

## What this run does not establish

- **It is not an A/B.** One arm ran. A comparison needs both arms in one invocation with their order alternated inside
  each repeat, the way the RTX 5090 gemma-4 experiment balances time and thermal drift. Until then no claim about Emmy
  against the fork is supported by this evidence.
- **The Emmy arm serves but cannot answer.** It reaches a serving state and then loses every completion to the
  `sample_tokens` RPC deadline, so it produces no throughput, TTFT or TPOT to compare. An earlier claim that a
  whole-program compile "stalls in the schedule search" was wrong: that boot was not stuck but grinding, because a
  degenerate prior had collapsed the ranking to enumeration order and each refused row re-resolved the entire program.
  There is also no baked Emmy image for this model, so the single-image, two-entrypoint mechanism the gemma-4 A/B uses
  does not exist here yet.
- **The Emmy numbers above are not serving numbers.** They are per-kernel and per-program latencies from a boot audit,
  a recorded golden and a tuning run. Nothing in them can be compared against the fork's tokens per second.
- **One envelope parameter is unreconciled.** This recipe runs at `gpu_memory_utilization` 0.80; the Emmy arm's last
  serving boot needed 0.90 to fit. Whichever value the joint recipe adopts, both arms must share it, and these numbers
  do not transfer to a run at 0.90.
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
