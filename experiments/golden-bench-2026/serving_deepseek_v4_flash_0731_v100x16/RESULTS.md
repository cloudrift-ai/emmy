# DeepSeek V4 Flash 0731 serving on 16× V100 SXM3

## Conclusion

This run characterizes the pinned 1Cat fork at the short serving envelope. It is **not** the Emmy-versus-fork A/B, and
nothing here compares a compiler. The Emmy arm could not be deployed, so only one arm ran.

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

## What this run does not establish

- **It is not an A/B.** One arm ran. A comparison needs both arms in one invocation with their order alternated inside
  each repeat, the way the RTX 5090 gemma-4 experiment balances time and thermal drift. Until then no claim about Emmy
  against the fork is supported by this evidence.
- **The Emmy arm cannot deploy today.** Two separate blockers, both recorded in the serving plan: a whole-program
  serving compile stalls in the schedule search and never reaches a serving state, and the last boot that did reach
  serving lost a completion to vLLM's 300 s `sample_tokens` timeout. There is also no baked Emmy image for this model,
  so the single-image, two-entrypoint mechanism the gemma-4 A/B uses does not exist here yet.
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
