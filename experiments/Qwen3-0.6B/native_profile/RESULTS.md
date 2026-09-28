# Native Qwen3 GPU profile on RTX 4080

A repeated linear kernel accounts for 69.2% of GPU kernel time in this short-request profile. Greedy token selection
accounts for another 9.3%. These are concrete optimization targets; this profile does not establish a benefit from
continuous batching or a paged KV cache.

## Protocol and result

The 2026-09-25 run profiles the same qualified Qwen3-0.6B artifact as the
[serving comparison](../native_serving/RESULTS.md), on one NVIDIA GeForce RTX 4080, driver 595.91.07. Source revision
is `13c07562`. Nsight Systems 2026.1.3 records CUDA calls and individual graph nodes. CPU sampling is disabled.
Three requests use `The capital of France is`, greedy decoding, and exactly sixteen output tokens with EOS ignored.
Each request has five input tokens, so the complete trace contains sixty token steps and forty-eight output tokens.
All three requests succeed. This is one profile run, including first-request graph capture, not a repeated timing
comparison or a CPU critical-path analysis.

| GPU operation | Total kernel time | Instances | Mean per invocation | Share |
| --- | ---: | ---: | ---: | ---: |
| `k_linear_mean_reduce_db6ac3__place_bcff5781af` | 1,770.120 ms | 1,680 | 1.054 ms | 69.2% |
| `native_sample` | 238.671 ms | 60 | 3.978 ms | 9.3% |
| `k_linear_mean_reduce_1fb2b6__place_95b1fbc966` | 119.483 ms | 1,680 | 0.071 ms | 4.7% |
| `k_linear_mean_reduce_1fb2b6__place_3cfe6ff299` | 81.177 ms | 1,680 | 0.048 ms | 3.2% |
| `k_linear_mean_reduce_1fb2b6__place_c0ae7243e0` | 77.674 ms | 1,680 | 0.046 ms | 3.0% |
| Output-head kernel `k_linear_mean_reduce_6ac318` | 69.612 ms | 60 | 1.160 ms | 2.7% |

The dominant linear kernel runs once per layer per token step: 28 layers times 60 steps. Sampling skips the first
four positions of each prompt, so its mean includes twelve near-empty invocations. Its median invocation is
4.797 ms. The current greedy implementation scans the vocabulary serially on one GPU thread; parallel reduction
is a separate possible improvement that must preserve lowest-index tie breaking and nonfinite-logit rejection.
Neither observation is an achieved speedup.

Sixty graph launches confirm captured execution for the sixty token steps. CUDA API time is mostly stream
synchronization, which includes waiting for GPU work. It must not be counted as removable CPU overhead. The profile
does not isolate exposed host gaps, and kernel-time percentages are not end-to-end speedup predictions.

## Evidence and reproduction

The run ID is `2026-09-25_07-48-44`; its sole row, `rtx4080x1_ac8d3750277c`, succeeded. The anonymized archive
`results_rtx4080x1.tar.gz` retains the redacted experiment record, server log, three responses, artifact/binary
hashes, and the kernel/API CSV summary. The summary is the source of the table above. The binary Nsight report and
SQLite export remain local because they contain opaque machine metadata.

Set `NATIVE_ARTIFACT` to the checkpoint-qualified serving bundle, build the matching native server, and run this
recipe with `emmy bench` and `--local`. The recipe takes the shared GPU lock, limits the profiler to twenty seconds,
and terminates its server. The artifact's weights are not included in the archive. See the linked serving report
for artifact qualification and the precision/schedule differences from stock vLLM.

## Publication privacy

The shared archive is an anonymized copy. Usernames, hostnames, GPU UUIDs, PCI addresses, network addresses, local
paths, filesystem details, and uptime were removed. Archive ownership and timestamps were normalized. Numeric
benchmark measurements are unchanged. Original records and binary traces remain local. `ANONYMIZATION.txt` records
these transformations; the shared records are not the unmodified originals emitted by the harness.

The execution revisions above predate consolidation of unpublished commits to exclude the private archives from
Git history. Runtime and compiler source match merged PR #890 at `b4b3774c`; this work changes only experiment
recipes, reports, and the plan.
