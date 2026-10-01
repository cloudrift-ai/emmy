# Native-serving baseline

All four configurations now start and complete the serving comparison. The corrected Emmy schedules produce the
same two fixed-prompt completions as stock vLLM, but every measured workload remains slower than stock. Reducing
activation capacity recovers KV-cache space within the existing integration. These results support improving
compiled GPU execution before expanding the native serving runtime.

The results below describe revision `2144b15a`. The current golden retains 139 compatible records for the three
fixed-shape pre-attention programs. Compiler normalization changed the other kernel sets; their 194 records were
removed because identity updates cannot preserve their measured computation. The retained records strictly decode,
but complete serving qualification now requires fresh RTX 4080 measurements. Historical results remain unchanged.

## RTX 4080 × 1 — 2026-09-19

### Protocol

Run `20260919T065247Z`, directory `2026-09-19_06-52-47`, used clean revision
`2144b15aa3f21e8325147f02d36951945c98747d` throughout all four configurations. The existing recipe ran once through
`emmy bench --local`; there was no failure-only rerun. Each server shut down, and no task-owned GPU process remained.

Qwen/Qwen3-0.6B is pinned to `c1899de289a04d12100db370d81485cdf75e47ca`, FP16, context 4,096, maximum four
sequences, a 256-token batched prefill limit, Triton attention, full CUDA graphs with explicit capture sizes, and
prefix caching disabled. The configurations are stock, width-16 decode, the single-token tier, and that tier with
activation capacity reduced from 1,024 to 256. Emmy uses the committed experimental golden with strict measured
evidence, an empty tune database, a fresh online-prior path, and a fresh pack directory for each configuration.
The cubin cache is shared; startup is not a cold-cache measurement.

Each fixed workload has three repeats, eight requests, two warmups, concurrency one or four, inputs 32/256/1,024,
and exactly 64 output tokens. Sampling is greedy with seed zero. The mixed-length and profiled workloads are
separate. Fixed-prompt completions run before benchmarks with temperature zero, seed zero, and 16 output tokens.

### Results

All four row records have `succeeded` status. Every repeated and mixed workload completed eight requests; each
profile completed two. The short-input concurrency-one comparison is:

| Configuration | Median TPOT range, ms | Median TTFT range, ms | KV-cache space, GiB |
| --- | ---: | ---: | ---: |
| Stock | 2.273–2.281 | 7.085–7.821 | 8.12 |
| Width 16, single-token tier off | 7.935–8.189 | 100.938–105.158 | 6.17 |
| Single-token tier, capacity 1,024 | 8.368–8.378 | 106.579–107.671 | 6.18 |
| Single-token tier, capacity 256 | 7.937–7.957 | 99.865–100.158 | 7.32 |

Ranges span three repeats, not confidence intervals. The full table, mixed-length result, and memory accounting
are in [SHAPES_MEMORY.md](SHAPES_MEMORY.md). The small-capacity configuration improves KV-cache space by 1.14 GiB
and 10,672 tokens over the larger single-token configuration. It still trails stock's cache capacity and latency.
Short concurrency-four Emmy results vary more across repeats; do not infer a universal single-token-tier advantage.

Both prompts produce exactly the same completion text in all configurations: “The capital of France is” begins
“Paris. The capital of Italy is Rome.”, and “2 + 2 =” begins “4, so the sum is 4.” Full responses are in each
`generation.json`. This is a bounded checkpoint-generation check, not broad task-quality or logit-parity evidence.
The independent [schedule qualification](SCHEDULES.md) passes all 32 synthetic-input checks at unchanged strict
numerical tolerances, plus all 333 schedule decodes and eight serving-program compiles on the final source.

The [dispatch profile](DISPATCH.md) records captured execution in every configuration. Most time is GPU work;
observed gaps between steps without GPU work are only a few microseconds. The current slow schedules, rather than
uncaptured per-kernel Python submission, are the principal measured limitation. This experiment does not establish
a native-runtime speedup or justify replacing the serving frontend.

### System and evidence

One RTX 4080, 16,376 MiB, `sm_89`; Core i9-14900K; Ubuntu 24.04.5, kernel 7.0.0-31-generic; driver 595.91.07;
NVCC 13.3.73; cuBLAS 13.6.0.2; Torch 2.11.0+cu130; vLLM 0.23.0; Transformers 5.14.1. Exact packages are retained
in each `requirements.txt`. The desktop shares the GPU. No cloud server was rented and no other task-owned GPU
workload overlapped serving measurements. The user removed the earlier GPU time limit before this run.

[results_rtx4080x1.tar.gz](results_rtx4080x1.tar.gz) contains the complete timestamped directory, all four system-only
experiment records, command and server logs, generation and benchmark JSON, memory snapshots, golden digests, and
CPU/GPU traces. Record and directory stems are:

| Configuration | Stem |
| --- | --- |
| Stock | `rtx4080x1_c1024_lstock_m1_f5b4395d2139` |
| Width 16 | `rtx4080x1_c1024_lbucket16_m0_242586f9983a` |
| Single-token, capacity 1,024 | `rtx4080x1_c1024_lm1_m1_6f930adf2e27` |
| Single-token, capacity 256 | `rtx4080x1_c256_lm1-s_m1_679bbdc79eb9` |

Hostnames, account names, home/workspace paths, private addresses, GPU identifiers, and archive owner metadata are
removed from publication. Hardware specifications, measurements, and executable payloads remain intact. Each
archive member is verified against its original with only those redactions. Local paths are descriptive, not
portable inputs.

### Previous runs and remaining work

The September 18 run at `0d5ac561` completed stock but timed out during initialization in every Emmy configuration.
Its reports and archive remain in Git at `2144b15a`. The September 16 run in merged PR #820 timed out during Emmy
compilation; that evidence remains at `27e11a30`. The current named archive replaces those runs with this complete
comparison. The historical API/execution-contract [investigation](INVESTIGATION.md) remains separately scoped.

Correct serving is now demonstrated for this bounded workload. Efficient general schedules, broader generation
parity, measured useful/padded rows, and separate activation/scratch allocation accounting remain future work.
The native implementation remains a static runtime foundation; complete cached native generation and HTTP serving
are not delivered by this PR.
