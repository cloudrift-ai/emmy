# Qwen3.8 GDN register schedule on V100

## V100 SXM2 16 GB — 2026-09-19

Loading external operands directly into MMA fragments and shuffling packed FP16 pairs makes the register schedule
2.37–2.87× faster than its first Volta implementation. Every completed comparison now beats eager PyTorch.
FP32 accumulation matches torch.compile at 12 heads and 128 tokens, and beats it by 1.25× at 512 tokens.
At 48 heads it remains 5–34% slower than torch.compile. FP16 partial accumulation remains slower than FP32.

The [hardware profiles](../gdn_profile/RESULTS.md) measure the corresponding instruction and stall changes.
This qualifies the inter-chunk state update, not full GDN prefill or model serving.

### Measurements

Microseconds per complete forward, including output assembly. The 12-head, 128-token values are medians of three
fresh processes; other entries have one process. Each new Emmy measurement has paired eager and torch.compile
measurements in the same process. The before column comes from the earlier run on this host with identical pins.

| Value heads | Tokens | Accumulation | Emmy before | Emmy now | Eager PyTorch | torch.compile |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 12 | 128 | FP32 | 128.26 | 52.05 | 62.43 | 52.74 |
| 12 | 128 | FP16 + promotion | 142.85 | 60.16 | 62.62 | 52.71 |
| 12 | 512 | FP32 | 559.10 | 213.76 | 239.79 | 267.49 |
| 12 | 512 | FP16 + promotion | 625.15 | 217.86 | 239.13 | 267.80 |
| 48 | 128 | FP32 | 285.70 | 116.22 | 141.98 | 86.90 |
| 48 | 128 | FP16 + promotion | 309.25 | 127.85 | 141.96 | 86.97 |
| 48 | 512 | FP32 | 1,209.34 | 480.77 | 542.24 | 459.41 |
| 48 | 512 | FP16 + promotion | 1,295.36 | 515.07 | 541.58 | 459.71 |

FP32 improves by 2.46–2.62×; FP16 improves by 2.37–2.87×. FP16 is still 1.9–15.6% slower than FP32 at the selected
promotion interval. The 1.3% difference between Emmy and torch.compile in the smallest FP32 case is comparable to
Emmy's repeat variation, so it is best described as parity. The 512-token, 12-head gain over torch.compile is larger.

The three smallest-case Emmy repeats range from 51.94 to 52.68 µs for FP32 and 59.71 to 60.54 µs for FP16.
Their spans are 1.42% and 1.38% of their medians. Every FP16 repeat is slower than every FP32 repeat.

All twelve runs pass strict eager-reference correctness at rtol=atol=0.001. Maximum absolute error is 5.01e-5 with
FP32 accumulation and 8.18e-5 with FP16 accumulation. All record CUDA-graph capture, the requested register schedule,
and three launches: the chunk-loop kernel and two small output-assembly kernels.

The chunk-loop kernel uses 255 registers per thread, except the 512-token FP16 case, which uses 254. All twelve
runs have zero shared memory and zero per-thread local-memory bytes. Register pressure increased from the earlier
168–255 registers, so avoiding spills does not mean occupancy is high. Two warps per CTA produce 48 CTAs at
12 heads and 192 CTAs at 48 heads. The schedule still carries the full key dimension in every owning warp.

### Protocol and implementation

The experiment reuses the original comparison's `chunk_state.py` and the ordinary `emmy run --bench --strict`
harness. No separate timing implementation is involved. State dimensions are 128 keys by 128 values, and each
chunk contains 64 tokens. Twelve value heads represent one TP4 rank of Qwen3.8-27B; 48 represent the whole layer.
There is no tensor-parallel communication in this test.

Chunk-local transforms are supplied as seeded synthetic FP32 inputs, shared by every backend. The operation starts
with a zero state and returns corrected values and the final state. It does not test model-level accuracy.

Pins are two warps per CTA, one register slot, eight logical Volta column fragments, and a four-step K chunk.
Both accumulation modes convert MMA operands to FP16 and keep carry state in FP32. FP16 partial sums promote
and clear every four MMA steps, or sixteen scalar products on Volta. The explicit atom pin selects this arithmetic
without enabling unrelated FAST_MATH approximations. These pins are unchanged from the before measurement.

The optimization reuses the ordinary direct fragment loader and its address analysis for materialized B operands.
Computed or non-unit-stride operands retain the C-fragment gather and repacking path. Volta C→A conversion now
packs adjacent FP16 values before shuffling them, reducing eight shuffles to four. Intermediate matrix results still
stay in registers and are reused. The change is structural and also applies outside GDN; it adds no model dispatch.

Each process requests ten warmups and 100 measured iterations. Eager PyTorch, fullgraph torch.compile with
max-autotune, and Emmy use external CUDA-graph capture. Reported values are minimum whole-forward latency.
Emmy uses O3, no repository golden, and fresh task-local tuning and prior files. The recipe allows 120 seconds
per benchmark compilation and 360 seconds per process. Nsight-instrumented timings are excluded from this table.

### Scope and earlier failures

This optimization sweep selects the twelve carry rows at 128 and 512 tokens. All twelve succeeded. It does not
repeat the earlier four 2,048-token timeouts or two full-prefill timeouts. In that earlier sweep all six hit the
360-second process limit without a timing or accuracy verdict; full-prefill lowering alone took about 315 seconds.
Those cases remain unqualified. Their records and all eighteen original rows remain in the
[before archive](https://github.com/cloudrift-ai/emmy/blob/3b47f9f116f1e730afa63ddbc1b57a01404e720a/experiments/Qwen3.8-27B/gdn_register/results_v100x1.tar.gz).

The [original comparison](../gdn_kernels/RESULTS.md) also measures FLA/Triton and FlashQLA full prefill. Their
operation includes work absent here, and they use uncaptured CUDA events. These carry timings cannot establish
a speedup over those implementations. No modern GPU was available for performance qualification.

### Reproduction and evidence

Run `emmy bench experiments/Qwen3.8-27B/gdn_register --ssh USER@HOST --filter workload=chunk_state`
with `--filter 'tokens=[15]*'`, adjusting the recipe's Python and CUDA paths for the host. This selects exactly
the twelve rows above; omitting the filters also attempts the larger and full-prefill cases.

The measured host has four Tesla V100-SXM2-16GB GPUs, a Xeon E5-2680 v4 with 24 exposed logical CPUs, and 219.5 GB RAM.
Each benchmark uses GPU 0. Software is Ubuntu 24.04.1, driver 580.178.04, CUDA toolkit 12.9.86, Python 3.12.3,
PyTorch 2.14.0+cu126, and Transformers 5.14.1. Each row archives its package versions. The source is
`82dd222a7de99cd810852b254faa47d853430968`, with clean staged inputs; all 345 staged Python source files match locally.

The latest run is `20260919T211725Z`, directory `2026-09-19_21-17-25/`, completed at 21:27:52 UTC.
Two preceding invocations stopped during staging without running kernels: one found uncommitted documentation,
and the other found root-owned bytecode from the earlier profiler. Committing the documentation and repairing
ownership resolved them; the profile recipe now disables bytecode writes. No failed measured row was selectively rerun.

[`results_v100x1.tar.gz`](results_v100x1.tar.gz) contains the latest raw directory: twelve system-only
`<variant>_<row_id>.experiment.yaml` records, twelve matching `_artifacts.tar.gz` files, and two logs.
Each nested archive holds `artifacts/measurement.json`, `measurement.log`, `status.txt`, `requirements.freeze.txt`,
and `versions.txt`. JSON preserves whole-forward timings, strict correctness, source hashes, and schedule pins.
The ordinary log reports attributes from the same cached cubin used for execution. All 26 outer files were
byte-verified after archiving; the raw directory is retained locally. The supplied host remains running.
