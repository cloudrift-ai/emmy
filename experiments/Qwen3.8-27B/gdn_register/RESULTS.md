# Qwen3.8 GDN register schedule on V100

## V100 SXM2 16 GB — 2026-09-20

Packed operand loads reduce the 48-head FP32 carry update from 116.22 to 90.58 µs at 128 tokens and from
480.77 to 381.61 µs at 512 tokens. These are 22.1% and 20.6% latency reductions from the preceding PR revision.
The longer case now beats torch.compile by 1.21×. The shorter case remains 4.4% slower than torch.compile.
Both 12-head cases also improve, and FP32 remains faster than FP16 partial accumulation.

All sixteen runs pass strict correctness. The [hardware profiles](../gdn_profile/RESULTS.md) examine the remaining
load pressure. These measurements qualify the inter-chunk state update, not full GDN prefill or model serving.

### Measurements

Microseconds per complete forward, including output assembly. Every 128-token entry is the median of three fresh
processes; 512-token entries have one process. Each new Emmy run has paired eager and torch.compile measurements.
The previous column comes from the preceding run on this host with identical pins and inputs.

| Value heads | Tokens | Accumulation | Emmy previous | Emmy now | Eager PyTorch | torch.compile |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 12 | 128 | FP32 | 52.05 | 44.76 | 62.20 | 52.78 |
| 12 | 128 | FP16 + promotion | 60.16 | 56.46 | 62.65 | 52.81 |
| 12 | 512 | FP32 | 213.76 | 186.78 | 237.70 | 267.68 |
| 12 | 512 | FP16 + promotion | 217.86 | 190.87 | 236.89 | 267.90 |
| 48 | 128 | FP32 | 116.22 | 90.58 | 141.98 | 86.80 |
| 48 | 128 | FP16 + promotion | 127.85 | 107.06 | 141.75 | 86.87 |
| 48 | 512 | FP32 | 480.77 | 381.61 | 541.75 | 460.05 |
| 48 | 512 | FP16 + promotion | 515.07 | 403.46 | 543.71 | 459.57 |

FP32 latency falls 12.6–22.1%; FP16 latency falls 6.2–21.7%. At 12 heads, FP32 beats torch.compile by 1.18× at
128 tokens and 1.43× at 512 tokens. FP16 remains 2.2–26.1% slower than FP32 at the selected promotion interval.
All entries beat eager PyTorch. The 4.4% gap to torch.compile at 48 heads/128 tokens exceeds the observed variation;
it is still a gap, not demonstrated parity.

| 128-token repeats | Emmy range, µs | Span / median |
| --- | ---: | ---: |
| 12 heads, FP32 | 44.37–44.84 | 1.04% |
| 12 heads, FP16 | 56.20–56.52 | 0.58% |
| 48 heads, FP32 | 90.11–90.83 | 0.79% |
| 48 heads, FP16 | 106.84–107.18 | 0.32% |

All runs use rtol=atol=0.001 against eager. Maximum absolute error is 5.01e-5 with FP32 accumulation and 8.18e-5
with FP16 accumulation, unchanged from the preceding run. All record CUDA-graph capture, the requested register
schedule, and three launches: the chunk-loop kernel and two output-assembly kernels.

Every carry kernel uses 255 registers per thread and no shared memory. Short cases and all FP16 cases use zero
per-thread local-memory bytes. Both 512-token FP32 cases now use 136 bytes per thread, versus zero previously.
The latency benefit survives this increase. It is a tradeoff, not a claim that the new loader eliminates spills
for every loop shape. The existing four-chunk GPU regression cases retain zero local memory in both modes.

At 48 heads/512 tokens, the FP32 carry kernel measures 287.1 µs and the larger output-assembly kernel 90.6 µs.
Individual kernel measurements use a separate timing pass and do not add exactly to the whole-forward result.
Output assembly and register pressure remain substantial costs after reducing operand-load instructions.

### Implementation and protocol

The existing Volta loader now uses wide loads when its flattened base and row stride are provably four-element
aligned and K needs no mask. FP16 reads four values in one 64-bit load. FP32 uses two 64-bit loads, each coupled to
FP16 conversion in inline PTX. This keeps the temporary floats inside the packed load. The same four-value helper
serves shared-memory drains. The proof reuses existing expression divisibility; unsupported alignment, K tails,
and strided canonical B retain the scalar gather. No schedule field or GDN-specific dispatch was added.

Native four-value FP32 loads improved the initial probe but increased local-memory use in the existing four-chunk
checks. Packing each pair at the load preserved those checks. The selected schedule remains two warps per CTA,
with a full 128-key state slice owned by each warp across chunks. It launches 48 CTAs for 12 heads and 192 for
48 heads on an 80-SM V100. The change does not remove the register demand that limits occupancy.

The experiment reuses `gdn_kernels/chunk_state.py` and the ordinary `emmy run --bench --strict` harness.
State dimensions are 128 keys by 128 values, with 64 tokens per chunk. Twelve value heads represent one TP4 rank
of Qwen3.8-27B; 48 represent the whole layer. There is no tensor-parallel communication in this test.
Chunk-local transforms are seeded synthetic FP32 inputs shared by every backend. The operation starts with zero
state and returns corrected values and final state. It does not test model-level accuracy.

Pins are two warps, one register slot, eight logical Volta column fragments, and a four-step K chunk. Both modes
convert MMA operands to FP16 and retain FP32 carry. FP16 partial sums promote and clear every four MMA steps,
or sixteen scalar products on Volta. Explicit atom pins select this arithmetic without unrelated FAST_MATH changes.

Each process requests ten warmups and 100 measured iterations. Eager, fullgraph torch.compile with max-autotune,
and Emmy use external CUDA-graph capture. Values are minimum whole-forward latency, with process medians above.
Emmy uses O3, no repository golden, and fresh task-local tuning and prior files. Compilation has a 120-second
budget and each process a 360-second limit. Nsight-instrumented timings are excluded from this table.

### Earlier results and limits

The preceding optimization replaced C-fragment operand gathers with direct loads and shuffled packed FP16 pairs.
It made the initial Volta implementation 2.37–2.87× faster; its twelve rows remain in the
[previous report](https://github.com/cloudrift-ai/emmy/blob/99c5ad453a6d7381e4f1421ff088b9e293b8c79e/experiments/Qwen3.8-27B/gdn_register/RESULTS.md)
and [previous archive](https://github.com/cloudrift-ai/emmy/blob/99c5ad453a6d7381e4f1421ff088b9e293b8c79e/experiments/Qwen3.8-27B/gdn_register/results_v100x1.tar.gz).
The current table isolates the additional packed-load improvement.

The four original 2,048-token rows and two full-prefill probes hit the 360-second limit without timing or accuracy
verdicts; full-prefill lowering alone took about 315 seconds. This sweep does not rerun them. Their records remain
in the [original archive](https://github.com/cloudrift-ai/emmy/blob/3b47f9f116f1e730afa63ddbc1b57a01404e720a/experiments/Qwen3.8-27B/gdn_register/results_v100x1.tar.gz).
They remain unqualified. The [original comparison](../gdn_kernels/RESULTS.md) also measured FLA/Triton and FlashQLA
full prefill with uncaptured events. Those operations and timing protocols differ, so these carry results cannot
establish a speedup over them. No modern GPU was available for performance qualification.

### Reproduction and evidence

Run `emmy bench experiments/Qwen3.8-27B/gdn_register --ssh USER@HOST --filter workload=chunk_state`
with `--filter 'tokens=[15]*'`, adjusting the recipe's Python and CUDA paths. This selects the sixteen rows above;
omitting the filters also attempts larger and full-prefill cases. Both head counts now have three short-case repeats.

The supplied host has four Tesla V100-SXM2-16GB GPUs, a Xeon E5-2680 v4 with 24 exposed logical CPUs, and 219.5 GB RAM.
Each run uses GPU 0. Software is Ubuntu 24.04.1, driver 580.178.04, CUDA toolkit 12.9.86, Python 3.12.3,
PyTorch 2.14.0+cu126, and Transformers 5.14.1. Each row archives package versions. The clean source revision is
`0493653a8a67fad561decaaafea654c6b7bf697f`; all 345 staged Python source files matched that checkout before the run.
Subsequent compiler edits only reformatted the Python return statement that emits the same CUDA call.

Run ID is `20260920T031222Z`, directory `2026-09-20_03-12-22/`, completed at 03:26:53 UTC. All sixteen rows succeeded;
none was selectively rerun. [`results_v100x1.tar.gz`](results_v100x1.tar.gz) contains sixteen system-only
`<variant>_<row_id>.experiment.yaml` records, sixteen matching `_artifacts.tar.gz` files, and two logs.
Each nested archive holds `artifacts/measurement.json`, `measurement.log`, `status.txt`, `requirements.freeze.txt`,
and `versions.txt`. JSON preserves timings, strict correctness, source hashes, and pins. The ordinary log reports
attributes from the execution cubin. All 34 outer files were byte-verified, and all 114 files and nested members
were scanned for secrets. The raw directory remains local, and the supplied host remains running.
