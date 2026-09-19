# Qwen3.8 GDN register schedule on V100

## V100 SXM2 16 GB — 2026-09-19

Volta support makes the tested GDN carry update runnable, but it is slower than eager PyTorch and torch.compile
on every completed case. FP16 partial accumulation is also slower than FP32 accumulation at the chosen promotion
interval. The carried state and intermediate matrix results fit in registers in all completed measurements.

Follow-up [hardware profiles](../gdn_profile/RESULTS.md) show low tensor-pipeline activity and large instruction-fetch
stalls. They quantify the operand preparation and FP16 promotion costs and compare them with the eager GEMMs.

### Question and protocol

This experiment measures the inter-chunk correction and state update introduced by the register schedule in
[PR #853](https://github.com/cloudrift-ai/emmy/pull/853), with Volta support from
[PR #854](https://github.com/cloudrift-ai/emmy/pull/854). It reuses the original comparison's workload definitions
and the ordinary `emmy run --bench --strict` harness. It adds no timing implementation.

The state has 128 key and 128 value dimensions. Chunks contain 64 tokens; the tested lengths contain two, eight,
and 32 chunks. Twelve value heads represent one TP4 rank of Qwen3.8-27B; 48 represent the whole layer. Each row
uses GPU 0 of the supplied four-V100 host. There is no tensor-parallel communication or model serving in the test.

The chunk-local transforms are supplied as FP32 inputs. The operation returns corrected values and the final state,
starting with a zero state. These are seeded synthetic inputs, with the same values supplied to each backend.
They are not a model-level accuracy test. Full prefill is a separate two-row probe using the existing Transformers
workload, which also includes normalization, gate accumulation, the chunk-local solve, and output calculation.

Emmy pins two warps per CTA, one register slot, eight logical Volta column fragments, and a four-step K chunk.
Each warp owns sixteen value rows and all key columns. One kernel executes the ordered chunk loop inside each CTA;
small surrounding kernels assemble the returned tensors. Both accumulator modes convert MMA operands to FP16 and
keep the carry in FP32. FP16 partial accumulation promotes and clears the partial sums every sixteen products.
The explicit atom pin selects this arithmetic without enabling unrelated FAST_MATH approximations.

Each process requests ten warmups and 100 measured iterations. Eager PyTorch, fullgraph torch.compile with
max-autotune, and Emmy use the same external CUDA-graph capture. The reported value is the minimum whole-program
end-to-end latency. The 128-token, 12-head case has three fresh processes per accumulation mode; all other cases
have one. Emmy uses O3, no repository golden, and new task-local tuning and prior files. The final recipe allows
120 seconds per benchmark compilation and 360 seconds for the whole process, with a 375-second command limit.

### Carry update measurements

Microseconds per complete forward. The 12-head, 128-token entries are medians across three processes; the other
entries are one process each. Each baseline is measured in the same process as its corresponding Emmy row.

| Value heads | Tokens | Accumulation | Eager PyTorch | torch.compile | Emmy |
| --- | ---: | --- | ---: | ---: | ---: |
| 12 | 128 | FP32 | 62.46 | 52.67 | 128.26 |
| 12 | 128 | FP16 + promotion | 62.36 | 52.70 | 142.85 |
| 12 | 512 | FP32 | 241.08 | 267.43 | 559.10 |
| 12 | 512 | FP16 + promotion | 239.45 | 267.24 | 625.15 |
| 48 | 128 | FP32 | 142.16 | 87.08 | 285.70 |
| 48 | 128 | FP16 + promotion | 142.24 | 87.10 | 309.25 |
| 48 | 512 | FP32 | 547.78 | 461.65 | 1,209.34 |
| 48 | 512 | FP16 + promotion | 548.54 | 461.07 | 1,295.36 |

Emmy is 2.09–3.55× slower than torch.compile across these rows. FP16 accumulation adds 7.1–11.8% to Emmy's latency
relative to FP32 accumulation. It requires extra warp shuffles when promoting Volta's distinct FP16 accumulator
layout into the FP32 layout. These measurements show no performance benefit from that option at this interval.
They cover one register layout and promotion interval, not an autotuning sweep.

Across the three 12-head, 128-token repeats, Emmy ranges from 128.00 to 129.37 µs with FP32 accumulation and from
135.61 to 142.85 µs with FP16 accumulation. The spans are 1.1% and 5.1% of their respective medians. Every FP16
repeat is slower than every FP32 repeat. The corresponding torch.compile ranges are 52.65–52.74 µs and
52.69–52.84 µs; eager ranges are 62.16–62.53 µs and 62.33–62.54 µs.

All twelve successful carry processes pass strict eager-reference correctness at rtol=atol=0.001. Maximum
absolute error is 5.01e-5 for FP32 accumulation and 8.18e-5 for FP16 accumulation. Every successful process records
CUDA-graph capture, the requested register schedule, and three launches regardless of chunk count. The chunk-loop
kernel dominates runtime; the other two kernels assemble the returned tensors.

The compiled chunk-loop kernel uses the following register counts, with zero per-thread local-memory bytes and
zero shared memory in every completed row. Counts are identical at 12 and 48 heads.

| Tokens | FP32 accumulation | FP16 + promotion |
| ---: | ---: | ---: |
| 128 | 168 | 195 |
| 512 | 255 | 209 |

The one-warp configuration used local memory in an initial check, so the final recipe uses two warps per CTA.
A register transport is a storage choice in the schedule; the compiled binary must still be checked for spills.
The updated diagnostics make that check part of the ordinary benchmark log.

All four 2,048-token comparisons hit the 360-second process limit after successful lowering and before producing
a timing record. Inductor compiler workers were active during long reference preparation in this sweep. These
failures do not supply a latency or an accuracy verdict for the large case, and are not kernel execution times.
Both 128-token full-prefill probes also hit the process limit. Deterministic lowering alone took 314.87 seconds
with FP32 accumulation and 315.37 seconds with FP16 accumulation; the benchmark worker then started, but neither
probe produced a timing record. Full-prefill performance and correctness remain unqualified by this run.

The complete matrix has eighteen terminal rows: twelve succeeded and six timed out with exit code 124. The failed
row IDs below identify their system records and raw artifacts in the archive. No failed row was selectively rerun.

| Workload | Heads | Tokens | Accumulation | Row ID |
| --- | ---: | ---: | --- | --- |
| Carry update | 12 | 2,048 | FP32 | `92ed89310d08` |
| Carry update | 12 | 2,048 | FP16 | `9da95cf5f6c3` |
| Carry update | 48 | 2,048 | FP32 | `26f8275a3988` |
| Carry update | 48 | 2,048 | FP16 | `5701c9dcd7fd` |
| Full prefill | 12 | 128 | FP32 | `4d50adc4e921` |
| Full prefill | 12 | 128 | FP16 | `d9e982cfc814` |

### Relation to other GDN implementations

The [original V100 comparison](../gdn_kernels/RESULTS.md) includes FLA/Triton and FlashQLA full-prefill measurements
from the pinned 1Cat-vLLM implementation. Those operations include work absent from this carry-only experiment and
use uncaptured CUDA events. Dividing their times by the chunk-state times would not establish a speedup. The direct
comparisons here are eager PyTorch and torch.compile on exactly the same chunk-state function.

### Reproduction and evidence

Run `emmy bench experiments/Qwen3.8-27B/gdn_register --ssh USER@HOST` with the Python and CUDA paths in the recipe
adjusted for the supplied host. The measured machine has four Tesla V100-SXM2-16GB GPUs and a Xeon E5-2680 v4 CPU
with 24 exposed logical CPUs and 219.5 GB RAM. It runs Ubuntu 24.04.1, NVIDIA driver 580.178.04, CUDA toolkit 12.9.86,
Python 3.12.3, PyTorch 2.14.0+cu126, and Transformers 5.14.1. Each row archives its package versions separately.

The kernel table reads register and local-memory attributes from the same cached cubin loader used for execution.
It does not compile a second diagnostic binary. The measured source is
`f40488b96f110c6b447e6d03f1dda8839019f00d`, with clean staged inputs. The invocation started at 19:29:54 UTC on
2026-09-19, with run ID `20260919T192954Z` and local directory `2026-09-19_19-29-54/`.

The final row completed at 20:27:44 UTC. The supplied host was retained; all four GPUs were idle with no device
memory allocated after the run. The original remote checkout was not modified.

[`results_v100x1.tar.gz`](results_v100x1.tar.gz) preserves the complete raw directory under
`2026-09-19_19-29-54/`. It contains eighteen `<variant>_<row_id>.experiment.yaml` system records and eighteen
`<variant>_<row_id>_artifacts.tar.gz` command results, plus the run logs. Each nested archive contains
`artifacts/measurement.log`, `artifacts/status.txt`, `artifacts/requirements.freeze.txt`, and `artifacts/versions.txt`.
The twelve successful rows also contain `artifacts/measurement.json`, with whole-forward latencies, strict
correctness, kernel-source hashes, and the realized schedule pins. The records and raw files are preserved exactly
as the harness produced them, including every timeout.
