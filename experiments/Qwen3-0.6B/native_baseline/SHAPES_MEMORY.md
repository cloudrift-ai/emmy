# Shape and memory evidence

The intended comparison is padded decode width 16, a single-token decode tier, and reduced activation capacity,
using the existing integration first. The Emmy runs did not produce usable serving measurements before their
readiness deadlines. There is therefore no measured padding saving, recovered KV capacity, or native-runtime memory
advantage. Preserve this as an unresolved comparison rather than infer a gain from configured widths.

## Completed stock baseline

The exact matrix, environment, and outcomes are in [RESULTS.md](RESULTS.md). Raw files are in
[results_rtx4080x1.tar.gz](results_rtx4080x1.tar.gz), under
`2026-09-16_08-00-39/rtx4080x1_c1024_lstock_m1_f5b4395d2139/`.
Each table row uses `c<concurrency>_i<input>_r{1,2,3}.json`; values are ranges across the three repeats.
Every repeat completed eight requests with 64 generated tokens per request.

| Concurrency | Input tokens | Median TPOT range, ms | Median TTFT range, ms | Output tokens/s range |
| --- | ---: | ---: | ---: | ---: |
| 1 | 32 | 2.278–2.282 | 6.894–8.084 | 420.9–426.0 |
| 1 | 256 | 2.304–2.312 | 10.076–11.008 | 409.2–411.7 |
| 1 | 1024 | 2.465–2.485 | 26.920–28.302 | 345.7–350.1 |
| 4 | 32 | 2.486–2.499 | 8.831–10.811 | 1513.9–1521.0 |
| 4 | 256 | 2.648–2.665 | 23.222–25.359 | 1308.8–1322.1 |
| 4 | 1024 | 3.540–3.591 | 42.298–49.925 | 883.3–894.7 |

The separate mixed-length case (`mixed.json`) completed eight requests with 3,434 total input tokens and 512 output
tokens. It measured median TPOT 2.869 ms, median TTFT 25.374 ms, and 1,175.8 output tokens/s. This is one observation,
not a repeated comparison. Partial-prefill chunk occupancy was not instrumented, so requested lengths alone do not
establish how many useful or padded rows each internal execution processed.

`server.log` reports 7.96 GiB available for KV cache, 74,480 cache tokens, and a theoretical 18.18 requests at the
4,096-token context limit. The actual scheduler limit in the recipe is four sequences; the theoretical capacity is
not an admission measurement. Graph capture reports 0.08 GiB. `memory.txt` records 10,188 MiB device memory used out
of 16,376 MiB after readiness; the desktop also uses this card, so that observation is not isolated process memory.

## Missing evidence

No successful Emmy lane reached the post-readiness memory snapshot. Activation, scratch, and KV bytes therefore
cannot be compared across widths or capacities. Summing pack buffer sizes would count per-layer and potentially
shared allocations incorrectly and would not substitute for an allocation measurement. The standalone runtime's
initial independent buffers also do not reproduce the Python dispatcher's liveness-based scratch reuse.

A rerun needs a working, bounded serving initialization path and stable source for the whole matrix. It should retain
actual scheduled useful/padded rows, allocation categories, admission outcomes, and mixed-length repeats. Only then
can it identify which shape or memory improvement needs a new scheduler rather than the current integration.
