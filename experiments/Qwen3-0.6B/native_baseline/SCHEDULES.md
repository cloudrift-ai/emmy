# Qwen3 serving schedule qualification

The follow-up to PR #835 separates kernel correctness from complete-model serving. Explicit placement cuts make
all eight distinct pre/post-attention programs executable in isolation. This is synthetic-input evidence; it does
not establish checkpoint generation parity or a serving speedup.

This qualification describes revision `2144b15a`. The current golden retains 139 compatible records; obsolete
kernel sets were removed after compiler normalization. See [RESULTS.md](RESULTS.md) for the current coverage limits.

## Protocol

The target is one RTX 4080 (`sm_89`), Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, FP16, decode width 16, single-token tier enabled, prefill width 256,
activation capacity 1,024, and maximum 256 batched tokens. The serving config generates eight structural programs
and four realization bindings per program: 1, 16, 256, and dynamic. The complete matrix is preserved.

Measurements use deployable NVCC optimization, five warmups, twenty iterations, captured whole-forward CUDA-event
latency, and direct Emmy-versus-eager checks at `rtol=atol=1e-3`. Strict comparisons use full-width Torch reductions
for both output checking and reference timing. These reference timings are not interchangeable with the previous
non-strict measurements. The desktop shares the GPU; small differences are not treated as improvements.

## Correctness fixes

The 256-token post-attention program initially differed from eager in 68 of 262,144 outputs, with maximum absolute
error 0.00390625. Disabling Torch's reduced-precision intermediate reductions reduced that error to 0.00003052 and
passed the same strict tolerance. The worker now scopes this precision setting to strict comparison jobs and
restores both the precision and independent split-K settings afterward, including on exceptions.

The single-token pre-attention query projection exposed a separate compiler error. A vector multiplied by weights
with head and channel axes was classified as a matrix multiply across those two weight axes. Both scalar matrix
tiling and tensor-core tiling could then produce incorrect query outputs. Matrix classification now checks operand
axis ownership; this case keeps its per-cell reduction schedules. The corrected full pre-attention program passes
strict comparison, with maximum absolute error 0.0001221 in the first fresh probe.

The second random seed exposed another error in single-token post-attention: scalar matrix-vector code multiplied
FP16 operands in FP16 before widening the rounded product into its FP32 accumulator. One of 1,024 final outputs
exceeded the unchanged strict tolerance. Disabling split-K and selecting cuBLASLt did not remove it; Inductor passed
the same reference check. Per-launch inspection identified the premature product rounding. Matmul decomposition now
declares FP16/BF16 products at FP32, and the NumPy interpreter honors a declared wider floating result before
computing. A small regression uses individually overflowing products whose wide dot product is finite. The failing
single-token probe then passed, with mean absolute error falling from 0.000159 to 0.000001907. Old recorded programs
retain their old computation, so qualification restarts from a fresh trace rather than reusing their timings.

## Search and placement findings

An equal-budget pilot compared agent proposals plus MCTS against MCTS alone: four candidate slots, patience two,
seed zero, empty databases and priors, separate cold cubin caches, a two-second first-iteration watchdog, and a
240-second outer limit per arm. Both reached the outer limit before completing a final search result. The hybrid
arm measured seven-cut and nine-cut post-attention proposals at 323.91 and 84.51 microseconds respectively. Those
are search-ranking measurements, not fresh whole-forward deployment timings. The MCTS-only database includes
child-kernel successes, which cannot be counted as successful whole-program candidates. No hybrid-versus-MCTS
speedup is established by this incomplete pilot.

The seven cuts used in the previous investigation were selected reduction boundaries, not every available boundary.
Cutting two additional normalized-input boundaries improved the width-16 post-attention candidate. For pre-attention,
cutting the two query/key output boundaries avoids a serial output sweep: the representative width-16 measurement
fell from about 2,188 to 95.84 microseconds. Larger pre-attention shapes remain slow and need further schedule work.

Exploratory compilation used a one-second prior-pricing budget and explicit placement and schedule pins. Deployment
qualification must replay recorded evidence without those overrides. No timing is copied between bindings or GPUs.

The retained node diagnostics compare the two prior components within each arm, not between unequal target sets:

| Arm | Prior | Spearman correlation | Median top-choice regret | Worst top-choice regret |
| --- | --- | --- | --- | --- |
| Hybrid | Offline | unavailable | 1.00× (5 pools) | 1.00× |
| Hybrid | Online | unavailable | 1.00× (5 pools) | 1.03× |
| MCTS only | Offline | −0.04 (2 pools) | 1.26× (6 pools) | 1.93× |
| MCTS only | Online | +0.93 (2 pools) | 1.00× (6 pools) | 1.00× |

The hybrid node store has 206 rows in 17 groups; 12 failed benchmarks and 168 non-leaf rows are excluded before
scoring. The MCTS-only store has 169 rows in ten groups; 13 failures and 126 non-leaf rows are excluded. These small,
partial pools do not establish whole-target search quality. The online diagnostics describe the fitted checkpoints
following each search, not held-out predictive performance.

## Qualification workflow

The release audit's preliminary reproduction originally used the default hardware golden even when the command
named a model golden. That redundant diagnostic also compiled the whole program once per child receipt. The release
audit now uses its strict offer decode and serving-matrix compile directly; the general reproduction table remains
available in prior evaluation. The existing regression observes every pipeline call and verifies that serving uses
only the selected records with strict evidence enabled, restoring the scope afterward.

The main workflow costs were broad fused candidates that hit the kernel watchdog, expensive unmeasured prior
pricing, and placement proposals that cut reductions but left a serial query/key output sweep. Unsupported tensor
pins failed explicitly rather than silently falling back. Keeping the complete config-derived inventory, recording
successful greedy kernel sets, and replaying from an empty database make these failures distinguishable from a
missing deployment schedule. A future search improvement should expose the output-owning placement cuts efficiently;
adding another model-specific benchmark script would not address that compiler problem.

## Fresh qualification — 2026-09-19 UTC

The corrected inventory was traced and measured at `f6186c4f`, then replayed at `cf031040` (documentation changes
only). It contains 333 measured rows for eight programs and 32 bindings. Seed-one collection and a fresh seed-zero
replay both pass every strict comparison. Replay uses an empty tune database and online checkpoint, strict measured
evidence, and no environment placement or prior-pricing overrides. All 333 rows decode, and the release audit
compiles all eight serving programs from this golden alone. This qualifies execution and synthetic-input accuracy;
it does not make every recorded schedule a good deployment choice.

The seed-zero captured whole-forward timings below come from `accurate-replay0/*.json` in
[`tuning_rtx4080x1.tar.gz`](tuning_rtx4080x1.tar.gz). The static programs retain all four config-derived bindings;
the ranges show those separate measurements, not four different static shapes.

| Program | Binding | Emmy µs | Eager µs |
| --- | --- | ---: | ---: |
| Post-attention, symbolic | 1 | 252.93 | 105.46 |
| Post-attention, symbolic | 16 | 152.75 | 93.06 |
| Post-attention, symbolic | 256 | 386.05 | 181.40 |
| Post-attention, symbolic | dynamic | 699.07 | 273.78 |
| Pre-attention, symbolic | 1 | 16,522.85 | 218.46 |
| Pre-attention, symbolic | 16 | 2,470.85 | 236.16 |
| Pre-attention, symbolic | 256 | 6,881.28 | 357.91 |
| Pre-attention, symbolic | dynamic | 9,088.00 | 551.79 |
| Post-attention, static 1 | all four | 252.83–253.10 | 101.32–105.46 |
| Post-attention, static 16 | all four | 152.41–152.75 | 93.04–93.10 |
| Post-attention, static 256 | all four | 384.83–386.56 | 180.81–181.54 |
| Pre-attention, static 1 | all four | 224.00–224.18 | 208.37–208.51 |
| Pre-attention, static 16 | all four | 84.69–84.77 | 221.22–221.24 |
| Pre-attention, static 256 | all four | 1,572.70–1,577.98 | 350.84–351.67 |

Final CPU validation exposed two scheduling interactions: chunked attention must choose its multiplicand type from
the streamed value, and a grouped-matvec refusal must reject two axes owned by the same operand without rejecting
valid transposed or broadcast pairs. Those checks were corrected after the initial GPU measurements. A final
seed-zero replay at `2144b15a` passes all 32 targets at the same strict tolerance, with empty tuning state and no
placement overrides. All 333 rows decode and all eight serving programs compile from the selected evidence alone.
Final raw results are `final-replay/*.json`, `final-replay.log`, and `final-live-audit.log` in the tuning archive.

Only static width-16 pre-attention is faster than this eager reference. Symbolic pre-attention remains especially
slow. The explicit serial reduction choices that made the candidates executable are not an efficient general
schedule policy. Further tuning must retain the fixed arithmetic and unchanged tolerances.

## Bounded serving startup

At `00d1914a` (a comment-only follow-up), the existing serving CLI booted Qwen3-0.6B with the corrected golden,
strict evidence, decode width 16, the single-token tier, capacity 256, and the baseline recipe's graph settings.
With a 180-second health deadline, it reached readiness and served both 32-input/16-output-token requests. The
CLI then shut down the server normally. Raw evidence is `startup-smoke.log` and `startup-smoke/server.log` in the
tuning archive. No generated text was retained by this diagnostic, so it does not establish checkpoint text parity.

The diagnostic reports 8.63 ms mean time per output token. It used two requests, no explicit warmup or greedy
sampling, and no repeats. Do not compare that number directly with the previous fixed-length, repeated stock
baseline or treat it as the full requested comparison. The later complete run at `2144b15a` is now reported in
[RESULTS.md](RESULTS.md): all four configurations start and complete every workload. All produce identical greedy
completion text for both fixed prompts. This bounded check does not establish general logit or task-quality parity.
Emmy remains slower than stock across the measured matrix; no serving speedup is claimed.

## System and retained evidence

One NVIDIA GeForce RTX 4080, 16,376 MiB, `sm_89`; Intel Core i9-14900K; Ubuntu 24.04.5; driver 595.91.07;
NVCC 13.3.73; cuBLAS 13.6.0.2; Torch 2.11.0+cu130; vLLM 0.23.0; Transformers 5.14.1. No cloud machine was rented.
The initial GPU session respected the user's two-hour allowance. The user removed that limit before final
revalidation and the complete serving run. Hostnames, account names, private addresses, paths, and archive ownership
are removed from the published evidence; hardware and software specifications remain.

The archive retains the exact qualified golden, serving config, seed-one commands and result JSON/logs, seed-zero
replay, release audit, failed pre-fix probes, and the two search pilots' logs, records, and prior diagnostics.
The checked-in experimental golden is the qualified input, not a recommended serving recipe. Search timings and
old program identities are retained only as explicitly historical diagnostics, separate from the fresh replay.

## Final validation

After the scheduling refinements and corpus refresh, the full suite with GPU access disabled reports 4,918 passed,
1,021 skipped, and nine expected failures. Lint and the updated recipe's dry run pass. Corpus expectations were not
weakened. The stored identities and derived loops reflect the wider products; three cases also needed child-identity
updates. Obsolete duplicate receipts in the two quantized projection cases were removed while preserving their
compiled kernel sets and schedule choices.

The model-golden gate reports seven passing files and four failing files. The same 67 rows fail on base `cab3b735`;
no model golden was re-recorded or exempted to hide these failures. The full failure-name list was compared for the
46-row EXL3 failure; the other files list every failing row in their test output.

| Model golden | Failing rows on base | Failing rows on this branch |
| --- | ---: | ---: |
| Gemma-4-12B-it, RTX 5090 | 6 / 352 | 6 / 352 |
| Gemma-4-12B-it, RTX 4090 | 13 / 240 | 13 / 240 |
| Qwen3.8-27B-EXL3, V100 | 46 / 154 | 46 / 154 |
| Qwen3.8-27B-GPTQ-Int4, V100 | 2 / 50 | 2 / 50 |

The final GPU-enabled full suite reports 5,162 passed, 739 skipped, 21 expected failures, and 26 failures. The same
26 tests fail on base `cab3b735`; a seven-file comparison reports 66 passed and one skipped alongside those failures.
Five failures request TMA on `sm_89`, one requires a native FP4 cell unavailable on this card, and twenty lack
measured serving-test evidence. No new failing test is introduced, and no test expectation was weakened.

The tuning archive includes CPU and GPU test logs, the base comparisons, final replay and release audit, lint,
model-golden results, and structural profile interval data. The full serving archive contains all four successful
row records and complete workload evidence. Efficient general schedules and broader generation parity remain open;
execution qualification and the requested bounded serving comparison are complete.
