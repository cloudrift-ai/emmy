# Frozen Qwen3 operator qualification

This experiment qualifies manually chosen schedules for the same 18 frozen Qwen3-0.6B operator computations on
V100, A100, and H100. It does not reproduce the paper's historical kernel boundaries or automatic-search claim.
It measures neither whole-layer latency nor model output quality nor serving performance.

After these measurements, `FAST_MATH` became enabled by default and now controls NVCC `--use_fast_math` as well as
compiler precision policies. The results below retain their recorded precise settings and do not qualify that new
default. These goldens explicitly pin `FAST_MATH: false`; the recipe now supplies `--nvcc-flags=--fmad=false` as a
custom flag to preserve the measured arithmetic settings. Use the archived measured revisions for exact reproduction.
Disabling contraction alone leaves other fast-math transformations enabled when the umbrella is true.

The second pass qualifies 49/54 operator/GPU pairs: 15 on V100, 17 on A100, and 17 on H100. All 49 now have complete
three-backend measurements, up from 36 in the first pass. Thirteen schedules change. A100 prefill attention improves
from 153.60 to 32.77 µs, and V100 attention from 215.04 to 135.68 µs. H100 decode down projection gains five-seed
qualification. Five targets remain unqualified; this is not a complete replacement for the paper's evaluation.

| Platform | Strict Emmy, five seeds | Complete measurements | Successful records | Wins over Inductor |
| --- | ---: | ---: | ---: | ---: |
| V100 | 15/18 | 15/18 | 15/18 | 6/15 complete |
| A100 | 17/18 | 17/18 | 17/18 | 9/17 complete |
| H100 | 17/18 | 17/18 | 17/18 | 8/17 complete |

Most qualified comparisons still lose to Inductor: 23/49 win. The gains are concentrated in decode projections and
the revised V100/A100 attention schedules. Reference implementation changes also alter the comparisons, as explained
below. The full first-pass archives and report are preserved inside each new platform archive.

## Corpus and measurement protocol

There are nine operators at sequence lengths 1 and 512, giving 18 targets per card and 54 operator/GPU pairs.
Dimensions follow Qwen/Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`: hidden width 1024,
intermediate width 3072, 16 query heads, eight key/value heads, and head width 128. Inputs and weights are synthetic
FP16 values, rather than checkpoint weights or sampled activations.

| Operator | Computation |
| --- | --- |
| RMS normalization | FP32 mean of squares over width 1024, epsilon 1e-6, FP16 output and weight product |
| Query projection | Linear projection from width 1024 to 2048 |
| Key/value projection | Linear projection from width 1024 to 1024; one shape represents K and V |
| Query normalization and RoPE | Width-128 normalization and rotary arithmetic over 16 heads |
| Key normalization and RoPE | The same computation over eight heads |
| Attention | Causal grouped-query scaled dot-product attention |
| Output projection and residual | Linear projection from width 2048 to 1024, then FP16 residual addition |
| Gated MLP | SiLU of the gate projection times the up projection, both from width 1024 to 3072 |
| Down projection and residual | Linear projection from width 3072 to 1024, then FP16 residual addition |

Both layer normalizations share one shape. Operators are not weighted by their frequency in a layer. Sequence-one
attention has no existing KV cache. RoPE cosine and sine inputs are independent synthetic values, not checkpoint
position encodings. Every target has identical embedded frontend and Loop IR programs across the three cards.
The second pass changes schedules and the compiler, while retaining these frozen programs byte for byte.

No `emmy tune` search ran. Candidates were chosen with explicit placement, worker, tile, reduction, and staging
pins, then measured through `emmy run`. Unavailable pins, slower choices, and strict failures remain diagnostics.
An unmeasured inventory is not a qualified golden. The committed experiment goldens remain schedule inputs; the
repeated experiment measurements are the qualification evidence.

The recipe runs five fresh processes per target, using seeds 0–4, ten warmups, and 100 timing iterations. Every
repeat is attempted, even after a failure. Each process uses only that target's golden evidence, a fresh tuning
database, no online prior, and strict evidence. It has a 180-second external limit, a 60-second compiler budget,
a two-second kernel watchdog, and ten seconds for the first iteration. A command row has an 1800-second limit.

Timings are the CLI's captured, interleaved whole-program measurements of eager PyTorch, Inductor, and Emmy.
Compilation is outside the timing window. The measured compilation used the default NVCC optimization level with
`--fmad=false` and precise division and square root. The full correctness suite uses its separate `-Xcicc -O1` lane
and supplies no performance numbers. No application clocks were fixed.

Strict Emmy correctness requires every output to agree with eager at `rtol=atol=1e-3`, with Torch reduced-precision
GEMM reductions disabled. Inductor uses `fullgraph=True, mode="max-autotune-no-cudagraphs"` and the CLI's existing
dtype-scaled admission check. These are different numerical gates. An admitted Inductor timing does not establish
the same strict elementwise agreement.

The eager and Inductor references execute the embedded frontend graph. They are not an unmodified Transformers
layer. In this pass, simple indexing maps use strided views instead of materialized gathers. This fixes Inductor
compilation of the RoPE targets and changes reference timing and tensor layout. Old and new reference speedup ratios
are therefore not interchangeable. The schedule comparison below compares Emmy latency directly, with this protocol
change disclosed; it is not a controlled measurement of a schedule change alone.

## Optimization findings

Thirteen schedule inputs change. Query, key/value, and output-residual projections at sequence length one use a
single 128-thread cooperative reduction on all three cards, replacing the former split tensor-core programs.
The down-residual projection uses 64 cooperating threads on V100 and 128 on H100. The faster 128-thread V100/A100
down projections failed seed 3; A100's 64- and 256-thread alternatives also failed. A100 retains its prior schedule.

V100 prefill attention computes the full output width in its tiles. A100 prefill attention stages its score
contraction through two asynchronous shared-memory buffers. H100 keeps its existing TMA attention schedule: the
tested smaller reduction chunk did not improve it. These choices follow measured results, rather than assuming that
the same tile or reduction works best on every architecture.

The table compares Emmy medians in microseconds. A ratio is reported only when both schedules passed five seeds.
The H100 down projection's first-pass 8.19 µs timing failed one seed and is excluded from the gain calculation.

| Platform | Changed target | First pass, µs | Second pass, µs | First / second |
| --- | --- | ---: | ---: | ---: |
| V100 | Query projection, seq. 1 | 8.56 | 5.64 | 1.52× |
| V100 | Key/value projection, seq. 1 | 6.34 | 4.17 | 1.52× |
| V100 | Output projection and residual, seq. 1 | 9.43 | 5.72 | 1.65× |
| V100 | Down projection and residual, seq. 1 | 12.77 | 7.01 | 1.82× |
| V100 | Attention, seq. 512 | 215.04 | 135.68 | 1.58× |
| A100 | Query projection, seq. 1 | 8.44 | 4.86 | 1.74× |
| A100 | Key/value projection, seq. 1 | 6.67 | 3.39 | 1.97× |
| A100 | Output projection and residual, seq. 1 | 9.30 | 4.65 | 2.00× |
| A100 | Attention, seq. 512 | 153.60 | 32.77 | 4.69× |
| H100 | Query projection, seq. 1 | 5.98 | 3.03 | 1.97× |
| H100 | Key/value projection, seq. 1 | 4.85 | 2.39 | 2.02× |
| H100 | Output projection and residual, seq. 1 | 6.53 | 3.16 | 2.06× |
| H100 | Down projection and residual, seq. 1 | Unqualified | 3.66 | — |

The manual pass rejected wider gated-MLP tiles. On V100, a wide shared-memory tile took about 7.47 ms and spilled
3768 bytes per thread; a smaller alternative took about 327 µs and still lost to the incumbent. H100's wider tile
also lost. A100's small apparent improvement was not promoted from a single trial. Smaller V100 tensor-core and
split-reduction residual projections still failed strict correctness. No failed timing appears as a qualified gain.

## Compiler changes and validation

The branch was rebased onto main through `f57df1295000df1337370546cf8181cdcd49b8d0`, preserving the tested first-pass
tree. The measured compiler retains the first-pass corrections: FP16 selection and SiLU rounding, precise CUDA
arithmetic, correct invariant division, GPU alias matching, split-root selection, predicate closure, and distinct
partial accumulator names. Their regressions, numerical diagnostics, original measurements, and report remain in the
first-pass archive included with each platform.

Two additional corrections were needed. The Torch reference uses a strided view when an index map is exactly a
broadcast, permutation, diagonal, or constant-zero selection, preserving offsets and noncontiguous storage. Other
index maps retain the clipped-gather implementation. The new rotary-slice regression fails with the old reference
and passes with the fix. The complete reference test file passes 40 tests on H100.

Golden decoding now follows an explicitly recorded kernel set when choosing the route for a child row. This fixes
Qwen3.8-27B-FP8 rows whose parent identity changed after the division correction while their recorded child schedules
remained valid. Those model goldens were not re-recorded. The first attempted decoder correction also broadened
identity checking for standalone rows and caused 253 suite failures. That unrelated contract change was removed;
the final change retains the existing receipt checks and only supplies the explicit route. Focused route, receipt,
and corpus tests pass, as do the selected Qwen and DeepSeek model rows.

The final H100 `make test` at `3271b623` passes: 7299 passed, 701 skipped, and 58 warnings in 1190.42 seconds.
`make lint` passes, including formatting of 831 files. The default suite includes strict model- and hardware-golden
decoding. All 62 measured rows in the paper goldens also pass a separate strict decode at the final golden revision.
Between the measured revisions, only the V100 schedule input changes; compiler and test code remain identical.

The final suite closes the first pass's 16 Qwen model-golden failures. An intermediate run was stopped after a
routing-row ownership failure was reproduced; the final regression checks both the child and the route owner.
Failed and interrupted validation logs remain diagnostic evidence. No result from those runs is substituted for
the final gate. At the measured revision, the core diff against main is 105 added and 174 removed lines, net −69.

The frozen paper Loop IR retains its existing reciprocal expressions. Replaying a golden does not retrace it
through the revised normalization. A newly traced operator may differ; the embedded programs define this experiment.
No tolerance was relaxed, no realization case was deleted to hide a failure, and no new expected failure was added.

## Hardware and source provenance

The same three caller-owned machines were used for both passes. All remain running. System records contain live
clocks, power, temperature, CPU, memory, driver, and filesystem observations.

| Platform | Live GPU | GPU UUID | Torch wheel | Driver | NVCC |
| --- | --- | --- | --- | --- | --- |
| V100 | Tesla V100-SXM2-16GB | GPU-fb047284-9557-a127-0787-70f97e92826a | 2.13.0+cu126 | 580.178.04 | 12.9.86 |
| A100 | NVIDIA A100-SXM4-40GB | GPU-dc5ba098-1a7a-08ea-d5be-fc71f0046c7f | 2.13.0+cu130 | 580.173.02 | 12.9.41 |
| H100 | NVIDIA H100 80GB HBM3 | GPU-4ca72d1f-d341-99d7-752c-c1c8da0526cc | 2.13.0+cu130 | 580.173.02 | 12.9.41 |

Python is 3.12.3, Transformers 5.14.1, CuPy 14.2, cppyy 3.5, and NumPy 2.5.3. Per-row package freezes are archived.
V100 requires the cu126 wheel for sm_70 support and preloads CUDA 12.9 NVRTC. The differing Torch CUDA builds limit
cross-platform comparisons despite their shared public version.

A100 and H100 were measured at `3271b62337ff729282b835e700f6f586de120703`. V100 was measured at `5eae5b6b`;
the only intervening change is its 64-thread down-projection golden. Compiler code and measurement recipes are
identical. Each checkout matched its 1493-file measured-source manifest before execution, and every final system
record reports a clean source tree. The source archives and manifests are retained under `reproduction/`.

| Source snapshot | Manifest SHA256 |
| --- | --- |
| A100/H100 measured source, `source-3271b623-measured.tar.gz` | `401b2200e91e3b4436ab1438f4b495c0db12d940832d16520e5f1007ed60df4e` |
| V100 measured source, `source-5eae5b6b-measured.tar.gz` | `79ba1f5b626e9ec05f998e484b0eac778eb507e2114d72f894f74495f7d1eea6` |
| Compiler-suite source, `source-3271b623.tar.gz` | `bf9241f960a16a2b934f03e55b10fdc2fd7dfb9dbc7024518e29f8d0e7410573` |

The suite snapshot hydrates the tracked offline-prior LFS payload; measured snapshots retain its pointer. Strict
benchmark replay uses the named golden and fresh database, so that prior payload supplies no measurement evidence.
Earlier provisional V100/A100 runs at `0ed6bfe7` retain the rejected 128-thread down projections. They are archived
as earlier complete runs, rather than combined with the final measurements. Staging failures before measurement
also remain recorded. No rows were selectively rerun to replace a failed result within an existing run.

## Repeated qualification

The denominator is 18 targets per card. Qualification requires all five strict Emmy checks. A complete comparison
also requires positive admitted timings from all three backends in every repeat. Experiment-record success includes
command finalization as well. The tables give microseconds: Emmy median [minimum–maximum], and median Inductor and
eager latency. Missing evidence is shown as a dash, never zero. Failed candidates are excluded from qualified timing
comparisons. Full vectors and correctness results remain in the archives.

### V100 SXM2 16GB

Run `20260922T094023Z`, from 2026-09-22 09:40:24 to 10:04:16 UTC. All 18 records are terminal: 15 succeeded and
three failed. All successful targets pass strict Emmy correctness and have complete comparisons. Decode gated MLP
still fails seeds 1, 3, and 4. The two prefill residual projections retain unmeasured inventories.

| Operator | Seq. | Emmy median [min–max], µs | Inductor, µs | Eager, µs | Qualification |
| --- | ---: | ---: | ---: | ---: | --- |
| RMS normalization | 1 | 2.98 [2.98–2.98] | 2.04 | 29.12 | Complete |
| Query projection | 1 | 5.64 [5.62–5.96] | 5.47 | 6.20 | Complete |
| Key/value projection | 1 | 4.17 [4.13–4.20] | 3.97 | 6.71 | Complete |
| Query normalization and RoPE | 1 | 2.80 [2.80–2.81] | 2.96 | 42.54 | Complete |
| Key normalization and RoPE | 1 | 2.80 [2.80–2.81] | 2.92 | 42.27 | Complete |
| Attention | 1 | 2.15 [2.14–2.15] | 2.00 | 35.74 | Complete |
| Output projection and residual | 1 | 5.72 [5.70–5.74] | 6.29 | 8.52 | Complete |
| Gated MLP | 1 | — | — | — | Strict failure, 2/5 pass |
| Down projection and residual | 1 | 7.01 [6.99–7.03] | 7.21 | 10.86 | Complete |
| RMS normalization | 512 | 4.46 [4.39–4.53] | 3.40 | 49.08 | Complete |
| Query projection | 512 | 67.29 [63.81–67.29] | 38.55 | 38.99 | Complete |
| Key/value projection | 512 | 58.78 [55.24–58.85] | 42.60 | 43.21 | Complete |
| Query normalization and RoPE | 512 | 14.63 [14.62–14.69] | 19.92 | 115.25 | Complete |
| Key normalization and RoPE | 512 | 8.99 [8.96–9.03] | 6.43 | 74.05 | Complete |
| Attention | 512 | 135.68 [135.02–135.68] | 267.78 | 486.74 | Complete |
| Output projection and residual | 512 | — | — | — | No measured schedule |
| Gated MLP | 512 | 306.18 [305.15–308.57] | 122.06 | 129.15 | Complete |
| Down projection and residual | 512 | — | — | — | No measured schedule |

Emmy wins six of the 15 complete comparisons, with separated repeat ranges. Prefill attention is 1.58× faster than
the first-pass schedule and 1.97× faster than current Inductor. Prefill gated MLP remains about 2.51× slower than
Inductor. The largest Emmy repeat spread is 6.1%, in prefill key/value projection. The decode down-projection gain
over Inductor is small despite the clear improvement over the old Emmy schedule.

The failed decode gated MLP's 22.61 µs median is diagnostic only. Its failures repeat the first-pass pattern: two
discrepant elements in seed 1 and one each in seeds 3 and 4. It is excluded from the qualified timing table.

Archive: `results_v100x1.tar.gz`, root `2026-09-22_09-40-23/`.

### A100 SXM4 40GB

Run `20260922T093632Z`, from 2026-09-22 09:36:32 to 09:56:33 UTC. All 18 records are terminal: 17 succeeded and one
failed. All successful targets pass strict Emmy correctness and have complete comparisons. Decode gated MLP remains
an unmeasured inventory. The previous down-projection schedule passes all five seeds and is retained.

| Operator | Seq. | Emmy median [min–max], µs | Inductor, µs | Eager, µs | Qualification |
| --- | ---: | ---: | ---: | ---: | --- |
| RMS normalization | 1 | 2.99 [2.99–2.99] | 2.33 | 32.32 | Complete |
| Query projection | 1 | 4.86 [4.85–4.87] | 5.39 | 6.53 | Complete |
| Key/value projection | 1 | 3.39 [3.38–3.39] | 4.02 | 5.49 | Complete |
| Query normalization and RoPE | 1 | 3.00 [3.00–3.05] | 4.23 | 47.96 | Complete |
| Key normalization and RoPE | 1 | 2.88 [2.88–2.88] | 4.18 | 46.74 | Complete |
| Attention | 1 | 2.22 [2.22–2.22] | 7.86 | 7.83 | Complete |
| Output projection and residual | 1 | 4.65 [4.63–4.65] | 6.09 | 7.73 | Complete |
| Gated MLP | 1 | — | — | — | No measured schedule |
| Down projection and residual | 1 | 11.72 [11.67–11.85] | 8.29 | 9.97 | Complete |
| RMS normalization | 512 | 3.62 [3.61–3.63] | 3.42 | 44.92 | Complete |
| Query projection | 512 | 21.65 [21.62–25.51] | 19.46 | 15.33 | Complete |
| Key/value projection | 512 | 15.86 [15.84–15.88] | 12.79 | 11.68 | Complete |
| Query normalization and RoPE | 512 | 9.24 [9.24–9.28] | 13.31 | 94.43 | Complete |
| Key normalization and RoPE | 512 | 6.11 [6.10–6.13] | 8.88 | 68.25 | Complete |
| Attention | 512 | 32.77 [32.41–32.86] | 37.53 | 37.93 | Complete |
| Output projection and residual | 512 | 28.18 [28.18–28.48] | 25.22 | 20.71 | Complete |
| Gated MLP | 512 | 51.61 [51.46–52.02] | 45.70 | 58.91 | Complete |
| Down projection and residual | 512 | 39.61 [39.56–39.61] | 27.22 | 27.32 | Complete |

Emmy wins nine of the 17 complete comparisons, with separated repeat ranges. Prefill attention is about 4.69×
faster than the first-pass Emmy schedule and 1.15× faster than current Inductor. Prefill projections and gated MLP
still lose to Inductor, as does RMS normalization with the improved reference. Prefill query projection has an 18%
Emmy repeat spread; small differences should not be generalized from its median. The other Emmy ranges are much
tighter, but these measurements still use unlocked clocks and separate runs.

Archive: `results_a100x1.tar.gz`, root `2026-09-22_09-36-32/`.

### H100 80GB HBM3

Run `20260922T093633Z`, from 2026-09-22 09:36:33 to 09:48:47 UTC. All 18 records are terminal: 17 succeeded and one
failed. All 17 successful targets pass strict Emmy correctness and have complete comparisons. Decode gated MLP has
no measured schedule. The archive-finalization failures from the first pass do not recur.

| Operator | Seq. | Emmy median [min–max], µs | Inductor, µs | Eager, µs | Qualification |
| --- | ---: | ---: | ---: | ---: | --- |
| RMS normalization | 1 | 2.60 [2.60–2.60] | 2.20 | 27.83 | Complete |
| Query projection | 1 | 3.03 [3.03–3.04] | 3.62 | 4.36 | Complete |
| Key/value projection | 1 | 2.39 [2.39–2.40] | 2.86 | 4.38 | Complete |
| Query normalization and RoPE | 1 | 2.48 [2.48–2.48] | 3.84 | 38.99 | Complete |
| Key normalization and RoPE | 1 | 2.44 [2.44–2.45] | 3.74 | 39.32 | Complete |
| Attention | 1 | 1.58 [1.58–1.58] | 5.73 | 5.74 | Complete |
| Output projection and residual | 1 | 3.16 [3.16–3.17] | 4.10 | 7.18 | Complete |
| Gated MLP | 1 | — | — | — | No measured schedule |
| Down projection and residual | 1 | 3.66 [3.65–3.70] | 4.17 | 8.58 | Complete |
| RMS normalization | 512 | 2.93 [2.93–2.94] | 2.51 | 36.42 | Complete |
| Query projection | 512 | 8.87 [8.85–9.13] | 6.86 | 6.88 | Complete |
| Key/value projection | 512 | 6.19 [6.18–6.22] | 5.19 | 5.05 | Complete |
| Query normalization and RoPE | 512 | 7.46 [7.46–7.47] | 6.52 | 66.42 | Complete |
| Key normalization and RoPE | 512 | 4.81 [4.80–4.81] | 6.41 | 50.40 | Complete |
| Attention | 512 | 17.64 [17.63–17.69] | 11.73 | 11.93 | Complete |
| Output projection and residual | 512 | 11.76 [11.71–11.78] | 8.07 | 9.10 | Complete |
| Gated MLP | 512 | 28.56 [28.19–28.67] | 18.17 | 20.78 | Complete |
| Down projection and residual | 512 | 16.23 [16.14–16.33] | 11.44 | 11.76 | Complete |

Emmy beats Inductor in eight of the 17 complete comparisons, with separated repeat ranges in each. Seven wins are
at sequence length one; the other is prefill key normalization and RoPE. Prefill attention remains about 1.50× slower
than Inductor, and gated MLP about 1.57× slower. Prefill RMS normalization now loses to the faster reference. The
largest Emmy repeat spread is 3.2%, in the prefill query projection. These results do not establish a prefill-wide win.

Archive: `results_h100x1.tar.gz`, root `2026-09-22_09-36-33/`.

## Numerical limits

The first pass found both reference discrepancies and candidate errors. For H100 decode gated MLP at element 246,
eager produced -28.296875 while Emmy and an FP64 projection rounded at the frontend's FP16 boundaries produced
-28.328125. For V100 prefill output projection plus residual, the corresponding values at element 121346 were
-22.96875 versus -23.0. V100 prefill down projection was a candidate error at its inspected element: eager and
rounded FP64 gave 47.25, while Emmy gave 47.1875. These are element-level diagnostics, not an FP64 proof of a whole
candidate, and none waives the strict gate.

In the provisional second-pass runs, the 128-thread V100/A100 down projection failed seed 3 at element 928.
The A100 result was -2.0859375 in Emmy versus -2.08203125 in eager. The 64-thread V100 replacement and the retained
A100 schedule pass that seed. The H100 cooperative replacement also closes the earlier seed-3 failure. The final
five-seed runs above, rather than any single passing trial, determine qualification.

## Reproduction and archived evidence

Run `emmy bench experiments/golden-bench-2026/kernels_frozen --local --filter card=v100 --no-teardown` on the exact
matching GPU, substituting `a100` or `h100`. Use the recorded software freezes; newer dependencies define a different
experiment. All complete runs retain every expanded row, including failures. No original experiment record was edited.

The final archives are `results_v100x1.tar.gz`, `results_a100x1.tar.gz`, and `results_h100x1.tar.gz`, with roots
`2026-09-22_09-40-23/`, `2026-09-22_09-36-32/`, and `2026-09-22_09-36-33/` respectively. `SHA256SUMS` gives their
digests. For exact CLI reproduction, use the measured revisions above. The final golden revision `5eae5b6b` has the
same compiler and recipe on every card. Source snapshots preserve the measured files independently of Git history.

Each named archive has its latest timestamped run as its root. Relative to that root, its retained members include:

| Members | Evidence |
| --- | --- |
| `*.experiment.yaml` | The 18 original system, source, execution, and terminal-status records |
| `*_artifacts.tar.gz` | Original declared command-result archives |
| `<row>/working.yaml`, `repeat-*.log`, `verification/repeat-*` | Replayed schedules, logs, structured measurements, and all repeat exit statuses |
| `<row>/requirements.freeze.txt` | Software used for that command |
| `reproduction/` | Exact source snapshots, manifests, commands, recipe, goldens, and measurement index |
| `diagnostics/manual-trials/` | Accepted and rejected proposals, CUDA dumps, logs, and tuning databases |
| `diagnostics/host-evidence/` | Setup, compiler checks, negative controls, statuses, and source verification |
| `diagnostics/compiler-validation/` | Shared final compiler-suite, lint, and decode evidence |
| `history/pass1/` | The complete previous platform archive, original report, and checksums |
| `history/pass2/<timestamp>/` | Earlier full qualification attempts and staging failures, with their original records |

The final runs contain 54 terminal records, 270 repeat status files, and 250 structured measurement files. The 20
missing measurements belong to the four unmeasured inventories. All 1006 critical members checked inside the
original row archives match their separate raw copies. Source archives match their manifests, and the three outer
archives are checked against the assembled raw trees. Records, logs, measurements, and SQLite evidence were scanned
for secrets. The first-pass archive bytes and checksums are preserved unchanged.

These results support individual qualified operator measurements with explicit missing rows. They do not support
a complete-platform speedup claim or a claim that automatic schedule search has been reproduced.
