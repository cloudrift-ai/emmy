# Frozen Qwen3 operator qualification

This experiment replaces the historical kernel boundaries with one common corpus on V100, A100, and H100.
It tests manually chosen schedules. It does not reproduce the paper's automatic-search claim, its historical kernel
figures, whole-layer latency, model output quality, or serving performance.

Five-seed strict Emmy qualification passes 48/54 operator/GPU pairs: 15/18 on V100, 17/18 on A100, and 16/18 on
H100. Complete three-backend measurements cover 36/54 pairs. These manually selected schedules mostly lose to
Inductor. The evidence supports reporting individual qualified kernels with their missing rows, not a general
speedup or a complete replacement for the paper's existing evaluation.

| Platform | Strict Emmy, five seeds | Complete measurements | Successful experiment records |
| --- | ---: | ---: | ---: |
| V100 | 15/18 | 11/18 | 11/18 |
| A100 | 17/18 | 13/18 | 13/18 |
| H100 | 16/18 | 12/18 | 8/18 |

Four H100 records failed during archive creation after all measurements passed. Those records remain unchanged.

## Corpus and protocol

The corpus contains nine complete operator computations at sequence lengths 1 and 512: 18 targets per GPU and
54 operator/GPU pairs. Dimensions follow Qwen/Qwen3-0.6B at revision
`c1899de289a04d12100db370d81485cdf75e47ca`: hidden width 1024, intermediate width 3072, 16 query heads, eight
key/value heads, and head width 128. The inputs and weights are synthetic FP16 values. They are not checkpoint
weights or activations sampled from model inference.

| Operator | Computation |
| --- | --- |
| RMS normalization | FP32 mean of squared values over width 1024, epsilon 1e-6, FP16 output and weight product |
| Query projection | Linear projection from width 1024 to 2048 |
| Key/value projection | Linear projection from width 1024 to 1024; one shared shape represents K and V |
| Query normalization and RoPE | Width-128 normalization, head transpose, and rotary arithmetic over 16 heads |
| Key normalization and RoPE | The same computation over eight heads |
| Attention | Causal grouped-query scaled dot-product attention |
| Output projection and residual | Linear projection from width 2048 to 1024, followed by FP16 residual addition |
| Gated MLP | SiLU of the gate projection multiplied by the up projection, both from width 1024 to 3072 |
| Down projection and residual | Linear projection from width 3072 to 1024, followed by FP16 residual addition |

Both layer normalizations share one shape. No operator is counted twice for frequency in a layer. Decode attention
here means a sequence of length one, with no existing KV cache. RoPE cosine and sine inputs are synthetic independent
values, rather than angles produced by the checkpoint's position encoding.

The operator definitions are in `operators.sh`. Every target has one frozen frontend program and one frozen Loop IR
program, identical across the three cards. GPU headers and schedule receipts differ. The SiLU targets were frozen
again after correcting their decomposition; the original traces are retained as diagnostics. The archive contains
the program and Loop IR digests and all rejected manual trials.

Schedules were chosen by explicit placement, worker, tile, reduction, and staging pins. No `emmy tune` search ran.
The first strict replay passed 50/54 pairs. The four failed pairs retain unmeasured inventories rather than measured
schedule receipts: V100 output/down residual projections at sequence length 512, and A100/H100 gated MLP at sequence
length one. An inventory is not a qualified golden.

The final recipe requests five fresh processes per target, with input seeds 0–4, ten warmups, 100 timing iterations,
and eager PyTorch, Inductor, and Emmy. It resets the aggregate knob and NVCC overrides and uses only that target's
golden evidence, a fresh tuning database per process, no online prior, and strict evidence. Every repeat is attempted
even after a failure. Each process has a 180-second external limit, a 60-second compiler budget, a two-second kernel
watchdog, and ten seconds for the first iteration. Each command row has an 1800-second limit.

Timings use the CLI's captured, interleaved comparison. Multi-kernel Emmy schedules use whole-program timings.
Compilation is outside the timing window. The deployable compiler uses NVCC's default optimization level, with
`--fmad=false` and precise division/square-root defaults. The final correctness suite uses its separate
`-Xcicc -O1` lane and supplies no performance measurements.

Strict Emmy correctness means every output agrees with eager at `rtol=atol=1e-3`. Torch reduced-precision GEMM
reductions are disabled during strict comparison. Inductor uses
`fullgraph=True, mode="max-autotune-no-cudagraphs"` and the CLI's existing dtype-scaled admission check. These are
different numerical gates; Inductor timing admission is not evidence of the same strict elementwise agreement.
The eager baseline executes the embedded frontend graph. In particular, normalized indexing operations can become
vectorized gathers. Its timings are not timings of an unmodified Transformers layer or native operator source.
Inductor compiles that same embedded graph. This limits interpretation of large eager speedups.

## Compiler corrections

Branch selection used to declare every result as FP32. Selecting two FP16 values then promoted a subsequent product
and lost its required intermediate rounding. Selection now retains the common branch dtype, including in type
propagation to its consumers.

The CUDA compiler no longer enables global fast math implicitly. Implicit multiply-add contraction is disabled to
retain rounding between separate frontend operations. Explicit tensor-core instructions retain their own
accumulation semantics. Obsolete NVRTC option plumbing was removed.

SiLU now computes `x / (1 + exp(-x))` in the input's computation dtype and rounds once to its output dtype. The
previous separately rounded reciprocal and multiplication changed a few FP16 results. Six realization cases were
updated for the changed expression. Three child identities in two FP4 cases were migrated to the corresponding new
kernels, retaining their existing schedules and placement routes; strict decode and realization passed afterwards.

Golden evidence now resolves registered GPU aliases on load. Device-reported names in the files had been compared
literally with canonical registry names, so strict replay rejected every recorded schedule on these cards. All 50
measured targets now strictly decode and replay offline. The failed runs before this correction remain archived.

The full correctness suite exposed another arithmetic rewrite: an invariant division became a reciprocal followed
by multiplication. With precise CUDA arithmetic this moved FP4 quantization across rounding boundaries. Removing
the rewrite restores parity for all three tested generic FP4 projection shapes. Atomic mean reduction still accepts
division by a state-independent count; division by the reduced value remains ineligible.
The split scheduler also follows the reducing operand past a captured scalar provider, matching kernel lowering.
Previously a scalar placed first in the operand list could make a legal softmax split disappear.

An output-sweep regression exposed an unbound coordinate in a selection predicate after a kernel split. Lambda
closure now includes predicate coordinates, and projection inlining keeps operand parameters before coordinates.
Causal attention retains tensor-core eligibility when its score reads its own coordinates. This fixes the split
without changing the measured paper kernels.

The realization corpus was refreshed for direct division. Eight composed or split cases retained their authored
schedules and passed strict decode and realization after updating changed child identities where needed. The
Sinkhorn regression now explicitly requests reciprocal multiplication in its frontend program; its tested Loop IR
and schedules are byte-identical to the previous case. No case was deleted or given a new expected-failure suffix.
Integration with newer main changed five more child identities in the two FP4 cases. Their source programs, pins,
and schedule choices remain unchanged; both cases pass strict decode and realization after that migration.

The paper corpus remains the already frozen Loop IR, including its existing reciprocal expressions. Replaying it
does not retrace it through the changed normalization. A newly traced operator may therefore differ from these
goldens. The embedded programs and loops, rather than a fresh trace of the snippet, define this experiment.

Hardware-specific tests now check their actual prerequisites. TMA requires sm_90+, native FP8 MMA requires sm_89+,
and native block-scaled FP4 requires consumer Blackwell. The available GPUs do not qualify those Blackwell paths.
The serving test fixture contains unmeasured scalar schedules scoped to the live card. It was regenerated for the
same 34 targets after the arithmetic changes made its recorded identities stale. Strict evidence remains enabled.

An unpinned OLMoE compile also exposed overlapping accumulator declarations: a cooperative reduction and a partial
copy of that reduction reused the same names. The existing positional renaming now includes exported accumulators
and accounts for the already bound root. A minimized regression fails at 32 and 64 cooperating threads before the
fix and passes afterwards. The original unpinned OLMoE construction compiles on H100, and its logits and next-token
choice match eager in the existing serving correctness test.

These changes affect numerical behavior and may affect performance. Historical measurements made with implicit
fast math are not interchangeable with the measurements here.

The newer main branch also introduced Qwen3.8-27B-FP8 model goldens. Sixteen rows in that separate model no longer
decode after removal of the invariant-division rewrite. An isolated check passes on main and with the root-selection
fix alone, then fails when only the division change is added. Those model recordings have not been re-recorded or
marked as expected failures. This is a compatibility limit of the correctness change; all paper rows still decode
and retain their measured executable artifacts.

## Numerical findings from manual selection

The strict failures are preserved. No tolerance was relaxed and no failed timing was promoted into a measured
golden receipt. A receipt from the initial seed is not sufficient for five-seed qualification: the V100 decode
gated MLP passes seeds 0 and 2 but fails seeds 1, 3, and 4 in both complete repeated runs. Its recorded seed-0
schedule remains reproducible evidence, but it is not a qualified paper result.

On H100, decode gated MLP seed 0 disagreed at two of 3072 output elements with the cooperative schedule and one
element with the tensor-core schedule. At index 246, eager's gate projection was 0.763671875 and the projection
computed in FP64 then rounded to FP16 was 0.76416015625. The up projection was -54.3125 in both. The resulting
outputs were -28.296875 for eager and -28.328125 for both Emmy and the FP64 diagnostic. This demonstrates a reference
rounding discrepancy at that element; it does not establish an FP64 proof for the complete candidate.

On V100, output projection plus residual at sequence length 512 also has a reference discrepancy at its reported
worst element. At index 121346, eager was -22.96875 and the FP64 projection rounded to FP16 before residual addition
was -23.0, matching Emmy. The down-projection diagnostic is less conclusive: at index 156780, eager and the rounded
FP64 result were 47.25, while the tested Emmy result was 47.1875. Both dot-product ordering and intermediate FP16
rounding matter in this strict composed-operator comparison. These observations do not waive qualification.

A smaller V100 tensor-core schedule with a single shared-memory staging buffer made the gated MLP pass the first
strict replay. The two-buffer compute-fill option is intentionally unavailable on Volta because it requires
asynchronous copies. A fallback reached after an unavailable pin is rejected, even when its output happens to pass.

## Hardware and software

All measurements use one GPU at a time. No application clocks were fixed. The VMs remain running at the user's
request. System records contain live observations, including clocks, power, CPU, memory, driver, and toolkit data.

| Platform | Live GPU | GPU UUID | Torch wheel | Driver | NVCC |
| --- | --- | --- | --- | --- | --- |
| V100 | Tesla V100-SXM2-16GB | GPU-fb047284-9557-a127-0787-70f97e92826a | 2.13.0+cu126 | 580.178.04 | 12.9.86 |
| A100 | NVIDIA A100-SXM4-40GB | GPU-dc5ba098-1a7a-08ea-d5be-fc71f0046c7f | 2.13.0+cu130 | 580.173.02 | 12.9.41 |
| H100 | NVIDIA H100 80GB HBM3 | GPU-4ca72d1f-d341-99d7-752c-c1c8da0526cc | 2.13.0+cu130 | 580.173.02 | 12.9.41 |

Transformers is 5.14.1, CuPy 14.2, cppyy 3.5, and NumPy 2.5.3. Full package freezes are archived. The V100 wheel
includes sm_70 support; the cu130 wheels do not. V100 additionally preloads CUDA 12.9 NVRTC for CuPy operations.
The differing CUDA wheel builds are a cross-platform limitation, even though Torch's public version is the same.

The final measured source revision is `f77a8b8a`, descended from merged compiler revision `e966f65c`. All three remote
checkouts matched the same 1525-file source manifest before execution. Its SHA256 is
`bf6241aa66053c5e6de20a0de05be4acb5234be1f8f38fe26bab0f5fbefe03db`.
Earlier A100 and V100 runs on `f0ff5c11` remain diagnostic evidence rather than the reported final measurements.
The final compiler revision is `e3cad7ff`, including main through `f57df129`. The later corrections and upstream
changes leave all 50 measured targets byte-identical in CUDA source, launch geometry, argument order, zeroing, and
descriptor metadata. The upstream exact-identity feature changes the internal `I_kernel` metadata on 67 emitted
kernels; schedule choices and executable artifacts remain identical. All 84 measured schedule rows strictly decode.
Artifact comparisons and the patch from the measured source are archived. Applying that patch to the archived source
reconstructs all 1526 files of the final source manifest exactly. Its SHA256 is
`a5e45add50a471d0c9cbcc0f32bf11c65e50ca1e1c9b455317138e7d80cf2254`.

## Compiler validation

The final H100 `make test` at `e3cad7ff` completed in 1209.17 seconds: 7277 passed, 701 skipped, and 16 failed.
Every failure is a Qwen3.8-27B-FP8 V100 model-golden decode described above. The full suite is therefore not green.
The 20 serving failures from the preceding integrated run all pass after the fixture refresh. `make lint` passes.
The default suite includes strict model- and hardware-golden decoding at this revision.

A100 focused checks passed 40 tests and skipped eight cases requiring unavailable hardware features. V100 rounding
and Sinkhorn checks passed seven tests. Restoring fast math makes both new rounding regressions fail, providing a
negative control. The partial-reduction lowering checks pass all 20 focused cases. Earlier full-suite logs, failed
attempts, exact commands, source manifests, and the final validation logs remain in the archives.

## Final qualification

The denominator is 18 targets per card. Emmy qualification requires all five strict checks to pass. A complete
comparison additionally requires positive, admitted timings from all three backends in every repeat. The experiment
record also includes command finalization: an Inductor compilation failure or archive error makes the row fail even
when Emmy passes all five seeds. Measurement completeness and experiment-record success are therefore separate.
No geometric mean or complete-platform performance claim is reported for partial coverage.

Tables show microseconds: Emmy median [minimum–maximum], with median Inductor and eager latencies. A dash is missing
evidence, never zero latency. `Emmy only` means strict Emmy correctness passed but the requested comparison failed.
The archives retain each backend's full repeat vector and correctness results.

### V100 SXM2 16GB

Run `20260922T060824Z`, from 2026-09-22 06:08:24 to 06:37:06 UTC. All 18 records are terminal: 11 succeeded and seven
failed. Emmy passes all five seeds on 15/18 targets, with 11 complete comparisons. The two prefill residual
projections have no measured schedule. Decode gated MLP fails three seeds. All four RoPE targets pass Emmy's strict
checks but Inductor compilation fails, with the same length-dependent errors described for A100 below.

| Operator | Seq. | Emmy median [min–max], µs | Inductor, µs | Eager, µs | Qualification |
| --- | ---: | ---: | ---: | ---: | --- |
| RMS normalization | 1 | 2.98 [2.98–2.98] | 2.02 | 150.52 | Complete |
| Query projection | 1 | 8.56 [8.54–8.61] | 5.19 | 6.12 | Complete |
| Key/value projection | 1 | 6.34 [6.33–6.36] | 3.94 | 6.64 | Complete |
| Query normalization and RoPE | 1 | 2.81 [2.80–2.83] | — | 238.27 | Emmy only |
| Key normalization and RoPE | 1 | 2.80 [2.80–2.81] | — | 241.34 | Emmy only |
| Attention | 1 | 2.14 [2.14–2.15] | 2.00 | 35.68 | Complete |
| Output projection and residual | 1 | 9.43 [9.38–9.47] | 6.29 | 8.42 | Complete |
| Gated MLP | 1 | — | — | — | Strict failure, 2/5 pass |
| Down projection and residual | 1 | 12.77 [12.75–12.78] | 6.05 | 10.72 | Complete |
| RMS normalization | 512 | 4.62 [4.61–4.65] | 5.49 | 402.98 | Complete |
| Query projection | 512 | 67.22 [63.87–67.22] | 38.47 | 39.42 | Complete |
| Key/value projection | 512 | 55.64 [55.18–58.91] | 41.51 | 42.12 | Complete |
| Query normalization and RoPE | 512 | 15.36 [15.34–15.41] | — | 1116.84 | Emmy only |
| Key normalization and RoPE | 512 | 9.15 [9.13–9.17] | — | 706.46 | Emmy only |
| Attention | 512 | 215.04 [210.53–216.32] | 269.82 | 486.91 | Complete |
| Output projection and residual | 512 | — | — | — | No measured schedule |
| Gated MLP | 512 | 306.52 [306.18–310.27] | 143.29 | 129.02 | Complete |
| Down projection and residual | 512 | — | — | — | No measured schedule |

Prefill RMS normalization and attention beat Inductor by about 1.19× and 1.25× respectively. Their timing ranges
remain separated. Prefill gated MLP is about 2.14× slower than Inductor and 2.38× slower than eager. The largest
qualified Emmy repeat spread is 6.8% for the prefill key/value projection; most decode measurements vary by under 1%.

The failed decode gated MLP has two discrepant elements on seed 1 and one each on seeds 3 and 4. Reported worst
values are -241.625 versus -241.875, 41.1875 versus 41.125, and -83.6875 versus -83.8125 respectively (Emmy versus
eager). Its measured median of 22.56 µs is diagnostic only and is excluded from the qualified table.

Archive: `results_v100x1.tar.gz`, root `2026-09-22_06-08-24/`.

### A100 SXM4 40GB

Run `20260922T060634Z`, from 2026-09-22 06:06:34 to 06:30:48 UTC. All 18 rows are terminal: 13 succeeded and five
failed. Emmy passes all five seeds on 17/18 targets; 13/18 have complete comparisons. Decode gated MLP has no valid
measured schedule, so strict evidence stops it before a timing is produced. All four RoPE rows pass Emmy correctness
but fail in Inductor. Length-one rows report `ValueError: The argument '((0)) + 64' is not comparable`; length-512
rows exceed Python's recursion depth during compilation.

| Operator | Seq. | Emmy median [min–max], µs | Inductor, µs | Eager, µs | Qualification |
| --- | ---: | ---: | ---: | ---: | --- |
| RMS normalization | 1 | 2.99 [2.98–2.99] | 2.34 | 142.23 | Complete |
| Query projection | 1 | 8.44 [8.41–8.53] | 4.75 | 6.65 | Complete |
| Key/value projection | 1 | 6.67 [6.64–6.71] | 4.09 | 5.54 | Complete |
| Query normalization and RoPE | 1 | 3.00 [3.00–3.00] | — | 228.71 | Emmy only |
| Key normalization and RoPE | 1 | 2.88 [2.88–2.88] | — | 220.65 | Emmy only |
| Attention | 1 | 2.22 [2.22–2.22] | 7.78 | 7.81 | Complete |
| Output projection and residual | 1 | 9.30 [9.16–9.35] | 6.28 | 7.87 | Complete |
| Gated MLP | 1 | — | — | — | No measured schedule |
| Down projection and residual | 1 | 11.82 [11.76–11.86] | 8.26 | 9.97 | Complete |
| RMS normalization | 512 | 3.63 [3.63–3.63] | 6.42 | 230.12 | Complete |
| Query projection | 512 | 21.68 [21.65–22.90] | 19.46 | 15.42 | Complete |
| Key/value projection | 512 | 15.84 [15.82–17.14] | 11.54 | 11.68 | Complete |
| Query normalization and RoPE | 512 | 9.25 [9.25–9.25] | — | 654.97 | Emmy only |
| Key normalization and RoPE | 512 | 6.12 [6.12–6.13] | — | 372.97 | Emmy only |
| Attention | 512 | 153.60 [139.43–177.49] | 42.33 | 43.01 | Complete |
| Output projection and residual | 512 | 28.18 [28.14–34.44] | 20.37 | 20.67 | Complete |
| Gated MLP | 512 | 51.71 [51.54–52.10] | 47.59 | 58.96 | Complete |
| Down projection and residual | 512 | 39.61 [39.56–39.67] | 27.00 | 27.32 | Complete |

Only prefill RMS normalization and the length-one attention computation beat Inductor among the 13 complete
comparisons. The latter has no KV cache and should not be presented as general decode attention. Prefill attention
is about 3.6 times slower than Inductor. Its Emmy timings vary by 27% from minimum to maximum, while both backends
slow together; output projection varies by 22%. Small timing differences should not be overinterpreted.
The large eager normalization/RoPE gaps mainly expose the embedded graph's indexing overhead. These A100 results do
not support a general kernel speedup claim.

Archive: `results_a100x1.tar.gz`, root `2026-09-22_06-06-34/`.

### H100 80GB HBM3

Run `20260922T062537Z`, from 2026-09-22 06:25:37 to 06:41:56 UTC. All 18 records are terminal: eight succeeded and
ten failed. Emmy passes all five seeds on 16/18 targets. Twelve targets have complete measurements, including four
whose final archive command reported that its temporary directory changed while being read. All five repeat
commands succeeded for each of those four targets. Their original records retain `failed`; this report does not
turn them into successful jobs.

| Operator | Seq. | Emmy median [min–max], µs | Inductor, µs | Eager, µs | Qualification |
| --- | ---: | ---: | ---: | ---: | --- |
| RMS normalization | 1 | 2.60 [2.60–2.60] | 2.42 | 122.00 | Complete |
| Query projection | 1 | 5.98 [5.86–6.01] | 3.31 | 4.34 | Complete |
| Key/value projection | 1 | 4.85 [4.81–5.00] | 2.82 | 4.29 | Complete |
| Query normalization and RoPE | 1 | 2.47 [2.47–2.47] | — | 184.78 | Emmy only |
| Key normalization and RoPE | 1 | 2.43 [2.43–2.43] | — | 186.12 | Emmy only |
| Attention | 1 | 1.58 [1.58–1.58] | 5.68 | 5.69 | Complete |
| Output projection and residual | 1 | 6.53 [6.46–6.79] | 3.58 | 7.17 | Complete |
| Gated MLP | 1 | — | — | — | No measured schedule |
| Down projection and residual | 1 | — | — | — | Strict failure, 4/5 pass |
| RMS normalization | 512 | 3.09 [2.75–3.10] | 4.69 | 186.92 | Complete |
| Query projection | 512 | 8.86 [8.86–8.87] | 6.76 | 6.79 | Complete; archive warning |
| Key/value projection | 512 | 6.19 [6.18–6.21] | 5.07 | 5.09 | Complete; archive warning |
| Query normalization and RoPE | 512 | 7.47 [7.47–7.47] | — | 418.15 | Emmy only |
| Key normalization and RoPE | 512 | 4.98 [4.98–4.98] | — | 293.11 | Emmy only |
| Attention | 512 | 17.67 [17.65–17.71] | 11.65 | 11.88 | Complete |
| Output projection and residual | 512 | 11.72 [11.71–11.77] | 8.12 | 9.09 | Complete; archive warning |
| Gated MLP | 512 | 28.49 [28.08–28.69] | 18.29 | 20.85 | Complete |
| Down projection and residual | 512 | 16.20 [16.16–16.23] | 11.52 | 11.66 | Complete; archive warning |

Again, only prefill RMS normalization and length-one attention beat Inductor among the complete measurements.
Prefill attention is about 1.52× slower and gated MLP about 1.56× slower. Prefill RMS normalization varies by 12.6%
across repeats, but its slowest repeat still beats Inductor's fastest. Most other prefill kernels vary by under 1%;
gated MLP varies by 2.2%.

The down-projection residual at length one fails seed 3 at one of 1024 elements: index 928 is -2.08203125 in Emmy
and -2.0859375 in eager. A separate FP64 projection, rounded to FP16 before residual addition, agrees with eager
at that element and passes the eager comparison over the whole output. This candidate remains unqualified; its
8.19 µs median is diagnostic only. Decode gated MLP has no measured receipt. The four RoPE targets fail Inductor
compilation while all five Emmy checks pass.

The archive race affected only transient compiler scratch directories. The recipe now excludes its temporary and
cubin-cache directories from row archives. The measured recipe, original result archives, warnings, and separately
downloaded measurement files are retained. No experiment record was edited and no failed row was selectively rerun.

Archive: `results_h100x1.tar.gz`, root `2026-09-22_06-25-37/`.

## Reproduction

Run `emmy bench experiments/golden-bench-2026/kernels_frozen --local --filter card=v100 --no-teardown` on the matching
GPU, substituting `a100` or `h100` for the other platforms. The recipe pins each Torch wheel and replays the committed
golden files. A failed inventory must remain a failed row until a valid measured schedule has been produced.

For the exact measured compiler, use revision `f77a8b8a` or extract `reproduction/source-f77a8b8a.tar.gz` from a
platform archive. Its `reproduction/recipe.yaml` preserves the measured command, including the archive race described
above. The current recipe changes only that archive finalization. To reconstruct the final compiler, apply
`reproduction/compiler-followup.patch` to the extracted source and verify `final7-source-files.sha256.json`.
The recorded software freezes are part of the protocol; reinstalling newer dependencies is a different experiment.

Every platform archive contains its latest timestamped directory and 18 original system-only experiment records at
that directory's root. Record status was never edited. Relative to that root, the retained members are:

| Members | Evidence |
| --- | --- |
| `*.experiment.yaml` | Original system, provenance, execution, and terminal status records |
| `*_artifacts.tar.gz` | The 18 declared command-result archives, preserved unchanged |
| `<row>/working.yaml`, `repeat-*.log`, `verification/repeat-*` | Exact golden, process logs, structured measurements, and exit status for every repeat |
| `<row>/requirements.freeze.txt` | Per-command software freeze |
| `reproduction/` | Measured source, integrated patch, recipes, goldens, manifests, comparisons, and per-repeat measurement index |
| `diagnostics/manual-trials/` | Accepted and rejected manual proposals, working goldens, logs, and tuning databases |
| `diagnostics/host-evidence/` | Host setup, compiler tests, negative controls, numerical diagnostics, and command status files |

The final runs contain 270 repeat status files and 250 structured measurement files. The 20 missing measurements
belong to the four unmeasured inventories and remain absent. All 898 critical files checked inside the original row
archives match their separately downloaded copies byte for byte. The outer platform archives preserve both copies.
`SHA256SUMS` records the digests of the three platform archives.

## Workflow findings

Manual scheduling exposed unavailable staging pins, incomplete numerical agreement, and reference arithmetic
differences. Exact-pin checks prevented fallback timings from being accepted under the requested schedule name.
The CLI's structured JSON, named realizations, and recorded child receipts were sufficient for measurement.

Future work should make rejected staging choices explain their semantic reason before fallback, and separate a
high-precision correctness oracle from agreement with an implementation used for performance comparison. Any such
change needs its own declared protocol and evidence; it must not retroactively turn these failures into passes.
