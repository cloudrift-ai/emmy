# Golden-bench kernel corpus

H100 evidence is consolidated in `results_h100x1.tar.gz`, rooted at `2026-10-02_23-00-41/`. Within that root, each
original bundle has a directory named after its former archive without `.tar.gz`. `ARCHIVE_INDEX.json` records
those directories and the original archive hashes. H100 member paths below are relative to each bundle directory.
Previously replaced recipe snapshots remain in Git history.

## Golden replay recovery (2026-10-09)

The compact normal form in #1106 left all five Qwen3-0.6B prefill goldens without rows or routes. Their freshness
tests were strict expected failures. This recovery restores measured routes on A100 40GB, H100 80GB, RTX 5090 and V100
SXM2 16GB, then removes those four marks. The RTX 4090 file remains empty and marked: that card was unavailable.

Composed cuts now leave their children open for later decisions, and replay follows the fresh children when their
identities change. The reproductions also exposed scalar accumulators sharing state across sibling loops, dropped
loop-carried dependencies, softmax operands compared by argument position, and attention tiles inheriting the wrong
query geometry. The fixes preserve accumulator scope, follow dependencies to a fixpoint, compare actual coordinate
values, and use the existing fragment projection for nested attention. Loop fusion stays maximal. Recording matches
the full decision set, ignoring only declared OFF defaults, so a measured schedule fills its existing proposal.
Compatible routes share one validation replay; conflicting alternatives still replay separately and every route is
checked. The H100 prefill's six routes validate in one GPU-free replay. A whole-target latency row with no schedule or
kernel measurement no longer creates an automatic pinned comparison; it supplies no recorded schedule. Explicit
proposals and measured schedules retain their comparison behavior.

The prior reproduction gate also found six recorded cuts whose alternate spelling explicitly cuts the final output
instead of leaving it as the remainder. Both arms produce the same exact child kernels. The shared placement ballot
now recognizes their identical kernel sets, including multiplicity, within the same fork. Changed or unknown child
identities do not match. This corrects the training and reproduction labels without changing the cut candidates,
recorded measurements or gate tolerance. The label-only re-export, before the later reduction-coordinate correction,
keeps all 2,781 candidate feature matrices and pool metadata unchanged. Labels change in 59 groups: 92 equivalent
complete-arm labels are added and 11 subset labels are removed under the existing preference for a complete recorded
arm.

The scope correction also changes the operand paths of two Qwen3.8 V100 cuts. Their updated paths mint the same 10 and
nine child identities, preserving their four measured rows and all proposals. A DeepSeek V100 cut returns the same 11
children in a different order. Replay matches unchanged children by exact identity before pairing changed kernels, so
a permutation preserves measurements while a changed body still loses its old timing. Nested hand pins derive their
producer names from fresh replay too: an unchanged kernel can have a new generated name, and its historical name must
not address the next compile. The same scope correction moves cut addresses in the LoRA experiment and the Sinkhorn
realization case. Their remapped arms mint identical children and preserve every kernel, schedule and case
expectation. Two structural checks now follow declared accumulator states instead of assuming their old generated
numbers. Final review also catches an enclosing accumulator seed lost through an unseeded strided loop. The shared
scope walk now treats ordinary and strided loops alike; 26 focused checks pass. None of the 2,225 stored kernels
contains a strided loop, and that loop type materializes after both prior pipelines, so this fix changes no recorded
identity, feature or measurement.

### Exact-card verification

These are stored-program golden replays with generated inputs, FP16, standard math and deployable O3. Each proof
starts with an empty tune DB, takes an unpinned pick under strict evidence, and compares against eager and
`torch.compile` with five warmups and 20 iterations. Emmy passes the strict eager check at rtol=0.001 and atol=0.001.
They qualify compiler routes and measurements; they do not rerun the historical actual-model or serving studies.

| Card | Shape | Emmy, µs | `torch.compile`, µs | Launches |
| --- | --- | ---: | ---: | ---: |
| A100 40GB | s1 | 70.54 | 59.31 | 13 |
| A100 40GB | s512 | 178.18 | 204.33 | 11 |
| H100 80GB | s1 | 38.74 | 40.81 | 13 |
| H100 80GB | s512 | 84.46 | 94.46 | 11 |
| RTX 5090 | s1 | 36.83 | 38.90 | 13 |
| RTX 5090 | s512 | 131.82 | 143.32 | 11 |
| V100 SXM2 16GB | s1 | 53.83 | 66.69 | 8 |
| V100 SXM2 16GB | s512 | 497.66 | 745.76 | 21 |

A100 decode remains slower than `torch.compile`; recording correctness does not establish a performance win. The
unchanged hardware decode latency rows retain 50.29 µs on A100, 30.39 µs on H100, 18.42 µs on RTX 5090 and 66.32 µs on
V100. The first three are faster than this recovery's proofs; their historical performance is not restored here. These
runs are not controlled before/after pairs. In particular, the historical V100 actual-model prefill number uses
different inputs and cannot establish a percentage change here. The table keeps the latest independent replay;
promoted canonical copies also pass fresh lowering and strict evidence selection. Prefill is re-recorded after the
reduction-coordinate correction changes its workspace geometry. The archive retains the earlier proofs as history.
The final V100 recording writes all 20 measured rows, seven routes and the whole-target timing before hitting its
110-second command limit. Its separate fresh-DB replay exits cleanly in 64 seconds, passes strict accuracy and leaves
the recorded file byte-identical. The timeout is retained in the evidence instead of being reported as a clean run.

All 19 missing decode proposals on the four available cards now have measurements. Twelve identical proposals in the
hardware goldens receive those same measurements after matching exact kernel identity, card, bindings, regime and
schedule. The empty A100, H100 and RTX 5090 hardware prefill targets also receive the qualified routes and rows: their
traced programs, root identities, origins and bindings match the experimental files exactly. Those prefill additions
preserve every existing entry. Each new hardware prefill passes freshness and an unpinned strict compile from an empty
DB.

### Repository inventory

The audit reads all 31 loadable hardware, recipe and experimental goldens, including files outside the default
repository evidence. A proposal here has neither kernel measurements nor whole-row latency. A row carrying only
whole-row latency is counted separately. A target has no rows only when none of its descendants along any recorded
route has a row; the CLI's first-parent grouping alone would miss shared descendants and empty targets.

| Inventory | Before | After |
| --- | ---: | ---: |
| Strict expected failures in this experiment | 5 | 1 |
| Rows without measurements or latency | 241 | 363 |
| Rows with whole-row latency only | 132 | 163 |
| Targets without any descendant rows | 39 | 32 |
| Rows with kernel measurements | 2,328 | 2,518 |
| Total rows | 2,701 | 3,044 |

The first recovery leaves 102 proposals already covered by measured alternatives or complete cut routes: 97 Gemma,
three A100 and two Llama LoRA entries. It also identifies 108 uncovered proposals. The expanded qualification targets
63 on available hardware: 25 Qwen3.8 FP8 on V100 SXM2, 16 Qwen3.8 NVFP4 and 16 Gemma on RTX 5090, and six Ministral
standard-math entries on H100. Another 30 require V100 SXM3 32GB, and 15 require RTX 4090. The SXM2 rental cannot
measure an SXM3 row. These counts describe proposals; whole-target checks can find additional rowless children. These
are the initial qualification counts, before invalidating incorrect Ministral measurements. The final inventory
retains 241 proposals covered by measured alternatives or complete parent routes. Another 122 lack that coverage: 36
on available cards, 30 requiring V100 SXM3 32GB and 56 requiring RTX 4090. The available-card count is the 20 FP8 and
16 NVFP4 proposals described below. The proposal total rises because the rounding correction withdraws incorrect
measurements while preserving their schedules. The 32 targets without descendant rows are a structural count within
their files; this audit does not claim that every one lacks matching evidence elsewhere.

### Hardware coverage

Coverage is compared by exact card, arithmetic and cache regime, operation, storage type, shape and schedule family.
These are audit categories; evidence still joins only on exact kernel identity and context. A100 sources already cover
the available families. After the rounding correction, a fresh audit of ten current-format H100 and RTX 5090 sources
finds 67 native-FP8 schedule rows and no remaining valid measurement. The corrected FP16 MMA paths restore model
coverage but do not fill that instruction category; there is no independently measured native-FP8 representative to
copy. The first promotions add the qualified V100 SXM2 prefill route, then three V100 SXM3 representatives from
existing exact-card measurements: embedding, normalization and a low-rank route. Their stored whole-target speedups
over `torch.compile` are 1.53×, 2.96×, 2.45× and 1.50×, respectively. An RTX 5090 expert projection adds one missing
shape at 3.3 µs versus 8.2 µs cuBLAS. These copied historical measurements pass fresh lowering and strict evidence
selection; SXM3 measurements were not rerun on the SXM2 rental.

| Hardware golden | Added representative coverage | Remaining source gap |
| --- | --- | --- |
| A100 40GB | Qualified Qwen prefill | No additional source category missing |
| H100 80GB | Qualified Qwen prefill | Native FP8 has no valid recorded representative after the rounding fix |
| RTX 5090 | Qualified Qwen prefill and OLMoE expert-projection shape | Native FP8 lacks valid records; additional NVFP4 candidates fail full-parent accuracy |
| V100 SXM2 16GB | Qualified Qwen prefill | Qualified FP8 routes are slow; no independent fast representative |
| V100 SXM3 32GB | Embedding, RMSNorm and LoRA prefill | Packed four-bit candidates lack independent timing; other proposals require this exact unavailable card |
| RTX 4090 | None | Qwen prefill and invalidated Ministral measurements require the unavailable card |
| RTX 4080 | None | The two exact-card experimental sources use the retired format |
| RTX PRO 6000 Max-Q | None | No exact-card source adds a missing category |

A child's same-input-greedy comparison is not evidence of a PyTorch speedup. Slow but correct recordings can establish
coverage without earning promotion as a fast representative. The qualified V100 SXM2 FP8 routes are slower than
`torch.compile`, so static and dynamic FP8 remain gaps in the hardware golden's fast representatives. The final NVFP4
retry completes 13 launches after the alignment correction, but fails 25 of 40,960 pad outputs. Its maximum absolute
error is 0.015625, and it is not uniformly closer to the high-precision oracle. Ten isolated original schedules and
three legal alternatives have only same-input-greedy references; six original schedules are unavailable. All 16
original proposals remain unqualified, so none is copied as a fast representative.
The final diagnostic reproduces the same 13 kernels and all 25 pad failures after the reduction-coordinate fix.
The normalized BF16 inputs match eager bit-for-bit. At the failing positions, the FP32 dot products differ from the
same-input FP64 products by at most 2.127e-5, crossing BF16 rounding boundaries. FP64 rounding selects Emmy's result
in seven cases and eager's in 18. This supports accumulation-order amplification; the complete context still fails
the existing FP64 gate, and these diagnostic taps supply no new timing evidence.

### Additional qualification

The Gemma inventory gains 23 measurements: seven original proposals, nine legal alternatives, and seven previously
rowless normalization statistics needed by their routes. All 552 historical measured rows remain unchanged. The
original unsupported or inaccurate schedules remain proposals. Seven full traced-program contexts pass fresh-DB,
unpinned strict evidence and accuracy checks: standard and fast prefill at 32 and 4096 tokens, standard at 2048, fast
at 512, and symbolic fast at a 512-token binding. Standard-2048 and both 4096-token contexts pass through the existing
FP64 comparison; the other four pass ordinary eager tolerance. Symbolic fast is 196.58 µs versus 205.93 µs for
`torch.compile`; the other six contexts are slower. The hardware golden already covers these operation and schedule
families, so these measurements stay in the recipe. These checks do not establish serving-matrix qualification.

Five of the 25 Qwen3.8 FP8 proposals on V100 SXM2 gain measurements after complete static-64 and dynamic-512 parent
checks. Both routes are much slower than `torch.compile`, so none becomes a fast hardware representative. The other 20
proposals remain unqualified: five schedules are unsupported, two isolated targets lack an independent eager
reference, and 13 depend on parent checks that fail. The last convolution checks still find 13 failing outputs at 16
tokens and one at 64 tokens. The 16-token normalization reduction differs by one FP32 ULP, changing two FP16-rounded
inputs; feeding those inputs to the Torch projection removes all 13 out-of-tolerance pad differences. That diagnosis
does not establish the cause of the remaining 64-token error. The accuracy gate and historical measurements remain
unchanged.

Ministral qualification exposed a separate compiler error: a scale multiplication rounded to FP16 or BF16 was moved
after a contraction in FP32. That changed the contraction's operands. The shared product walker now preserves narrow
rounding inside the operand, including invariant-factor rewrites. Focused before/after tests reproduce the error and
pass with the correction. The corrected computation has different identities and features, so old affected timings
cannot remain evidence even where the stored Loop IR itself is unchanged. The archive preserves the old records,
including three H100 prefill recordings made earlier during this recovery; those are superseded, not current proofs.

The correction affects Ministral on H100, RTX 5090 and RTX 4090. Route paths move, but semantic cut remapping keeps
all recorded kernels and schedules available for fresh qualification. The unavailable RTX 4090 retains 181 kernels, 37
routes and 209 rows; 33 measurements and nine whole-target latencies become unmeasured proposals. On RTX 5090, 23
hardware and 50 recipe measurements lose their old timings, together with five hardware and two recipe target
latencies. H100 withdraws 66 measurements and 13 whole-target latencies. These counts include descendants whose fresh
workspace types changed during restamping. H100 and RTX 5090 are re-recorded on their exact cards; unsupported old
schedules remain proposals alongside qualified alternatives.

The corrected RTX 5090 hardware routes pass all eight contexts. The four 4096-token contexts pass through the existing
strict FP64 comparison: candidate maximum error equals eager maximum error, and the candidate is no less accurate
overall. The shorter RTX hardware contexts pass ordinary eager tolerance. All 16 H100 contexts qualify: pre1, pre32
and symbolic pre-attention pass ordinary eager tolerance in both math modes; post1, post32, pre4096, post4096 and
symbolic post-attention use the existing FP64 comparison in both modes. The RTX 5090 recipe qualifies all 16 contexts:
pre4096, post32, post4096 and symbolic post-attention use the FP64 comparison in both math modes; the other eight pass
ordinary eager tolerance. The accuracy gate is unchanged. Correct arithmetic is often slower: H100 post4096 takes
about 33.7–34.2 ms against 3.7–3.8 ms for `torch.compile`, and RTX 5090 post4096 takes about 22 ms against 11 ms. The
former faster timings computed different arithmetic and are not valid performance references. All 16 H100 contexts
also pass fresh frontend lowering against the complete canonical evidence file with empty databases. The literal
latency-only realization command passes after the comparison fix. RTX 5090 passes all 16 canonical recipe compiles
and all eight canonical hardware compiles under strict evidence.

The NVFP4 checks also exposed an unaligned asynchronous shared-memory copy in a fused multi-output GEMM. Its copied
operand bypassed an existing slab-cover check. Applying that same check to the shared multi-output fill rejects the
invalid schedule; a legal dividing tile passes. The check applies only along the copied contiguous axis: a
transposed B operand can clamp whole outer N rows. A focused LoRA regression catches that distinction and preserves
both recorded schedules. This is a scheduling restriction, and loop fusion stays maximal.

Three older experimental files still use the retired `configs`/`loops` format and cannot load. They are preserved for
their authors. The audit excludes the realization corpus and serving fixtures, whose untimed rows are test inputs. No
realization case has an `_xfail_` suffix, and no recipe was tagged `prior-pending` to hide a failure.

### Final validation

The final priors use all 13 finalized repository golden sources, a fresh tune DB at version 9 and feature version 10.
Both fits use five folds and the default model settings. The schedule fit ranks a recorded choice first in 702 of
1,188 training groups and 553 held-out groups; the unchanged schedule reproduction gate passes all 91 slices.
The placement fit ranks a recorded choice first in all 2,751 training groups and 2,660 held-out groups; its unchanged
reproduction gate passes all 112 slices. Schedule rank reproduction is distinct from reproducing an exact cold pick:
the separate H100 cold schedule evaluation matches three of 125 choices, with no evaluation errors. Cold placement
evaluation reproduces all 1,474 decisions with no errors. The archive retains the earlier fits as superseded evidence.

The remote suites expose three graph-capture failures from a test fixture that leaves its element-count placeholder
unsubstituted in two generated kernels. The same omission exists on main. Binding the count fixes all three focused
tests; these direct CUDA fixtures bypass compiler and prior selection.

The suites also expose a pre-existing constant-binding error when the new prior selects source storage for a linear
weight. The generated buffer name had replaced the original binding name, leaving the supplied weight unbound.
Preserving the original name fixes the shared layout path. All seven focused layout tests pass; single and joint
source arms retain identical kernel identities, stored kernel bodies and feature values. No measurement or prior
changes are needed. The accepted CLI recordings bind constants through unchanged source metadata, not that binding
name. An audit of 44 recorded source-layout routes finds source metadata on all 147 non-scalar constants in their
13 traced targets; the recording inputs remain unchanged.

The FP8 byte-staging fixture also used narrow scaling that the corrected compiler must keep inside the operand,
making its requested asynchronous raw-byte stage invalid. Explicit full-width scaling matches the neighboring raw
byte test. All stage and bit-identical assertions remain, and the corrected fixture plus both narrow-rounding checks
pass on H100.

Five quantized serving checks also fail before compilation because their test hooks omit the backend's context
argument. The hooks now accept and forward it, including the corresponding native NVFP4 hook. Their graph, dtype,
quantization and numerical assertions remain unchanged. All five H100 checks pass in focused runs. The RTX 5090
native NVFP4 fixture also needs its matrix schedule pins scoped to the matrix output, leaving the activation encoder
free to choose its own schedule. The same native-instruction assertion and accuracy limits pass after that correction.

The fixed-slot MoE fixtures on Volta, Ampere and Hopper also contain a singleton expert schedule that is no longer
offered. The existing completion mechanism supplies supported cooperative schedules for the same exact kernels.
These are untimed test inputs; no benchmark measurement is replaced. All three platforms pass the integrated
capture cases with their original capture and accuracy assertions in the final remote suites.

The GDN state failures reveal a separate reduction value-numbering error. Binding a reduction axis discarded the
offset and stride of its load coordinates, so sums of different tensor slices could collapse into one. The corrected
number retains those expressions while preserving alpha equivalence and free-coordinate parameters. The small
shifted and strided reductions reproduce the error before the fix and pass afterward. The original fused GDN CPU
replay now has exact output, with state and history differences below 1.8e-10 and 1.5e-8; its H100 GPU regression
passes the unchanged tolerance. The tune DB advances to version 8 because exact identities change, and feature
version 9 invalidates datasets and priors that used the old reduction numbering.

The stored-body comparison covers all 2,225 kernels. Of 327 changed derived bodies, 319 have the same exact typed
identity under the corrected numbering. The other eight are unmeasured NVFP4 roots. None of the 2,518 kernel
measurement rows records changed arithmetic. A separate fresh-lowering audit covers all 591 programs in all 31
files, with no errors or timeouts. It finds 19 changed contexts: eight Qwen prefill contexts across hardware and
experimental files, four Ministral cut paths, three LoRA contexts, three DeepSeek contexts, and one GPTQ context.
The Qwen cut workspaces lose a redundant leading unit dimension. That changes their stored kernel identities and
requires new measurements on A100, H100, RTX 5090 and V100 SXM2. It changes no fusion boundary. The unavailable
cards lose no additional measurements from this correction. Exact-child comparisons preserve measurements for
the Ministral and DeepSeek path changes. The LoRA routes were already stale before the reduction correction.
LoRA, DeepSeek and GPTQ retain every stored kernel and row after their path corrections. V100 prefill also permutes
three changed children. Their complete statement trees match after workspace renaming and removing the obsolete
leading zero coordinate, which establishes the intended correspondence before re-recording. All 20 original V100
schedules still decode on those intended fresh computations. No heuristic semantic matching was added to the compiler.
All 248 realization-corpus freshness checks also pass without changing their case files. The same reduction-numbering
fix restores the MoE rider's one-token and 16-token placement evidence. Reinstating only the old numbering reproduces
both strict-evidence failures; the corrected numbering emits both programs against the existing fixture.

Refreshing changed child kernels also revealed that an unchanged root could retain its old whole-target timing.
Restamping now clears that timing whenever a recorded descendant or decision changes. Unchanged individual kernel
measurements and unrelated target timings remain intact. Two reproductions fail before this correction; all eight
focused checks pass afterward.

The Ampere and Hopper suites also expose a normalization cycle in an atomic split of native attention. Two free
axes repeatedly exchange order because their role comparison reads the current ordering of affine terms. Comparing
canonical expressions under fixed coordinates removes that feedback. The bounded reproduction detects the two-state
cycle before the correction; the corrected diagnostic realizes all 13 candidate arms in three seconds, with every body
converging within two rounds. All 85 focused normalization and split checks pass, as does the local native attention
check with its original accuracy assertion. The same old-versus-fixed reproduction confirms the correction on sm70,
so the V100 skip tied to that stall is removed. The old full runs are stopped with their partial logs and profiles
preserved. They are superseded diagnostic runs, not successful full-suite results. All 2,225 stored kernels retain
identical normalized programs, derived bodies, exact identities and structural features after the axis-order fix.
Recorded measurements remain valid. The general canonicalization rule still changes, so tune DB version 9 and
feature version 10 invalidate prior caches and datasets before a fresh import and refit.

The final fresh-lowering audit checks all 591 programs across all 31 files under the corrected normalization rule.
Every program is unchanged, and all source digests match the canonical files. Seven attempts reach the 110-second
limit; their successful retries on an otherwise quiet machine establish the final verdicts. The archive retains those
capped attempts and one corrected input-path setup error separately from the 591 successful checks.

Both final prior refits and all 203 reproduction slices pass. All three remote full suites pass on commit
`fe3cdee26`, with the final version-10 priors. Each run invokes `make test` and exits zero. Their single expected
failure is the unavailable RTX 4090 prefill case.

| Remote GPU | Passed | Skipped | Expected failures | Pytest time |
| --- | ---: | ---: | ---: | ---: |
| A100 | 6,775 | 980 | 1 | 3,738.41 s |
| H100 | 6,876 | 879 | 1 | 2,385.56 s |
| V100 SXM2 | 6,751 | 1,004 | 1 | 3,771.13 s |

Each suite collects 7,752 worker items; the terminal summaries also include four collection-time skips. No worker
restarts or reruns occur. The later lint correction wraps a tuple and removes an unused module alias; production
Python has the same AST as the tested source. Two test-only formatting changes also preserve their ASTs. Final
remote `make lint` passes at `4f3a8bf56`, including Ruff checks, all 975 files formatted and test-duration formatting.

`tuning_golden_recovery_2026-10-09.tar.gz` preserves the before/after inventories, recording and strict replay JSON
and logs, promoted golden snapshots, exact-match checks, hardware/software metadata, and final validation evidence.
The work ran on GCP A100/H100, CloudRift V100 and the local RTX 5090. Full suites run only on the remote machines.
Archive SHA-256: `9ec57c0ff7db18612fbd83347be2d505ca6b741f042ccee1cece1b4dce4f01b6`.

## V100 FP16 decode: fused attention and output, vector weight reads (2026-10-07)

Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, layer zero, FP16, deployable O3, fast math off, on a
Tesla V100 SXM2 16GB (UUID `GPU-fb047284-9557-a127-0787-70f97e92826a`, driver 580.178.04, NVCC 12.9.86, Torch
2.13.0+cu126, Transformers 5.14.1). Each pair is one fresh process timing `torch.compile` and Emmy on the Hugging Face
layer (`emmy run MODEL --layer 0 --bench --bench-backends eager,tcompile,emmy`, warmup 10, iters 100), with a fresh
tune DB, strict evidence and `--strict` accuracy. Every decode `torch.compile` cache was seeded with the nine saved
Inductor choices of the round-8 archive.

The decode golden takes two changes. Its measured cut fuses the one-key attention value reduction into the output
projection (eight launches instead of nine). Body normalization now folds the one-key softmax weight exactly:
`v * (exp(s - s) / exp(s - s))` becomes `v - (s - s)`, which gives the same bits for every input, inf and NaN
included, without the exponent or the division. The new `coop/v<n>` reduction lets each lane read `n` adjacent weight
elements as one vector load; the recorded rows use it for Q, K/V, attention with output, gate/up and down.

| Decode, seven pairs | `torch.compile` median, µs | Emmy median, µs | Median paired gap, µs | Launches |
| --- | ---: | ---: | ---: | ---: |
| main `5c3c0ea12`, golden not deployable under strict evidence | 55.252 | 70.110 | +14.858 | 14 |
| Fused cut, exact softmax fold, scalar reads | 55.406 | 56.051 | +0.591 | 8 |
| Fused cut, exact softmax fold, `coop/v<n>` rows | 55.084 | 53.931 | −1.135 | 8 |

A negative gap is an Emmy lead. Main's row ran without strict evidence because its golden lost its cut route when
main changed how cut pieces form; it fell back to the prior's 14 launches. In the last group every pair favored Emmy
(gaps −0.317 to −1.948 µs), all fourteen backend runs passed accuracy, and every Emmy run had strict measured evidence.
An earlier seven-pair group on the same rows before the final catalog change gave −0.862 µs.

Per kernel, Inductor and Emmy were close before the vector reads: under Nsight Compute with flushed caches and base
clocks the only clear loss was gate/up (21.1 µs against 19.6 µs). In isolation the vector rows took K/V from 4.4 to
4.0 µs, attention with output from 5.5 to 4.5 µs, and down from 6.9 to 5.5 µs; gate/up moved only from 17.0 to
16.7 µs, near the card's streaming bandwidth.

Prefill, three fresh-process runs of the unchanged 21-launch golden: Emmy 484.864 µs median against
`torch.compile` 624.674 µs (main: 482.304 against 614.904). The prefill golden takes no `coop/v<n>` rows.

The benchmark decodes one token with one key, so its softmax has a single element. Real decode attends over the
whole KV cache, where the softmax fold does not apply; the vector reads do.

`tuning_v100x1_round9_2026-10-07.tar.gz` holds each run's JSON and log: `ab-s1` and `ab-s512` (main against the
branch), `final-1` and `final-2` (the recorded rows), `final-s512`, the pinned layer screens `e2e-1` and `e2e-2`, the
single-kernel screens `kab-*`, the record run `rec2`, and the Nsight reports `ncu-emmy`, `ncu-tc` and `prof1`.
`tuning_v100x1_round8_2026-10-06.tar.gz` and `tuning_v100x1_round8_rebased_2026-10-07.tar.gz` hold the earlier
rounds, including the nine saved Inductor choices. `results_v100x1.tar.gz` is the full two-shape recipe run
`20261006T224258Z` (root `2026-10-06_22-42-58/`); it predates the softmax fold and the vector rows, and measured decode
at 55.962 µs against 55.362 µs for an unpinned `torch.compile` cache.

## V100 prefill gate/up scheduling (2026-10-05)

This round tests whether measured gate/up schedules improve the FP16 prefill layer on a Tesla V100 SXM2 16GB. Two
rows were added to the prefill golden: a smaller MMA tile and then a grouped CTA raster. No compiler code or recipe
changed. The candidates were screened and selected manually from measured schedules. The decode golden was
re-measured without a schedule change.

The comparison uses Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, layer zero, sequence
length 512, deployable O3 and fast math disabled. It ran on GPU `GPU-60f18d3b-342c-913e-5f44-1bba2d7a7c8a`,
with Ubuntu 24.04.1, driver 580.178.04, NVCC 12.9.86, Torch 2.13.0+cu126 and Transformers 5.14.1. Each arm uses
a fresh process and tune DB, ten warmups, 100 iterations, strict measured evidence, and the scaled actual-model
accuracy check. The order alternates. Both arms keep 21 launches and the same ordered CUDA sources except for
gate/up.

| Pair | Order | Previous Emmy, µs | Final Emmy, µs | Reduction |
| --- | --- | ---: | ---: | ---: |
| 1 | Previous then final | 503.296 | 486.400 | 3.36% |
| 2 | Final then previous | 501.760 | 480.768 | 4.18% |
| 3 | Previous then final | 494.592 | 485.888 | 1.76% |
| 4 | Final then previous | 504.320 | 486.400 | 3.55% |
| 5 | Previous then final | 503.296 | 482.304 | 4.17% |
| 6 | Final then previous | 494.080 | 486.912 | 1.45% |
| 7 | Previous then final | 491.520 | 477.184 | 2.92% |

All fourteen arms pass accuracy. The median paired reduction is 16.896 µs, or 3.36%; every pair improves. Gate/up
changes from a measured `f4x2` tile at about 112.5 µs to `f2x2` with `gm8` raster at about 100.4 µs.
Separate pairs measured the tile and raster choices independently. Their median whole-layer reductions were 2.54%
and 2.40%, respectively. Those intermediate percentages are not added to the 3.36% final gain.
Nearby tile, work, stage, down, output and attention choices that lost their screens were not added to the goldens.

The final two-row recipe ran from source `00a9195e6` on 2026-10-06. Both rows succeeded. Each model run compares
Emmy against eager and `torch.compile` on the same input. Five fresh-process golden replays per row pass strict
accuracy and strict evidence. Within each shape, the model run and all repeats use identical ordered CUDA source
hashes. Those source hashes also match the preceding recipe run before the branch was rebased.

| Shape | Emmy model, µs | `torch.compile` model, µs | Emmy lead | Strict replay median [range], µs | Launches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode, s1 | 58.260 | 64.497 | 9.67% | 57.991 [57.937–58.099] | 9 |
| Prefill, s512 | 483.840 | 604.857 | 20.01% | 482.304 [479.232–484.864] | 21 |

After the rejected screens, another actual-model prefill run with strict evidence and the recipe's scaled eager
check selected the same 21 CUDA sources and measured 485.888 µs. A separate fixed-tolerance `--strict` check stopped
before timing: 6 of 524,288 outputs exceeded rtol=0.001 and atol=0.001, with maximum absolute error 0.00391. The
recipe's scaled actual-model checks and all ten strict golden replays passed; they use different accuracy checks and
inputs.

Two further fresh prefill Inductor caches gave 601.995 and 597.630 µs versus Emmy's 482.304 and 484.352 µs. Both
backends passed the scaled eager check in each run, and Emmy selected the same 21 CUDA sources as the rebased recipe.

Three further fresh Inductor caches on this card chose different decode times, while Emmy stayed near 58 µs. These
are separate s1 model runs with the same revision and golden, ten warmups and 100 iterations. Reusing the fastest
cache repeated its result. Each backend passed the scaled eager comparison with a fullgraph `torch.compile` run.

| Inductor cache | `torch.compile`, µs | Emmy, µs |
| --- | ---: | ---: |
| A | 64.361 | 57.883 |
| B | 63.471 | 57.970 |
| C | 54.331 | 57.937 |
| C, repeated | 54.440 | 57.856 |

The generated Python kernels for A and C differ only in cache paths, but their saved Inductor autotune choices
differ. Seeding fresh caches with all nine choices from A or C reproduced 63.920 and 54.404 µs, respectively.
Changing only the down-projection choice in the A set gave 60.592 µs; restoring A's choice in the C set gave
57.329 µs. The fastest valid cache leaves Emmy 6.64% slower than `torch.compile` on this decode shape. The 9.67%
lead in the recipe is one cache choice, not a general V100 claim. The prefill result is one FP16 layer and shape,
not request-level serving evidence. Golden replays use their own inputs and strict comparison; their times validate
the selected sources and are not compared with the model's `torch.compile` time.

A further single-kernel decode screen tried a vectorized cooperative reduction for the down projection. It changed
the launch grid and took 330.069 µs versus 7.537 µs for the existing measured schedule on the same Loop IR input.
The losing variant was not taken to a full-layer comparison.

Later isolated prefill screens found no better gate/up warp layout, Q tile, or Q stage. A smaller V-projection tile
measured 33.6 µs against 43.2 µs for the existing tile on a standalone Loop IR input. The precision-correct pinned
golden replay could not compile its Q projection under strict evidence: no measured row covered the offered schedule.
An earlier replay without the required fast-math-off pin hung in Q and does not test the V candidate. There is no
valid full-layer result for this tile, so no V row was added.

That round's `results_v100x1.tar.gz` had root `2026-10-06_04-19-46/`, run ID `20261006T041946Z`, two succeeded
system-only experiment records, two `*_artifacts.tar.gz` bundles and logs. Each bundle holds
`torch-compile/model.json` and `verification/repeat-{0,1,2,3,4}`. The separate
`tuning_v100x1_round7_2026-10-05.tar.gz` retains the paired JSON, screening results, post-screen checks, Inductor
autotune choices, recording logs and the two earlier recipe runs under `v100-round7-evidence/`. The previous report
sections and raw records remain in Git history.

## V100 decode and prefill after output scheduling (2026-10-05)

This round asks whether another measured decode schedule helps and whether the two V100 layer shapes still replay
cleanly. It also investigates why the earlier rental measured `torch.compile` near 57.5 µs for decode while the next
rental measured about 61–62 µs. The same physical Tesla V100 SXM2 16GB card was used in all three rounds, confirmed
by its GPU UUID. The new output-projection schedule is the only change to the V100 goldens; prefill is re-measured,
not retuned. No compiler code changed.

### Decode schedule

The comparison uses Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, layer zero, sequence
length one, FP16, deployable O3, and fast math disabled. Each arm is an actual-model run in a fresh process and tune
DB, with captured whole-layer timing, ten warmups, 100 iterations, eager and `torch.compile` beside Emmy, strict
accuracy, and alternating order. The candidate pins 256 threads for the output projection; the baseline selects
the previous golden with strict evidence. Only the output kernel source changes. Both arms retain nine launches.

| Pair | Order | Previous Emmy, µs | New Emmy, µs | Reduction | `torch.compile`, previous / new, µs |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Previous then new | 58.084 | 57.856 | 0.392% | 61.111 / 60.945 |
| 2 | New then previous | 58.027 | 57.613 | 0.712% | 60.994 / 61.937 |
| 3 | Previous then new | 57.970 | 57.937 | 0.057% | 61.008 / 60.957 |

All six arms pass. The median paired reduction is 0.392%, or 0.228 µs. The output kernel's measured launch falls
from 5.49–5.59 to 5.09–5.12 µs; the smaller whole-layer effect includes run variation. Output schedules with 64
and 512 threads and shared K/V schedules with 64 and 256 threads lost their single-run kernel screens. The chosen
256-thread output row was measured through the golden recording path. The recorder's separate replay of an older
named row exited nonzero because that synthetic route lacked evidence; the measured greedy row was written before
that failure. Only that row was added to the committed golden. Its fresh-lowering check and an independent, unpinned,
strict actual-model replay pass and select it again.

### Fresh recipe replay

The final two-row recipe ran from clean source `03e384647` on 2026-10-05. It used the same model revision, layer and
precision regime, with sequence lengths one and 512. The machine was Ubuntu 24.04.1 with an Intel Xeon E5-2680 v4,
driver 580.178.04, NVCC 12.9.86, Torch 2.13.0+cu126, Triton 3.7.1 and Transformers 5.14.1. Both rows succeeded.
Their model runs compare the same input against eager and `torch.compile`; five fresh-process golden replays per row
check strict accuracy and strict evidence against their own inputs. All ten replays pass, and each shape's model run
and repeats use identical ordered CUDA source hashes.

| Shape | Emmy model, µs | `torch.compile` model, µs | Emmy lead | Strict replay median [range], µs | Launches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode, s1 | 57.937 | 59.728 | 3.00% | 57.883 [57.628–58.045] | 9 |
| Prefill, s512 | 487.936 | 621.534 | 21.49% | 499.200 [492.544–502.272] | 21 |

The prefill golden and its selected sources did not change in this round. Its whole-layer repeat range is about
9.7 µs despite per-kernel sums near 450 µs, so the 487.936 µs model run is one observation, not a new prefill
speedup over the prior round. The strict replay medians validate stability and accuracy; their different input and
reference path are not compared with the model's `torch.compile` time.

### Why the compiled reference moved

Two otherwise identical decode runs with separate empty Inductor caches chose different autotune configurations.
Repeating each cache preserved its result. Emmy stayed near 58 µs throughout:

| Inductor cache | First `torch.compile`, µs | Repeat, µs | Seeded fresh cache, µs |
| --- | ---: | ---: | ---: |
| Fast choice | 58.046 | 58.083 | 58.383 |
| Slow choice | 66.971 | 67.215 | 66.757 |

Nine of the ten generated Python source hashes match across the caches; the tenth wrapper differs only in its
absolute cache paths. The saved autotune choice for the same fused attention-value reduction differs: the fast
choice uses two output elements per program, the full 2,048-element reduction and 16 warps; the slow choice uses
eight output elements, a 64-element reduction chunk and two warps. Copying all nine choices into fresh caches
reproduces the two timings. Replacing only this reduction's choice in the fast set raises `torch.compile` to
69.025 µs, while replacing a different reduction's choice leaves it at 57.446 µs. That is causal evidence that
Inductor's autotune selection can move the reference by more than the earlier cross-rental difference on this same
card. The earlier rental's cache was not saved, so its exact selected configuration cannot be identified.

The faster rental's package freeze also held 15 extra CUDA 13 packages. Installing those exact versions without
changing Torch, Triton or Transformers left fixed-cache `torch.compile` times effectively unchanged: fast
58.083 → 58.039 µs and slow 67.215 → 66.761 µs. Those packages do not explain the measured cache split. With a
fast cache, Emmy and `torch.compile` are essentially tied near 58 µs; the 3.00% lead in the official recipe is a
within-run result for its chosen compiled reference, not a general V100 lead.

### Evidence and limits

The original archive for this earlier replay had root `2026-10-05_02-27-57/`, run ID `20261005T022757Z`, two
succeeded system-only experiment records, the corresponding `*_artifacts.tar.gz` bundles, and logs. It remains in
Git history; the current named archive is described above. Each earlier bundle holds
`torch-compile/model.json` and `verification/repeat-{0,1,2,3,4}`. The separate
`tuning_v100x1_round6_2026-10-05.tar.gz` retains paired JSON, screens, recording logs, package freezes and both
sets of Inductor autotune choices under `v100-round6-evidence/`. A first infrastructure attempt failed before
measurement because the rented image lacked Python venv support; no values from it are used. The rented V100 was
terminated after both archives were verified.

These results support two FP16 layer shapes on this card. They do not establish serving latency or a cross-card
V100 advantage. A V100-qualified serving artifact and a request-level comparison remain necessary for a serving
claim.

## V100 decode: Q and down source layouts (2026-10-04)

The previous V100 golden left Q and down projections as split pairs, for 11 launches. This round records source
layouts for both. The compiler now selects nine launches without pins. No compiler code changed in this round.

The card is a Tesla V100 SXM2 16GB with driver 580.178.04, NVCC 12.9.86 and Torch 2.13.0+cu126. The model is
Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, layer zero, sequence length one, FP16,
deployable O3 and fast math disabled. Each process uses a fresh tune database. The comparison uses captured whole-layer
timing, ten warmups and 100 iterations, with eager and `torch.compile` beside Emmy. All six runs pass strict accuracy
and strict evidence. The order alternates to expose drift.

| Pair | Order | Previous, µs | New, µs | Reduction | `torch.compile`, previous / new, µs |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | New then previous | 61.212 | 58.206 | 4.91% | 61.959 / 61.462 |
| 2 | Previous then new | 61.763 | 57.937 | 6.20% | 62.354 / 61.433 |
| 3 | New then previous | 62.733 | 58.197 | 7.23% | 62.288 / 61.996 |

The median paired reduction is 6.20%. The new whole-layer time stays at 57.94–58.21 µs while the previous golden
varies from 61.21 to 62.73 µs. The new result is 3.26–3.80 µs faster than `torch.compile` measured in its own
process in these three runs. Each arm reproduces its ordered CUDA source hashes. Seven sources are common; the Q and
down split pairs become one source each. Their per-launch sums change only from 51.57–51.84 to 50.81–51.51 µs, so
removing the two launches accounts for most of the measured whole-layer gain.

In the first pair, Q changes from 4.560 + 1.880 µs to 6.081 µs and down from 5.425 + 2.107 µs to 7.282 µs.
Those isolated launch timings diagnose the change; they are not the whole-layer comparison. The unmeasured prior
chose a 37.6 µs schedule for source down. A pinned 256-thread cooperative reduction measured about 7.3 µs and was
recorded. The source Q kernel uses the measured 128-thread cooperative reduction. Nearby Q and gate/up thread counts
did not improve their screened kernel times, so their existing schedules remain.

The first synthetic golden replay pinned an old output-projection row across the new full-layer route and failed
strict accuracy. Its timing is excluded. The new Q and down rows and routes were retained from the clean greedy
measurement. A fresh-lowering check and a separate unpinned, strict actual-model replay passed for the trimmed
golden. The accepted golden is `golden/qwen3-06b-s1_v100.golden.json`; raw JSON, logs, intermediate goldens, hardware
details and checksums are in `tuning_v100x1_round5_2026-10-04.tar.gz`.

The previous rental measured `torch.compile` near 57.5 µs, versus 61.4–62.4 µs on this rental. The cause of that
difference is not established, so the within-run result here does not establish that the gap on the previous card
closed. This is one decode shape on one card, not a serving or cross-card result.

## V100 decode: measured source layouts (2026-10-04)

The compiler can now offer a transposed constant in its folded layout or original storage layout and choose from
measured kernel rows. On Qwen3-0.6B decode, the V100 golden selects original storage for gate/up, shared K/V, and the
output projection. Each choice is an ordinary kernel-set fork after maximal fusion. The final strict compile takes
all three choices without pins and runs 11 kernels, down from 14 in the original golden.

The card is a Tesla V100 SXM2 16GB with driver 580.178.04, CUDA 12.9.86 and Torch 2.13.0+cu126. The model revision is
`c1899de289a04d12100db370d81485cdf75e47ca`, layer zero, sequence length one, FP16, O3 and fast math disabled.
Each process uses a fresh tune database. Whole-layer times are CUDA-graph captured, with ten warmups and 100 iterations;
`torch.compile` and eager run alongside Emmy. Every run below passes strict evidence and the unchanged strict accuracy
check against eager. The order alternates to expose timing drift.

| Pair | Order | Original, µs | Final, µs | Reduction | `torch.compile`, original / final, µs |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Final then original | 66.801 | 61.386 | 8.11% | 57.490 / 57.549 |
| 2 | Original then final | 66.982 | 61.099 | 8.78% | 57.413 / 57.527 |
| 3 | Final then original | 66.944 | 62.825 | 6.15% | 57.488 / 57.541 |

The median paired reduction is 8.11%. The final whole-layer timer varies more than its kernels: the sum of measured
launch times stays at 51.49–51.64 µs, versus 56.80–56.96 µs for the original. All three final runs have the same 11
CUDA source hashes, schedules and launch order. `torch.compile` stays near 57.5 µs, leaving about 3.6–5.3 µs of
whole-layer gap in these runs. This is one decode shape on one card, not a serving or cross-card result.

The selected source layouts replace one split pair each. One adjacent pair's per-launch timings show where the work
changed; these launch times are diagnostics and are not summed to claim a whole-layer speedup:

| Projection | Original split, µs | Source layout, µs | Launches |
| --- | ---: | ---: | ---: |
| Gate/up | 18.182 + 2.158 | 17.548 | 2 → 1 |
| Shared K/V | 4.667 + 2.535 | 5.052 | 2 → 1 |
| Output | 4.192 + 2.077 | 5.575 | 2 → 1 |

The gate/up layout alone won two earlier full-layer pairs (67.042 versus 65.175 µs and 67.102 versus 64.331 µs).
Adding the corrected K/V layout won two pairs against gate/up alone (64.398 versus 64.057 µs and 64.632 versus
63.260 µs). The initial K/V golden accidentally stored the folded body under source input names; its measurement did
not deploy. Re-forming the source kernel fixed its golden body, and the final pairs above use only that corrected
kernel. An isolated replay that pinned the K/V schedule across a synthetic full-layer input failed strict accuracy;
its timing is excluded. Every actual-model full-layer result above passes.

Other source layouts remain unpromoted. The Q projection's source kernel measured 6.895 µs, while its measured
folded split cost about 6.5 µs, so the selector kept the folded layout. In a 5/20 screen, the output source layout
with the prior's unmeasured warp schedule took 25.69 µs and 86.17 µs for the full layer; a measured `t128`
cooperative reduction instead took 5.61 µs for that kernel and passed full-layer accuracy. The golden stores that
row. The source layout is therefore evidence-selected, not a rule that all transposed weights should use source storage.

The raw JSON, logs, intermediate goldens, hardware record, failed probes and checksum manifest are in
`tuning_v100x1_round4_2026-10-04.tar.gz`. The accepted golden is `golden/qwen3-06b-s1_v100.golden.json`. These results
are suitable as evidence for this V100 layer and shape; broader performance claims need more shapes and cards. A
separate sequence-length-512 check reached its 120-second CPU compilation limit, so this round has no new prefill
timing or accuracy result.

## H100 scheduling experiments: target gain, broader regression (2026-10-03)

No compiler optimization from this four-hour continuation is retained. Moving independent row reductions after
the value product is issued improves the target H100 layer by 0.33%, but an existing smaller attention case loses
in all six balanced pairs, with a median slowdown of 1.84%. That repeatable cost outweighs the modest target gain.
Larger chunks, cross-chunk pipelines, register-held queries and balanced reduction trees also fail to justify a
change. The report and compressed evidence are the complete change: compiler, identities, schedules, goldens,
priors, FP16 boundaries, disabled fast math and numerical tolerances remain unchanged.

### Reordering independent row reductions

The smallest candidate moves independent row reductions and the pivot update after the asynchronous value product
is issued. The full wait remains before shared-slot release, the next accumulator rescale and final stores.
It uses the existing WGMMA primitives and keeps the non-WGMMA order unchanged. The selected kernel retains 168
registers, 80 KiB shared memory and 128 CTAs of 128 threads, with no spills or local memory.

Six fixed balanced pairs use Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, layer zero, sequence
length 512, FP16, O3 and fast math disabled. Every process gets a fresh tune database and dump, ten warmups and
100 iterations. All twelve processes finish successfully and pass the unchanged scaled comparison against eager.
Each also measures `torch.compile`.

| Pair | Order | Baseline, µs | Candidate, µs | Reduction |
| --- | --- | ---: | ---: | ---: |
| 1 | Baseline then candidate | 82.45415 | 81.88307 | 0.6926% |
| 2 | Candidate then baseline | 82.57477 | 82.62154 | −0.0566% |
| 3 | Baseline then candidate | 82.18585 | 82.15384 | 0.0389% |
| 4 | Candidate then baseline | 82.77170 | 82.27939 | 0.5948% |
| 5 | Baseline then candidate | 82.90708 | 82.62892 | 0.3355% |
| 6 | Candidate then baseline | 82.81354 | 82.54277 | 0.3270% |

Five pairs win. The median paired reduction is 0.3312%; the geometric mean reduction is 0.3224%. Attention's
arithmetic mean falls from 12.57127 to 12.12623 µs, or 3.54%. This is a small whole-layer effect on one shape and
card. The source and all launch metadata are identical within each arm; across arms, only attention's CUDA body
changes. The ten unrelated complete CUDA objects stay identical.

SASS inspection shows that ptxas already overlaps some independent reductions in the baseline. Both versions put
their final matrix instruction adjacent to the wait, so this result does not establish newly enabled overlap or
identify one instruction as its cause. The measured benefit is from the complete instruction-order change.
Eight CPU ordering and causal-bound checks pass. Four GPU cases pass three seeds each at 0.001 absolute and
relative tolerances, covering one- and two-stage asynchronous copies, TMA and an active causal bound. Five fresh
database replays of the stored complete program pass against eager at those same tolerances. Every replay reports
maximum absolute error 0.001953125, mean absolute error 8.138e-08 and the same eleven selected sources. Absolute and
relative tolerances apply together; this is not a zero-error result.

The final separate actual-model comparison passes the scaled check and reproduces all eleven complete CUDA objects
from the paired candidate. It measures 81.964 µs for Emmy, 79.738 µs for `torch.compile` and 198.792 µs for eager.
Those single-process values are not the paired speedup claim, and Emmy still trails the compiled reference here.
These target-shape results warranted checking another existing attention shape before retaining the change.

The noncausal head-width-64 corpus case, shape `(1, 4, 256, 64)`, regresses in an initial screen and in every
subsequent pair. The six-pair campaign fixes its existing pinned schedule and alternates execution order. All
twelve processes finish successfully and pass eager comparison at 0.001 absolute and relative tolerances, with
maximum absolute error 0.000244140625. Each arm reproduces its complete CUDA object in every process.

| Pair | Order | Baseline, µs | Candidate, µs | Slowdown |
| --- | --- | ---: | ---: | ---: |
| 1 | Baseline then candidate | 6.00939 | 6.10436 | 1.5804% |
| 2 | Candidate then baseline | 6.01351 | 6.14434 | 2.1757% |
| 3 | Baseline then candidate | 6.01384 | 6.03956 | 0.4276% |
| 4 | Candidate then baseline | 5.93217 | 6.10945 | 2.9885% |
| 5 | Baseline then candidate | 6.04015 | 6.08316 | 0.7121% |
| 6 | Candidate then baseline | 5.97935 | 6.10545 | 2.1089% |

The median paired slowdown is 1.8447%; the geometric mean slowdown is 1.6617%. Both versions use sixteen CTAs of
128 threads, 128 registers, 32 KiB dynamic plus 1 KiB static shared memory, and no local memory or stack. Their
1,416 SASS instructions have identical opcode counts. Those controls rule out a resource-count explanation but
do not identify the cause. An unmeasured greedy selection has a different schedule and is excluded from this
comparison. A separate noncausal head-width-128 case, shape `(1, 8, 512, 128)`, improves from 11.040 to 10.662 µs;
that is one screen, not a paired result.

The global reorder and its tests are therefore removed. No existing schedule decision expresses this scalar
ordering independently. A shape heuristic or a new scheduling knob is not justified by the small target benefit.

Matched Nsight Compute captures select the first exact attention launch in each immutable compiler arm. Both
finish successfully with kernel replay, cache flushing, no profiler clock adjustment and four metric passes.
They use the same 128-CTA grid, 128 threads per CTA and 168 registers. These cold selected-launch diagnostics do
not measure warm whole-layer latency.

| NVIDIA metric | Baseline | Row reorder |
| --- | ---: | ---: |
| Tensor warp instructions | 49,152 | 49,152 |
| Load/store warp instructions | 149,504 | 149,504 |
| Total warp instructions | 3,870,720 | 3,871,232 |
| Active warps / scheduler active cycle | 1.000 | 1.000 |
| Eligible warps / scheduler active cycle | 0.3294 | 0.3405 |
| Wait stalls | 22.228% | 21.560% |
| Barrier stalls | 11.735% | 12.008% |
| Long-scoreboard stalls | 14.502% | 15.580% |
| Cold diagnostic duration, µs | 15.104 | 14.720 |

The instruction mix and slightly higher issue eligibility are consistent with a modest scheduling change. Stalls
do not improve uniformly: both barrier and long-scoreboard shares rise. The capture does not isolate a single
cause for the unprofiled gain. Counter access requires the existing passwordless profiler privilege; no driver
permission or clock configuration changes are made.

### Larger reduction chunks

A prototype extends the existing atom-major shared layout to two complete K atoms. It reuses the same fill and
descriptor maps for raw projections, computed inputs, transposed weights and transposed attention values.
Twelve H100 numerical controls pass. All eleven accepted k4 CUDA objects remain unchanged, and the maintained
H100 golden remains current. The added capability does not produce a qualifying schedule and is excluded.

| Candidate | Baseline, µs | Candidate, µs | Comparison |
| --- | ---: | ---: | --- |
| Q projection, k8 | 82.420 | 84.042 | Separate actual-model screens |
| Gate/up, k8 | 20.815 | 23.961 | Exact-production closed kernel |
| Attention, k8 / output N128 | 12.280 | 12.516 | Exact-production closed kernel |
| Attention, k8 / output N64 | 12.335 | 14.386 | Exact-production closed kernel |

Q's own model child increases from 6.537 to 6.739 µs. Its closed reproduction differs from production, so those
closed timings are excluded. Gate/up and the first attention model attempts time out without JSON results;
their supported closed comparisons supply the isolated findings instead. Every closed candidate passes strict
same-input comparison against the exact accepted Emmy source. This supplements the independent numerical controls;
it does not create a whole-layer result for those candidates.

Gate/up increases from 168 to 196 registers without local memory. Attention N128 reaches 255 registers and eight
local bytes, versus the baseline's 168 registers and no local memory. Reducing its output width to N64 removes
the local bytes but still uses 255 registers and loses substantially. The N64 grid also doubles to 256 CTAs and
activates the existing causal bound. Compared with accepted attention, duplicated score work rises 25%, value
product work falls 37.5%, and total tensor arithmetic falls 6.25%. It is not an equal-work comparison.

The weighted row-statistic controls pass. The first weightless controls expose an independent tracing bug: a
dropped None weight leaves epsilon in the weight position. The same approximately 1e-6 scaling reproduces under
the old k4 lowering against both original eager Torch and the unchanged NumPy oracle. No tracer change or
tolerance relaxation is included in this optimization work.

### Overlapping attention work

The first cross-chunk prototype passes numerical checks only after ptxas serializes its asynchronous matrix
products. Both accumulator-read and injected-wait warnings are retained. Moving the partial wait to the loop
bottom removes one warning but still inserts a full wait before the back edge. Explicit operand fences do not
remove that drain. These builds do not demonstrate overlap.

A rotated loop follows the ordering used by
[FlashAttention-3](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp): prepare the first probability fragment in a
prologue; issue the next score and previous value product; wait for the score; perform independent scalar work;
then finish the value product before reusing its operands and shared slot. All asynchronous products finish
inside their iteration. This extends the existing shared loop scheduler rather than adding an attention emitter.

Seven numerical cases pass three seeds each at unchanged 0.001 absolute and relative tolerances. They cover one,
two and three chunks, two- and three-stage asynchronous copies, TMA and active causal bounds. A small control's
SASS contains 98 scalar instructions between the partial and full waits, with no writes to pending operands and
no injected-wait warnings. Exponentials remain after the full wait. An additional completed-value fence does not
move them. The production kernel keeps 168 registers and no local memory, matching baseline.

| Rotated schedule | Baseline isolated, µs | Candidate, µs | Shared memory |
| --- | ---: | ---: | ---: |
| Two-stage asynchronous copy | 12.309 | 13.699 | 80 KiB |
| Three-stage asynchronous copy | 12.171 | 12.469 | 112 KiB |

Neither setting qualifies. Three stages recover much of the loss, consistent with more time to prefetch the next
tile, but the timings do not prove that cause. The attempted three-stage TMA spelling is refused before candidate
execution and supplies no timing. A later tail-fill control guards only unused transactions, retaining every
asynchronous-copy commit and every needed TMA phase. Its assembly bypasses sixteen payload-copy instructions
while retaining the commit. Seven GPU cases pass three seeds each at the same tolerances. The exact-production
closed kernel costs 12.571 µs versus a 12.176 µs baseline isolated measurement. It is also rejected. A final
four-stage TMA attempt emits the unchanged baseline in both actual-root and closed compilations. The matching
three-stage asynchronous-copy control does change the selected source. Cut pieces may drop an unsupported
published schedule restriction and retain their accepted selection; successful compilation alone does not prove
that the requested stage was realized. The exact TMA refusal on this root remains unresolved. The synthetic
four-stage TMA case is legal, but that does not establish production reachability. No GPU candidate timing is
assigned to this unselected attempt.

### Other instruction and data movement changes

Hoisting invariant query descriptors produces byte-identical addressed SASS across all 2,264 instructions, so
that change is rejected. A separate prototype keeps invariant query fragments in registers through the existing
register-A WGMMA path. Nine GPU cases pass three seeds each at unchanged 0.001 tolerances. These cover both ordinary
and production head layouts, including eight warps. All ten unrelated complete CUDA objects remain unchanged.

The production four-warp variant uses 225 registers instead of 168 and 64 instead of 80 KiB shared memory, without
spills. Its strict closed timing is essentially tied: 12.421 versus 12.440 µs. The eight-warp variant costs
21.219 µs and uses 179 registers. The query loads overlap the initial key/value copies in assembly, but that
does not establish a useful latency gain. Both variants and their added tests are excluded. A BF16 source control
checks packed operand mapping only; it is not a BF16 GPU correctness result.

A balanced local reduction tree shortens the actual addition and maximum dependency chains from fifteen to four
before the first shuffle, with the same 168 registers and no spills. Nine independent GPU cases pass three seeds
each at 0.001 tolerances. Its standalone strict screen improves from 12.307 to 12.160 µs, a modest signal.
Combining it with the row reorder loses: 12.132 versus 11.863 µs in the contemporaneous pinned comparison,
or 2.26% slower. The isolated comparison also loses, 12.017 versus 11.863 µs. Neither the tree nor the combination
is retained, and neither receives additional whole-layer pairs.

This reassociation preserves leaves and FP32 arithmetic but does not promise bitwise rounding. A prepared renderer
test for ordinary and Volta fragment layouts was not run on a GPU after the combination lost. The old single-V100
connection timed out, CloudRift listed no rentals, and the existing Google Cloud inventory had no running non-Hopper
GPU. These are availability findings, not numerical or performance results on another card.

### Remaining work

Projection normalization and RoPE fusion reaches a broader lowering gap. The projection and head statistic own
different output sweeps, and the statistic recomputes the projection instead of reading its rounded fragment.
The focused reproducer emits two scalar contraction nests. Generic fragment reuse, producer grid ownership and
statistic stores must be addressed before this becomes a useful schedule experiment. No fusion restriction is
introduced.

Production TMA coverage is still worth resolving before a larger producer/consumer warp-specialized schedule is
implemented. Its synthetic numerical coverage does not answer the actual-root refusal. A later candidate needs
exact production source verification, balanced whole-layer measurements and the same precision controls. The
failed pipeline and register-query probes here do not establish that warp specialization is the only useful path.

### Controls and retained evidence

The accepted V100 result below landed in #1031 while these experiments were running. This continuation stays on
the same branch in #1034 and does not change that claim. The immutable baseline compiler is
`dcb0807e0c2ab6850a5793d3a28593dc5028b403`, with the same compiler as main after #1019. Rebased main at
`0ab9c3e13296d8bb45d5036ee080c32c2057ed43` emits the same eleven complete baseline CUDA objects. Temporary
integration of the row reorder reproduces the measured attention source, changes no field of the ten unrelated
H100 CUDA objects and leaves all fourteen V100 decode CUDA objects unchanged. That integration is subsequently
reverted because of the smaller H100 regression. The maintained H100 golden remains current; no golden is recorded.

All GPU experiments in this continuation use the existing H100 80 GB HBM3 card with 132 SMs. The environment is
Python 3.12.3, Torch 2.14.0+cu130, Transformers 5.14.1, Triton 3.8.0, NVCC 12.9.41 and driver 580.178.04. Matched
NVIDIA captures use Nsight Compute 2025.2. No new VM is created. A100 stays stopped, and the 4×V100 VM remains unused.

This round's evidence is in four directories inside the single H100 results archive:
`tuning_h100x1_round4_wider_k_2026-10-03`, `tuning_h100x1_round4_register_query_2026-10-03`,
`tuning_h100x1_round4_attention_overlap_2026-10-03` and `tuning_h100x1_round4_balanced_fragment_2026-10-03`.
Their original manifests verify 6,664, 1,873, 15,067 and 1,947 payloads respectively. The consolidated archive
preserves all 34,234 original files from thirteen H100 bundles, including the current recipe records and earlier
diagnostics. Its manifest verifies those files plus the archive index. Every file is byte-identical to its source;
paths are safe and relative, with no symlinks or AppleDouble files. Sources, experimental patches, numerical checks,
commands, software versions, timeouts and failed attempts remain intact. Unrun controls are explicitly marked.

The final local suite passes: 5,788 tests passed and 1,307 skipped in 397.39 seconds. Lint and Git LFS object checks
pass. Both Make targets use the existing environment, with setup skipped and a Transformers 5.14.1 overlay; the
local suite does not rerun the GPU experiments above. The complete diff changes no compiler or test code, so no
corpus regeneration, golden recording, prior refit or new duration entries are required.

## One-warp V100 decode and rejected H100 alternatives (2026-10-02)

An existing one-warp schedule reduces V100 decode latency by 0.62% across six balanced whole-layer pairs. It changes
only the fused Q/K normalization, RoPE and score kernel. The tested H100 causal bounds and smaller tiles do not
justify replacing its accepted schedule. This phase adds one measured experiment-golden row and diagnostic evidence;
there is no compiler, kernel-identity, precision, maintained golden or prior change.

### V100: a small gain from one-warp reductions

Both arms use the pinned Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, layer zero, sequence length
one, FP16, O3 and fast math disabled. The compiler is main after #1019, `e53587a910d9e6372814800e22e88a06f742fc31`.
Each process uses a fresh tune database, strict measured evidence, no recording, ten warmups and 100 iterations.
Execution order alternates over six fixed pairs. All twelve completed samples are retained, including the loss.

| Pair | Order | Baseline, µs | Candidate, µs |
| --- | --- | ---: | ---: |
| 1 | Baseline then candidate | 67.328 | 67.072 |
| 2 | Candidate then baseline | 67.520 | 67.136 |
| 3 | Baseline then candidate | 67.456 | 66.880 |
| 4 | Candidate then baseline | 67.789 | 66.688 |
| 5 | Baseline then candidate | 67.136 | 67.392 |
| 6 | Candidate then baseline | 67.328 | 66.816 |
| **Arm medians** | | **67.392** | **66.976** |

The reduction is 0.416 µs, or 0.617%, between the arm medians. Five pairs win; the loss costs 0.256 µs. Arm means
are 67.426 and 66.997 µs, a 0.636% reduction. This is a modest result on one shape and card, not an established
advantage over `torch.compile`. The final separate model comparison measures 66.500 µs for Emmy and 54.345 µs for
the compiled reference. Those single-process numbers are not the paired speedup claim.

The pairs time Emmy alone and explicitly report unchecked correctness. Separate model comparisons before and after
them pass both Emmy and `torch.compile` against eager under the unchanged scaled tolerance. All twelve timing
samples have whole-program end-to-end semantics, fourteen launches and the exact expected source inventory.
The candidate changes one CUDA body; the other thirteen complete CUDA objects remain identical. Five fresh-database
strict replays of the stored complete program also pass against eager at 0.001 absolute and relative tolerances,
with zero reported error and the same fourteen selected sources.

The selected kernel keeps sixteen CTAs but uses 32 instead of 128 threads per CTA. Both cooperative reductions
remain, while their cross-warp shared-memory collectives disappear. Shared memory falls from 48 bytes to zero;
registers rise from 32 to 40, with no spills. All cuts and intermediate FP16 boundaries remain. An isolated strict
same-input comparison against the exact accepted Emmy source reports zero error and 2.326 versus 2.738 µs. That
closed kernel has no independent Torch boundary, so this check supplements the complete model validation.

The existing recording command writes the added row's measured cost of 2.349 µs. Fresh databases select it without
pins. A retained control containing only a latency snapshot correctly keeps the old schedule: a snapshot alone is
not selection evidence. All 35 original kernel definitions, 15 routes, 22 measured rows and program/provenance fields
remain unchanged. The later recording comparison selects the candidate itself and is not a second independent
comparison against the old kernel.

Current-layout unsplit projections lose their operation-matched comparisons. The best screened Q candidate costs
6.922 µs against about 6.567 µs for the accepted partial plus finalizer; shared K/V's better candidate costs
8.443 versus 7.118 µs. More independent accumulators and output lanes do not recover the finalizer cost here.
Their experimental catalog extensions remain only in archived patches and correctness evidence. No layout change,
fusion restriction or numerical relaxation is proposed.

### H100: causal skipping and smaller tiles do not pay here

All H100 probes keep the original mask, FP16 boundaries and accepted input shapes. Enabling the existing causal
bound in the accepted one-wave schedule gives 83.180 µs for the layer against an 82.767 µs baseline screen, even
though its attention kernel is slightly faster. This does not justify changing the general policy, which also has
an older head-width-256 regression control.

Two existing smaller schedules allow the causal bound naturally by increasing the grid from 128 to 256 CTAs.
Supported closed-kernel comparisons use the exact accepted production source as their same-input reference. Both
pass strict comparison with zero reported error at unchanged 0.001 absolute and relative tolerances. Their candidate
sources match strict full-program CPU lowering, with all ten unrelated CUDA bodies unchanged. The smaller output
tile additionally preserves all ten unrelated complete CUDA objects, including their launch metadata.

| Attention schedule | Query rows / output channels | Threads | Shared memory | Registers | Candidate / baseline isolated, µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smaller query tile, mma.sync | 32 / 128 | 64 | 64 KiB | 234 | 15.329 / 12.257 |
| Smaller output tile, WGMMA | 64 / 64 | 128 | 64 KiB | 128 | 14.072 / 12.340 |

The accepted schedule uses 64 query rows, 128 output channels, 128 threads, 80 KiB shared memory and 168 registers.
Both candidates have no spills. These isolated losses reject these particular schedules; they do not measure a
whole-layer change or prove that every smaller tile loses. Earlier incomplete model and intermediate-IR attempts
are retained as failures, not timings. The supported comparisons remove that earlier measurement uncertainty.

Packing eight normalization rows per CTA makes both selected normalization kernels slightly slower. Its model JSON
reports a 0.50% lower total, followed by terminal timeout status 124. That single run with a nonzero terminal status
is insufficient for acceptance. The experimental catalog extension is excluded.

The earlier NVIDIA profiles still support investigating bulk TMA staging and producer/consumer warp specialization
on H100. The bounded alternatives here do not establish a smaller scheduling fix for its remaining gap. V100's
unsplit projection question remains tied to weight layout and complete-layer costs. Neither larger change is
implemented or claimed as a future speedup by this phase.

### Controls and retained evidence

Fresh baseline model screens pass on single V100 at both sequence lengths and on H100 prefill. V100 s1 measures
67.644 µs versus 54.963 µs for `torch.compile`; V100 s512 measures 497.664 versus 605.335 µs; H100 s512 measures
82.767 versus 79.642 µs. These are setup controls, not repeated comparisons. Reference tuning differs across
processes, so changes from earlier reports cannot be attributed to Emmy.

The first CPU source audit compared readable rendering with benchmark rendering and wrongly suggested source drift.
Repeating it with matching rendering proves all fourteen V100 CUDA objects and all eleven H100 CUDA bodies equal
the prior accepted/profile sources. Both captures and the correction are retained. No identity change or restamp
is required. Only this experiment's V100 decode golden changes; all other canonical recipe archives stay unchanged.

The cards are a Tesla V100 SXM2 16GB and an H100 80GB HBM3. V100 uses Torch 2.13.0+cu126 and NVCC 12.9.86;
H100 uses Torch 2.14.0+cu130 and NVCC 12.9.41. Both use Transformers 5.14.1 and driver 580.178.04. Measurements
use fresh task-owned runtime and cache directories. Cold
setup timeouts, refused pins, unsupported intermediate-IR attempts and malformed output-path attempts are retained
with their terminal statuses. A100 is stopped with its persistent disk retained. The 4×V100 VM is unused.

V100 qualification and rejected probes are retained in `tuning_v100x1_round3_2026-10-02.tar.gz`. H100 evidence is in
the `tuning_h100x1_round3_diagnostics_2026-10-02`, `tuning_h100x1_round3_m32_priceprobe_2026-10-02` and
`tuning_h100x1_round3_pvn64_priceprobe_2026-10-02` directories of the consolidated H100 archive. The source audit is in
`tuning_round3_source_audit_2026-10-02.tar.gz`. Each archive includes a verified checksum manifest. The edited
experiment golden passes fresh lowering. Final local CPU validation passes: 5,776 tests passed and 1,305 skipped
in 412.19 seconds. Lint passes. GPU correctness is established by the separate V100 and H100 checks above.

## Matched NVIDIA profiles of the remaining gaps (2026-10-02)

These captures compare the accepted H100 prefill and single-V100 decode selections with the kernels actually used
by `torch.compile`. The built-in profiling child runs Emmy and eager PyTorch once; it does not collect the compiled
reference. The diagnostic commands therefore wrap the original model benchmark with Nsight Systems and Nsight
Compute. The pinned model revision, shape, precision, source and golden evidence remain unchanged.

Systems attribution uses complete ordered graph replays, excluding eager execution, Inductor compilation trials
and the benchmark's repeated single-kernel graphs. Kernel boundaries differ, so an operation's comparison includes
all kernels that produce its result. Nsight Compute captures use the same replay, cache and clock settings on both
sides of each comparison. Isolated kernel replay does not preserve whole-graph cache behavior; its durations are
not new whole-layer benchmark results. NVIDIA documents these replay and cache effects in its
[profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

### H100: attention work and data movement

The reference has 13 launches: vendor GEMMs, cuDNN attention and Triton normalization/elementwise kernels. Emmy has
11 after the accepted shared K/V change. In complete late graph replays, attention takes 12.672 µs versus 10.272 µs;
Q takes 6.944 versus 5.952 µs, and gate plus up takes 19.200 versus 17.664 µs. Shared K/V wins, 8.896 versus
9.792 µs. Both reference gate/up kernels are included. These are diagnostic kernel-group medians.

Matched cold attention captures use four occurrences per launch configuration, kernel replay, node profiling,
17 metric passes, cache flushing and no profiler clock adjustment. Both Nsight Compute invocations pass the model's
scaled correctness checks. Medians below describe isolated replay, not whole-layer cache behavior.

| Attention metric | Emmy | cuDNN reference |
| --- | ---: | ---: |
| CTAs / threads per CTA | 128 / 128 | 64 / 384 |
| Tensor work, GFLOP | 2.147 | 1.342 |
| Load/store warp instructions | 149,504 | 15,872 |
| Total warp instructions | 3,870,720 | 1,053,598 |
| DRAM read, MB | 4.233 | 4.220 |
| Requested L2 traffic, MB | 23.699 | 15.564 |
| Active / eligible warps per scheduler active cycle | 0.997 / 0.328 | 2.286 / 0.359 |
| Isolated diagnostic duration, µs | 15.072 | 12.912 |

cuDNN executes 37.5% less tensor work. This calculation weights each instruction by its matrix dimensions;
the raw instruction counts alone would exaggerate the difference. It is consistent with skipping causal tiles,
while Emmy's existing schedule walks all 512 keys. The reference loop bounds were not fully reconstructed, so
this does not establish the gain from changing Emmy's causal bound or CTA count.

Disassembly shows bulk TMA loads/stores and separate producer/consumer warps in cuDNN. Eight consumer warps receive
232 registers each; one producer warp remains active after three other reserved producer warps exit. Emmy uses
individual asynchronous copies and a uniform register allocation. The reference executes 9.42 times fewer
load/store instructions, but reads almost the same DRAM bytes. Instruction accounting differs for TMA, so this is
not a ninefold traffic saving. Its eligible-warp count improves only 9%, and its barrier-stall ratio is higher.
Neither low occupancy nor barriers alone explains the comparison.

Q supplies an equal-tensor-work control: Emmy executes 20.78 times as many load/store instructions and 2.29 times
as many total instructions as the vendor kernel. Shared K/V already reduces traffic and a launch. Gate/up is
nearly tied in cold replay, 21.216 versus 21.600 µs, despite favoring the reference in the late graph trace.
That difference needs a comparison preserving whole-graph cache locality before another schedule is selected.
The cold reference K/V captures are startup eager launches with the same vendor function, shape, strides and launch
geometry as the compiled graph's external matrix multiplies; they are not captures of warmed compiled invocations.

Both Systems attempts end with a CUDA launch failure during Emmy timing. The second nevertheless retains 264
complete Emmy graphs and 4,843 complete compiled-reference graphs in one worker. The last 100 of each in an
overlapping time window have median graph spans of 82.320 and 83.056 µs, respectively; the instrumented comparison
slightly favors Emmy, while the final unprofiled comparison slightly favors the reference. Kernel-duration sums
are 80.672 and 71.440 µs, leaving very different inter-kernel gaps. Those sums cannot explain the net latency gap
by themselves, and the failed trace is not a successful benchmark.

The next work is bulk TMA staging and producer/consumer warp specialization through the existing schedule
mechanisms, followed by a measured causal-work alternative. Each needs balanced unprofiled whole-layer pairs and
unchanged numerical checks. A blanket increase in CTA count or removal of barriers is not supported by these data.
Raw reports, disassembly, generated sources, commands, software versions, both failed traces and a checksum manifest
are retained under `matched-profile/` in the H100 archive's `tuning_h100x1_round2_matched_profiles_2026-10-02`
directory.

### V100: split projection finalizers

The actual compiled model uses nine Triton kernels, with no cuBLAS kernel in that graph. Emmy uses fourteen kernels.
A successful short Systems run distinguishes that model from a second worker's reconstructed Torch comparison,
which uses some different launch configurations. Complete graph attribution finds the largest deficit in input
normalization plus Q/K/V: 24.831 µs for Emmy versus 16.448 µs for the reference. Emmy wins attention plus O,
18.192 versus 22.272 µs, and loses post-attention normalization plus the MLP, 37.792 versus 32.832 µs.
These are sums of per-node medians in the instrumented trace, not new unprofiled performance results.

Matched cold projection captures use the exact reference function and launch geometry verified in the Systems
trace and generated source. Both sides use kernel replay, flushed caches, base clock control and 18 metric passes.
Each Emmy comparison includes its partial reduction and finalizer; comparing only its main projection would omit
work the reference completes inside one kernel.

| Cold diagnostic, µs | Emmy partial | Emmy finalizer | Reference complete projection |
| --- | ---: | ---: | ---: |
| Q | 9.024 | 2.688 | 8.640 |
| Shared K/V | 9.536 | 3.616 | 9.952 |
| Gate/up | 21.216 | 3.168 | 19.808 |
| Down | 12.160 | 2.912 | 18.592 |

The main Q and K/V kernels read essentially one pass of their weights: about 4.2 MB each. Their isolated times
are close to the reference, while Emmy adds the finalizer. K/V's partial is slightly faster despite much lower
occupancy. These measurements support the extra reduction stage as a concrete cost; they do not establish poor
occupancy or inflated DRAM transactions as the cause.

Gate/up's partial already reaches about 85% of reported DRAM utilization and adds a 3.168 µs finalizer. Down is
faster in the captured comparison, so extra launches are not a universal explanation. Fused work differs: reference
gate/up includes normalization, while reference down includes SiLU and multiplication. Emmy pays a separate
normalization and materializes the half-precision SiLU/product before down. These row differences are not equal-work
speedups and cannot be summed into a whole-layer result.

In that down configuration, the reference executes 24.117 million FP32 FMA thread instructions versus 3.146 million
in Emmy's partial, despite nearly equal DRAM reads. Its source recomputes the FP32 SiLU/product within each output
tile. Materializing Emmy's product once avoids that repeated work, so removing this cut is not an established win.

The down reference above matches the longer Systems run's selected 1,024-CTA, 512-thread configuration. The shorter
run selects a different configuration, and its exact register/shared-memory allocation was not captured by Compute.
The reference attention/O captures likewise match no selected Systems configuration and remain unselected tuning
diagnostics. They do not establish a hardware-counter comparison for that group. Q/K/V and gate/up configurations
match both traces.

The source explains a relevant design difference. Triton reads row-major output-by-reduction weights and completes
the reduction in one CTA. Emmy uses transposed reduction-by-output weights and global split partials. Q writes a
64 KiB partial workspace; K/V writes 128 KiB. Changing the split factor alone previously failed to improve the
whole layer. A legal unsplit realization with an appropriate weight layout is the next targeted comparison.

The reference also retains FP32 Q/K/V and gate/up buffers, and their consumers do not restore intermediate FP16
rounding. Only specific boundaries, including key/RoPE and the final output, store FP16. Emmy preserves its own
explicit FP16 boundaries. This corrects the earlier diagnostic note claiming that all consumer rounding was
retained. Matching the reference's kernel boundaries is not permission to drop those numerical obligations.

All fifteen targeted CLI runs exit successfully, pass the actual-model Emmy and compiled-reference numerical checks,
and reproduce all fourteen accepted Emmy kernel source hashes. The initial combined Compute filter captures an
unselected tuning kernel despite exiting successfully; the initial Systems run times out at 600 seconds. Both are
preserved alongside the successful shorter trace. Copied Torch cache entries retain absolute source paths, and four
old-cache configuration records were changed during profiling; their before/after contents are retained and their
original bytes restored. This cache behavior and the observed tuning variation limit claims about a particular
reference configuration across processes.

Raw reports, exports, sources, launch-selection records, commands and failed attempts are retained in
`tuning_v100x1_round2_matched_profiles_2026-10-02.tar.gz`. The canonical unprofiled recipe archives remain unchanged.

### Validation after profiling

Profiling adds no kernel, golden or prior changes. CI exposed an attention availability test whose placement and
cross-CTA split were still chosen by the prior. The test now pins its intended fused, unsplit realization and
keeps the one-kernel and tensor-core assertions. Focused checks pass on the exact CI merge with Python 3.13.
The full local CPU suite passes after both pins: 5,716 passed and 1,250 skipped in 359.65 seconds.

## Shared K/V prefill and V100 input normalization (2026-10-02)

Sharing K/V production improves prefill on H100, A100, RTX 4090 and RTX 5090. Separating the input RMSNorm statistic
improves V100 decode. H100 has the clearest reduction; the RTX improvements are small. H100 prefill and V100 decode
still do not establish a reliable advantage over `torch.compile`.

Each comparison uses six baseline/candidate pairs with alternating execution order. Both arms use the pinned
Qwen3-0.6B layer and measurement settings described in the baseline below. Every process starts with a fresh tune
database and uses strict evidence without recording new timings. The model comparisons use scaled correctness;
all twelve processes per card pass. Each accepted golden also passes five fresh-process strict replays. The selected
routes need no manual pins. No completed pair is dropped, and every rejected or incomplete probe remains archived.

Whole-layer times are microseconds. The reduction is the difference between the two arm medians divided by the
baseline median. It is not the median of paired differences or a comparison with the historical baseline table.

| Card and shape | Baseline median | Candidate median | Lower latency | Winning pairs | Launches before / after |
| --- | ---: | ---: | ---: | ---: | ---: |
| H100, s512 | 86.765 | 82.518 | 4.90% | 6 / 6 | 12 / 11 |
| A100, s512 | 179.456 | 175.957 | 1.95% | 6 / 6 | 12 / 11 |
| RTX 4090, s512 | 156.501 | 155.904 | 0.38% | 5 / 6 | 12 / 11 |
| RTX 5090, s512 | 129.950 | 129.353 | 0.46% | 6 / 6 | 12 / 11 |
| V100, s1 | 68.288 | 67.396 | 1.31% | 5 / 6 | 14 / 14 |

The losing RTX 4090 pair costs 0.171 µs; the losing V100 pair costs 0.192 µs. V100's arm means are 68.139 and
67.432 µs, a 1.04% reduction. These are modest changes, and another GPU, shape or load can give a different result.
The first RTX 4090 protocol hit its 115-second setup limit before producing a baseline JSON record. Its complete
six-pair protocol restarted with a fixed 300-second process limit; the initial attempt remains in the archive.

### Compiler and evidence changes

Independent output sweeps can now align through coordinates established by shared input loads. This extends the
previous flat-domain reform to the differently ordered row and channel coordinates in prefill K/V. Equal shared
coordinates remain fixed; only the remaining equal-volume domain is flattened and expanded. The mapping must be
injective and preserve output index order. Numerical tests cover one and two shared axes, distinct row/channel
values, incompatible mappings and the existing symbolic and windowed single-axis cases. Fusion remains maximal,
and ordinary cuts still offer separate producers.

Each accepted prefill golden adds one shared K/V kernel, one producer route and one measured schedule row. All
previous programs, kernel definitions, routes and rows remain unchanged. Nine unrelated emitted kernels are
byte-identical between the two arms; Q's complete CUDA body is identical after its generated function and workspace
names are aligned. The recorded shared K/V rows cost 7.350 µs on H100, 19.850 µs on A100, 17.821 µs on RTX 4090
and 13.938 µs on RTX 5090. These costs select the new route through normal evidence.

V100 instead cuts the raw input mean square and applies normalization while reading the Q and shared K/V projections.
This removes the separate normalized input vector without changing the layer's 14 launches. Seven kernel
definitions, four routes and four measured rows are added; all 28 original definitions, 11 routes and 18 rows remain
unchanged. The measured new rows cost 1.933, 4.623, 2.510 and 4.637 µs. Source changes are confined to that statistic,
Q/K/V normalization and consistent K/V channel order, plus generated names and an unused down-projection argument.

The nested cut used to construct this candidate exposed a pin-consumption bug: a parent placement pin could apply
again to the parent's remaining work when another pin targeted a newly created child. The parent remainder now
consumes its decision while explicitly named children retain theirs. A regression test checks the resulting three
pieces instead of four. This affects explicit placement pins; accepted qualification uses unpinned evidence.

All six maintained hardware goldens and nine maintained model goldens remain current without restamping. No
maintained golden or prior weight changes in this round. The five edited goldens belong to this experiment. Existing
measurements are never overwritten to make a slower or changed kernel look unchanged.

### Rejected probes and remaining costs

H100's first, wider shared K/V tile wins six pairs but costs 11.519 µs in isolation, above the old separate K/V sum
of 11.357 µs. It therefore does not win normal evidence selection. The accepted narrower tile costs 7.350 µs and
improves the complete layer more. A100's earlier larger reduction tiles and eight-warp candidate lose; its selected
smaller reduction tile wins all six pairs. RTX 4090's smaller row tile loses, and its eight-warp shared tile reduces
register use from 150 to 104. RTX 5090's eight-warp shared tile ties and its shallower pipeline loses.

V100 profiling shows gate/up near 86% of measured cold DRAM throughput, while Q, K/V and O reach roughly 56–58%
and spend 59–65% of sampled warp time stalled on long scoreboards. Its wider Q split wins only three of six pairs
and is tied on average, so the original split stays. Cutting an input normalization factor instead of the mean square
also ties. Moving the post-attention RMSNorm statistic separately preserves O but makes gate/up repeat more
normalization work; the new gate/up and scalar costs outweigh the saving. Existing evidence continues to select the accepted
input-only change. No mathematical precision or correctness tolerance is changed.

The V100 trace has 14 Emmy launches versus nine for `torch.compile`, including five projection partial/final pairs.
The reference uses a different weight orientation, so its unsplit geometry cannot simply replace those schedules.
A useful next investigation is a legal unsplit projection with coalesced weight access, measured in the whole layer;
the current split-factor trials do not establish that it will win.

The exact-source H100 attention profile has 128 CTAs on 132 SMs, 168 registers per thread and 80 KiB shared memory.
It exposes about one active warp per scheduler and 0.36 eligible warps, with no eligible warp in 63.79% of sampled
cycles. Compute utilization is 27.95%, L2 20.43% and DRAM 6.60%. Fixed-latency waits and barriers outweigh GMMA waits.
Smaller row-tile probes are refused by existing schedule legality before CUDA emission; they provide no timing.
An A100 attention schedule with more, smaller CTAs times out in two 115-second attempts and one fixed 300-second
attempt without producing a timing. These attempts establish no performance result, and the old schedule remains.
The remaining question is whether a supported smaller attention tile can improve latency hiding without adding
more synchronization or changing rounding. Another broad schedule sweep is not supported by this evidence.

RTX 5090 gate/up reaches 45.68% tensor throughput, 33.81% SM utilization and 51.47% L2 throughput, with 384 CTAs,
256 threads, 104 registers and 50,176 total shared-memory bytes. A larger row tile was attempted to reuse each
weight tile across more rows, but its bounded attempt returned no measurement. RTX 4090 profiling is blocked by
`ERR_NVGPUCTRPERM`; no counters were obtained and host permissions were left unchanged. Profiled durations are
diagnostics only, never the unprofiled latency claim.

The qualification evidence retains all paired records, strict repeats, commands, source comparisons and rejected
trials. H100 uses the `tuning_h100x1_round2_2026-10-02` directory in its consolidated archive. The other archives are
`tuning_a100x1_round2_2026-10-02.tar.gz`, `tuning_rtx4090x1_round2_2026-10-02.tar.gz`,
`tuning_rtx5090x1_round2_2026-10-02.tar.gz` and `results_v100x1_round2_diagnostics_2026-10-02.tar.gz`.

## Final five-card recipe after the second round (2026-10-02)

All ten model comparisons and all fifty strict golden replays pass. These are the unchanged two-shape recipe on
each exact card, with the accepted evidence and the same protocol as the baseline. Captured whole-forward model
latencies are microseconds. The balanced pairs above establish the improvements; this table validates the final
selections and retains the contemporaneous reference timings.

| Card | s1 Emmy | s1 `torch.compile` | s512 Emmy | s512 `torch.compile` | Launches s1 / s512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A100 40GB | 49.688 | 55.896 | 176.333 | 180.120 | 14 / 11 |
| H100 80GB | 29.200 | 31.729 | 82.942 | 82.037 | 14 / 11 |
| V100 SXM2 16GB | 67.648 | 60.849 | 496.640 | 637.012 | 14 / 21 |
| RTX 4090 | 24.545 | 29.163 | 155.502 | 162.778 | 14 / 11 |
| RTX 5090 | 20.473 | 24.591 | 129.344 | 135.717 | 16 / 11 |

The strict golden replays use different inputs and a different reference path. Their medians and full ranges are
validation results, not the values to compare against the model's `torch.compile` column.

| Card | s1 median [range], µs | s512 median [range], µs |
| --- | ---: | ---: |
| A100 40GB | 50.712 [50.404–51.054] | 175.787 [174.763–176.299] |
| H100 80GB | 24.506 [23.907–25.088] | 83.877 [83.508–84.241] |
| V100 SXM2 16GB | 67.968 [67.464–68.367] | 496.640 [491.520–499.200] |
| RTX 4090 | 24.625 [24.576–24.726] | 155.467 [154.770–156.160] |
| RTX 5090 | 20.472 [20.471–20.473] | 129.783 [129.308–130.139] |

Within each shape, the model run and all five strict repeats have identical ordered CUDA source hashes, schedules
and shared-memory sizes. Changed shapes match the accepted candidates; unchanged shapes match the baseline.
H100's final recipe precedes the parent placement-pin fix. Fresh strict compiles of both H100 shapes reproduce
byte-identical complete CUDA under the integrated parent in the same compilation context, confirming the fix leaves
these unpinned selections unchanged. Later formatting changes preserve the Python ASTs.

Every canonical archive contains two succeeded system-only experiment records, raw command artifacts, source
provenance and logs. Earlier canonical snapshots remain in Git at the round's base; the newly measured baselines
remain in the separate archives below. The table's V100 archive was replaced by the latest replay above; its
2026-10-02 values in this section remain historical. That earlier V100 rental was terminated.

| Card | Canonical archive | Root member | Executed source |
| --- | --- | --- | --- |
| A100 | `results_a100x1.tar.gz` | `2026-10-02_09-21-02/` | `9c81582c4` |
| H100 | `results_h100x1.tar.gz`, bundle `results_h100x1` | `2026-10-02_08-35-50/` | `8872d9346` |
| V100 | `results_v100x1.tar.gz` | `2026-10-06_04-19-46/` | `00a9195e6` |
| RTX 4090 | `results_rtx4090x1.tar.gz` | `2026-10-02_09-13-19/` | `eb33f345c` |
| RTX 5090 | `results_rtx5090x1.tar.gz` | `2026-10-02_09-17-30/` | `9c81582c4` |

All five use Transformers 5.14.1. The recorded Torch versions and CUDA compilers differ between hosts, so comparisons
between cards include those software differences. Exact GPU UUIDs, clocks, operating systems and package freezes
are preserved in the matching records and artifacts.

| Card | Host | Torch | nvcc | Driver |
| --- | --- | --- | --- | --- |
| A100 | `bench-keep-a100-0921-1621-6784` | 2.14.0 | 12.9.41 | 580.173.02 |
| H100 | `bench-gb-h100-0924-1252-99aa` | 2.14.0 | 12.9.41 | 580.178.04 |
| V100 | `riftvm`, single SXM2 card | 2.13.0+cu126 | 12.9.86 | 580.178.04 |
| RTX 4090 | `riftvm`, single RTX 4090 | 2.14.0 | 13.3.73 | 580.159.03 |
| RTX 5090 | `kenshin` | 2.13.0 | 13.0.88 | 580.173.02 |

Final validation passes the full CPU suite with 5,716 passed and 1,250 skipped in 367.03 seconds, including
maintained-golden freshness and prior reproduction. The full RTX 5090 suite passes with eight workers: 6,583 passed
and 398 skipped in 1,117.76 seconds. Its initial missing test-entrypoint setup failure is retained separately.
All ten experiment goldens pass a final freshness check without rewriting anything. Lint passes after formatting
the new code and tests. The RTX 4090 tuning archive retains the CPU, lint and freshness logs; the RTX 5090 tuning
archive retains the complete GPU suite and its setup provenance.

## Five-card baseline for the second optimization round (2026-10-02)

All ten model comparisons and all fifty strict golden replays pass on the merge of PR #1011,
`2ffe2b81be3a24f27a9dbfb10a5b274527a677f2`. H100 prefill and V100 decode remain slower than the same-input
`torch.compile` reference. These measurements validate the pinned routes before this round's changes.

The unchanged recipe runs Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, layer 0, with sequence
lengths 1 and 512, O3, fast math disabled, 10 warmups and 100 iterations. Every process starts with a fresh tune
database, requires measured evidence and disables new timing writes. Model comparisons use the scaled correctness
check. Each of the five subsequent golden replays uses strict correctness. The table reports captured whole-forward
model times in microseconds; the three backends receive identical inputs within each process.

| Card | s1 Emmy | s1 `torch.compile` | s512 Emmy | s512 `torch.compile` | Launches s1 / s512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A100 40GB | 51.769 | 54.491 | 178.347 | 180.120 | 14 / 12 |
| H100 80GB | 29.370 | 31.673 | 86.781 | 80.486 | 14 / 12 |
| V100 SXM2 16GB | 68.335 | 65.009 | 500.736 | 639.631 | 14 / 21 |
| RTX 4090 | 24.651 | 28.978 | 156.976 | 162.343 | 14 / 12 |
| RTX 5090 | 20.469 | 36.849 | 129.833 | 135.189 | 16 / 12 |

The strict golden replays use different inputs and a different reference path from the model comparison. Their
medians and full ranges are validation results, not substitutes for the model's `torch.compile` comparison.

| Card | s1 median [range], µs | s512 median [range], µs |
| --- | ---: | ---: |
| A100 40GB | 50.783 [50.615–51.314] | 178.176 [177.835–179.883] |
| H100 80GB | 24.538 [24.363–24.738] | 88.315 [87.883–88.685] |
| V100 SXM2 16GB | 67.704 [67.584–68.006] | 497.664 [495.616–502.272] |
| RTX 4090 | 24.699 [24.571–24.751] | 156.315 [155.819–156.613] |
| RTX 5090 | 20.471 [20.470–20.478] | 130.025 [129.605–130.592] |

All cards use task-owned source checkouts and runtime extensions. Initial A100 and RTX 4090 invocations inherited
an older runtime that rejected the `dependent_launch` field; those setup failures remain in the raw evidence.
The reported runs rebuild the runtime from the exact source above. RTX 5090 was rebuilt and repeated too.
Its ordered kernel sources, schedules and shared-memory sizes match the previous canonical qualification on both
shapes. Its reference timing varies substantially: decode measured 24.592 µs in the first invocation, 36.849 µs in
the rebuilt-runtime recipe and 30.810 µs in a supplemental check. Accepted changes therefore need contemporaneous
balanced pairs, not a comparison with a historical reference column.

A supplemental strict model-form prefill check on the unchanged RTX 5090 baseline fails on 5 of 524,288 outputs,
with maximum absolute error 0.00390625 and mean absolute error 0.0000539. The corresponding scaled model comparison
and all five strict golden replays pass. No tolerance changes or retry-until-pass procedure were used. Model tracing
samples new inputs per process; the CLI's seed controls the golden reference but does not seed those model inputs.
The failed model's input tensor is not persisted by the CLI, so this supplemental sample cannot be replayed exactly.
The RTX 4090 supplemental strict prefill check timed out before returning a verdict; that attempt supplies no
correctness or performance result. Supplemental strict decode checks pass on both cards.

The raw baseline archives preserve the system-only experiment records, command logs, JSON measurements and
provenance. A100, RTX 4090 and RTX 5090 archives also retain the initial environment qualification attempts.
Only the single V100 SXM2 machine was used.

| Card | Archive | Successful recipe root |
| --- | --- | --- |
| A100 | `tuning_a100x1_round2_baseline_2026-10-02.tar.gz` | `2026-10-02_06-51-38/` |
| H100 | `results_h100x1.tar.gz`, bundle `tuning_h100x1_round2_baseline_2026-10-02` | `2026-10-02_06-30-53/` |
| V100 | `results_v100x1_round2_baseline_2026-10-02.tar.gz` | `2026-10-02_06-31-09/` |
| RTX 4090 | `tuning_rtx4090x1_round2_baseline_2026-10-02.tar.gz` | `2026-10-02_06-47-49/` |
| RTX 5090 | `tuning_rtx5090x1_round2_baseline_2026-10-02.tar.gz` | `2026-10-02_06-39-47/` |

## Shared K/V decode producers (2026-10-01)

Sharing the K/V producer lowers whole-layer decode latency on A100, H100, V100 and RTX 4090. The selected route keeps Q
separate and replaces the four K/V kernels with two, reducing the layer from 16 launches to 14. V100 still trails the
same-input `torch.compile` reference. The equivalent RTX 5090 trial saves only 0.02–0.04 µs per pair, so that card
keeps its existing selection.

These are contemporaneous baseline and candidate processes on the same card, using Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, layer 0, sequence length 1, deployable O3, `EMMY_FAST_MATH=0`,
10 warmups and 100 iterations. Each process starts with a fresh tune database, requires measured evidence and
disables new timing evidence. All eager-referenced Emmy and `torch.compile` checks pass. A100, RTX 4090 and RTX 5090
use strict correctness throughout; V100's first pair and all H100 pairs use the scaled check. V100's remaining two
pairs also pass strict correctness. No tolerance changed. H100's manual commands used the unversioned model name;
the cache audit records the same revision in `refs/main` since September 30, before these trials. Its final recipe
pins that revision explicitly.

Whole-layer Emmy times below are microseconds. The reduction compares the two arm medians; it is not a comparison
with the earlier baseline table. Every individual timing is retained in the corresponding raw archive.

| Card | Pairs | Baseline median | Candidate median | Lower latency | Selection |
| --- | ---: | ---: | ---: | ---: | --- |
| A100 40GB | 3 | 53.895 | 51.769 | 3.94% | shared K/V |
| H100 80GB | 12 | 29.158 | 27.739 | 4.87% | shared K/V |
| V100 SXM2 16GB | 3 | 72.431 | 67.704 | 6.53% | shared K/V |
| RTX 4090 | 3 | 26.349 | 24.676 | 6.35% | shared K/V |
| RTX 5090 | 3 | 20.490 | 20.456 | 0.16% | unchanged |

All ten unrelated kernel sources remain byte-identical. The two Q kernels retain identical bodies, arguments,
launch geometry and shared memory; only their generated function names change. Every arm keeps the same sources,
schedules and shared-memory sizes across repeats. A100's reference stays between 57.6 and 58.3 µs. V100's reference
medians are 59.231 µs for baseline processes and 58.500 µs for candidate processes; that drift is smaller than the
4.726 µs separation between Emmy medians. RTX 4090's three paired gains are 1.641, 1.748 and 1.763 µs; their median
is 1.748 µs, while the difference between arm medians reported above is 1.674 µs. Its reference varies from 27.331
to 29.249 µs. RTX 5090's tiny reduction does not establish a useful layer improvement.

H100 needed more sampling: the first six pairs had four wins and a 0.609 µs separation between arm medians. A fixed
set of six additional pairs balanced execution order; there was no further sampling. Across all twelve pairs, the
candidate wins ten and loses two, by 0.751 and 0.654 µs. The median paired gain is 1.549 µs; no sample is excluded.
The `torch.compile` medians stay at 31.674 and 31.670 µs in baseline and candidate processes. Every pre-run GPU
process list is empty, and all captures use the same UUID and 1980/2619 MHz clocks, at 34–40°C. The raw protocol
retains all 24 timings and their execution order, including both losses.

The compiler can now give independent outputs with equal iteration domains a common coordinate. A rectangular
domain maps to the flat coordinate by quotient and remainder. Existing normalization then combines the two producer
reductions. This applies only where output ownership and binding make the substitution legal; ordinary cuts still
offer separate producers. No fusion gate, grouped cut, new schedule family or benchmark implementation was added.
The deployed K/V route uses the existing split reduction and cooperative partial schedule.

Each selected experiment golden adds three kernel identities, two routing decisions and two measured schedule rows.
Every previous program, kernel, route and row is retained unchanged. Measurements come from the exact named card.
Fresh unpinned strict-evidence replay selects the route through those rows. A100's paired source is `55ea5c75e`,
equivalent to parent compiler integration `9989b6b37`; V100 used `4fd360ef0` and `bd3132c35` over the baseline repairs.
After merging main `754d1afa6`, fresh strict compiles on A100 and V100 retained all 14 ordered CUDA sources and launch
signatures. RTX 4090 and RTX 5090 pairs ran on that merged source at `c902cfbb4`. H100's paired remote source is
`1b44919c9`, equivalent to `59e3d8d66` over its baseline repairs.

Several rejected probes remain in the evidence. V100's eight-output-lane K and V schedules lost in full-layer runs.
A smaller 128-thread K/V partial also lost to the selected 256-thread partial, 4.6 versus 4.3 µs in the bounded
isolated comparison, so it was not promoted. A standalone normalized child initially had a different parent identity
and output order; its timings were never used as full-layer evidence. The accepted route is derived from the actual
full-layer parent. Global pins that changed unrelated decisions were likewise rejected before acceptance.

`tuning_a100x1_2026-10-01.tar.gz`, `tuning_v100x1_2026-10-01.tar.gz`, `tuning_rtx4090x1_2026-10-01.tar.gz` and
`tuning_rtx5090x1_2026-10-01.tar.gz` retain the paired JSON, logs, task databases, working goldens, source audits,
failed probes and exact command protocols under `2026-10-01-a100/`, `2026-10-01-v100/`, `2026-10-01-rtx4090/` and
`2026-10-01-rtx5090/`, respectively. The V100 work used the single SXM2 card throughout.
H100's corresponding evidence is under `2026-10-01-h100/shared-kv/` in the consolidated archive's
`tuning_h100x1_2026-10-01` directory, alongside the prefill profiling and rejected trials described below.

## Final five-card recipe after route selection (2026-10-01)

All ten model comparisons and all 50 strict golden replays pass after selecting the four shared K/V routes.
Each model comparison uses the explicitly pinned revision above, the same inputs
for all three backends, O3, fast math disabled, 10 warmups and 100 iterations. Each of the five following replays
uses a fresh tune database, strict correctness and strict evidence, without recording new measurements.

Captured whole-forward latency is in microseconds. These final checks validate the selected routes; the interleaved
pairs above establish the improvement over the previous selections.

| Card | s1 Emmy | s1 `torch.compile` | s512 Emmy | s512 `torch.compile` | Launches s1 / s512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A100 40GB | 51.086 | 56.247 | 177.664 | 181.541 | 14 / 12 |
| H100 80GB | 27.255 | 31.710 | 87.347 | 81.992 | 14 / 12 |
| V100 SXM2 16GB | 68.335 | 62.498 | 500.224 | 641.360 | 14 / 21 |
| RTX 4090 | 24.626 | 29.096 | 156.160 | 162.816 | 14 / 12 |
| RTX 5090 | 20.549 | 24.591 | 129.572 | 146.398 | 16 / 12 |

The separate strict golden replays give these medians and full ranges, also in microseconds. Their input and timing
path differs from the model comparison, so they are not the numbers to compare with `torch.compile`.

| Card | s1 median [range] | s512 median [range] |
| --- | ---: | ---: |
| A100 40GB | 50.603 [50.214–50.935] | 178.688 [177.835–179.541] |
| H100 80GB | 24.190 [23.873–24.614] | 88.448 [88.221–88.637] |
| V100 SXM2 16GB | 67.644 [67.644–68.066] | 496.640 [491.520–498.688] |
| RTX 4090 | 24.755 [24.676–24.773] | 156.160 [155.989–156.501] |
| RTX 5090 | 20.543 [20.532–20.559] | 129.916 [129.800–130.892] |

Within each row, all six processes keep identical ordered CUDA sources, schedules and shared-memory sizes. Each
selected decode row matches its accepted candidate; RTX 5090 matches its unchanged baseline. Prefill sources match
the baseline on four cards. V100's one differing prefill source only renames local coordinates: its addresses,
arguments, launch geometry and shared memory remain equivalent. Compiling the old source with only the coordinate
reform reproduces that change, so it is attributable to this work rather than the intervening main merge. Its other
20 prefill sources remain byte-identical. The exact old, reform-only and final sources are in the V100 tuning archive.

At the end of this earlier round, the canonical result archives held the runs below. These roots remain in Git
history; the current V100 root is recorded at the top of this report. Each run had two succeeded system-only
experiment records, raw artifact bundles and logs. The earlier baseline runs remain in the baseline archive
described below. No benchmark timing was copied into an existing reference row to hide a regression.

| Card | Archive | Root member | Executed source |
| --- | --- | --- | --- |
| A100 | `results_a100x1.tar.gz` | `2026-10-01_20-14-59/` | `c902cfbb4` |
| H100 | `results_h100x1.tar.gz` | `2026-10-01_22-17-55/` | `0c7a36e8a` |
| V100 | `results_v100x1.tar.gz` | `2026-10-01_20-17-08/` | `ab9647672` |
| RTX 4090 | `results_rtx4090x1.tar.gz` | `2026-10-01_20-21-25/` | `e56914c03` |
| RTX 5090 | `results_rtx5090x1.tar.gz` | `2026-10-01_20-15-51/` | `c902cfbb4` |

H100's package freeze renders the task clone's inherited local Git origin in its editable requirement. The raw
provenance audit verifies that the neutral-directory import, installed editable path and executable shebangs all
resolve the intended task checkout at the recorded commit. The misleading origin URL did not select different
benchmark code. The RTX 4090 tuning archive also retains an initial staging failure from a dirty copied golden;
that attempt ran no GPU benchmark. The canonical run began from the clean committed source.

Finalization corrected coordinate substitution so that a self-referenced coordinate keeps its original parameter
position. The original NVFP4 and RMSNorm corpus cases then passed on RTX 5090 and A100 without changing either
case. Qualification of this final compiler preserves the measured Qwen evidence: the actual model comparison was
repeated on A100, RTX 4090 and RTX 5090, with every ordered source hash, schedule and shared-memory size unchanged.
Fresh H100 model-form compiles likewise reproduce all sources and launch configurations.

V100 decode retains all 14 source hashes. In prefill, the fix restores the one renamed kernel above to the exact
source of the successful baseline; all 21 launch configurations, arguments and address expressions remain equivalent.
Thus 140 of the 141 source hashes across the ten workloads match the final snapshots, and the remaining difference
is this coordinate rename back to the baseline. Both V100 shapes also pass one additional strict replay with one
warmup and one iteration. These short checks establish correctness, not new performance evidence. The three-card
records are under `post-fold-validation/` in their tuning archives; H100's are under `fold-source-check/`, and V100's
archive retains the exact pre-fix and fixed CUDA graphs, source diff and strict replay records.

Main's BF16/FP4 and runtime changes were subsequently merged through `c536afe4e`. Rebuilt task-owned runtimes at
`0c7a36e8a` pass all ten workloads. A100, RTX 4090 and RTX 5090 repeat the actual model comparisons with scaled
correctness, strict evidence, 10 warmups and 100 iterations; every source hash, schedule and shared-memory size
matches the canonical results. V100 repeats both strict 1/1 correctness checks, with its complete lowered CUDA
graphs unchanged from the fixed compiler. H100 reruns the whole two-row recipe, including all ten strict repeats;
its sources, schedules and observed launch geometry remain unchanged. The H100 tables and canonical archive above
now use that latest run. Its preceding `2026-10-01_20-56-48` archive is retained byte-for-byte under
`prior-final-replay/prior-canonical-results.tar.gz` in the H100 tuning archive. The new qualification records are
under `post-main-validation/` for the three-card checks, `2026-10-01-upstream-qualification/` for V100, and
`merged-runtime/` for H100. These checks preserve the accepted comparison; they do not select another candidate.

## FP4 correctness found during finalization (2026-10-01)

The full RTX 5090 suite exposed twelve FP4 accuracy failures after merging main. A separate clean checkout of
`c536afe4e`, with its own rebuilt runtime and the same GPU and dependencies, reproduces the three representative
errors exactly. TMA and cp.async outputs still agree bit for bit. Disabling fast math or enabling precise division
passes those controls, identifying an approximate quotient crossing an e2m1 encoding boundary.

The fix at `b59142774` gives static FP4 encoding an explicit round-to-nearest f32 divide, retaining the f32 divisor
and fast math elsewhere. All twelve original failures pass at default fast math and the suite's O1 setting, with
unchanged tolerances. All 431 repository golden freshness checks, nine FP4 realization-case freshness checks and
all ten benchmark golden checks pass without restamping. This repair changes no selected Qwen route; the benchmark
uses unquantized weights. FP4 throughput was not measured in this experiment. The RTX 5090 tuning archive retains
the failed run, clean-main controls and corrected checks under `fp4-gate/`.

The complete RTX 5090 suite at the same source passes with eight workers: 6,460 passed and 394 skipped in
1,252.12 seconds. An earlier 32-worker run lost one worker during the GDN state handoff/reset test. Its replacement
passed that test, as did an isolated rerun and the complete eight-worker run. The cause of the exit remains unknown;
the archive retains both full logs, the isolated check and the available system events. The CPU suite also passes
at this source: 5,621 passed and 1,233 skipped.

## Five-card baseline before the next optimization round (2026-10-01)

The current pinned kernels still pass correctness on all five exact cards. Eight of the ten same-input Hugging Face
layer comparisons favor Emmy. V100 decode remains slower than `torch.compile`, and H100 prefill retains a smaller
loss. A100 prefill is close enough to parity that its lead is timing-sensitive. These measurements validate the
existing selections; they do not measure a new compiler optimization.

The target is Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`, layer 0, sequence lengths 1 and
512. Every model-form process compares eager, `torch.compile` and Emmy on the same inputs, with deployable O3,
`EMMY_FAST_MATH=0`, 10 warmups and 100 iterations. The committed exact-card golden supplies measured evidence.
Five fresh-process golden replays follow each model comparison, with strict correctness and strict evidence. Each
repeat has a separate tune database and disables recording new measurements, so an early repeat cannot change a
later repeat's kernel selection.

Captured whole-forward latency in microseconds. Ratio is `torch.compile` / Emmy; above 1 means Emmy is faster.

| Card | s1 Emmy | s1 `torch.compile` | s1 ratio | s512 Emmy | s512 `torch.compile` | s512 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5090 | 20.534 | 25.892 | 1.261× | 130.160 | 138.625 | 1.065× |
| RTX 4090 | 26.359 | 29.180 | 1.107× | 156.501 | 162.992 | 1.041× |
| A100 40GB | 53.356 | 57.534 | 1.078× | 179.541 | 181.409 | 1.010× |
| H100 80GB | 30.051 | 31.650 | 1.053× | 86.757 | 81.869 | 0.944× |
| V100 SXM2 16GB | 71.360 | 58.628 | 0.822× | 501.760 | 643.811 | 1.283× |

The five strict golden replays give the following median and full range, in microseconds. These are a separate
input and timing path; use the same-input model-form table above for comparisons with `torch.compile`.

| Card | s1 median [range] | s512 median [range] |
| --- | ---: | ---: |
| RTX 5090 | 20.531 [20.528–20.553] | 129.844 [129.480–130.472] |
| RTX 4090 | 26.440 [26.347–26.458] | 156.315 [155.819–156.331] |
| A100 40GB | 53.207 [52.470–53.895] | 179.541 [178.176–180.053] |
| H100 80GB | 25.600 [25.188–25.715] | 88.707 [88.275–88.851] |
| V100 SXM2 16GB | 71.552 [71.270–72.000] | 499.200 [491.008–501.760] |

All ten recipe rows succeeded, and all 50 strict replays passed. Every replay matches its paired model run's ordered
CUDA source hashes, schedules and shared-memory sizes. Decode uses
16 launches on each card; prefill uses 12 on A100, H100, RTX 4090 and RTX 5090, and 21 on V100. Both compiled
model-form backends pass the scaled accuracy check. The strict golden checks use the existing tolerance without
changes. Historical speedup ratios are not a controlled compiler comparison: the software environment and reference
timings have changed, especially on RTX 5090. Candidate acceptance needs contemporaneous baseline and candidate
processes, not a comparison with a previous day's reference time.

The first run exposed two validation problems. The format change in #1007 had left all ten benchmark goldens
unreadable. Conversion through the preceding compiler's importer recovered 144 per-kernel measurements and 61
routing decisions without new measurements. The 41 older aggregate timings remain in the migration audit archive;
they are not per-kernel performance rows in the current format. All ten converted files pass fresh-lowering checks.
Automatic golden replay also treated a descendant's schedule as whole-target pins when its row supplied the target
name. That produced failing extra prefill variants despite correct model-form and greedy golden results. Automatic
replay now uses descendant rows as measured evidence and only pins rows that measure the complete target.

The initial failed attempts remain in the raw archive. Their timings were not recorded over the committed evidence.
V100's first prefill replay was interrupted while the CPU was compiling placement alternatives; later complete
repeats show that the delay was compilation, not a GPU hang. A copied V100 virtual environment also retained old
launcher paths. Its benchmark used the intended task code, as a neutral-directory import audit confirmed. Both the
original and task environment registrations were restored and verified, with no other package changes. H100's
benchmark used the correct task interpreter, but its original package-freeze command used an old pip launcher.
The raw freeze is preserved beside a separate audit and corrected freeze. The recipe now captures packages through
the benchmark's Python interpreter.

The baseline source is `5694af721`; V100 uses the equivalent cherry-picked changes at `dea05fd94`. The exact GPU
UUIDs are unchanged from the September 30 table below. Package freezes and system records retain the full environment.
All cards use Transformers 5.14.1. The compiler toolkit and PyTorch package versions are listed separately because
they need not use the same CUDA libraries.

| Card | PyTorch package | Triton | nvcc | Driver |
| --- | --- | --- | --- | --- |
| RTX 5090 | 2.14.1 | 3.8.0 | 13.0.88 | 580.173.02 |
| RTX 4090 | 2.14.1 | 3.8.0 | 13.3.73 | 580.159.03 |
| A100 40GB | 2.14.1 | 3.8.0 | 12.9.41 | 580.173.02 |
| H100 80GB | 2.14.0 | 3.8.0 | 12.9.41 | 580.178.04 |
| V100 SXM2 16GB | 2.13.0+cu126 | 3.7.1 | 12.9.86 | 580.178.04 |

Baseline snapshots are retained in the Git LFS archive `tuning_baseline_2026-10-01.tar.gz`. Each root below contains
its two system-only
`<variant>.experiment.yaml` records, `<variant>/torch-compile/model.json`, five
`<variant>/verification/repeat-N` JSON files and their status files, the working golden, package freeze and logs.

| Card | Root member within the baseline archive | Run ID |
| --- | --- | --- |
| RTX 5090 | `2026-10-01/baseline/rtx5090/2026-10-01_17-14-41/` | `20261001T171441Z` |
| RTX 4090 | `2026-10-01/baseline/rtx4090/2026-10-01_17-24-20/` | `20261001T172420Z` |
| A100 | `2026-10-01/baseline/a100/2026-10-01_17-28-34/` | `20261001T172834Z` |
| H100 | `2026-10-01/baseline/h100/2026-10-01_17-17-59/` | `20261001T171759Z` |
| V100 | `2026-10-01/baseline/v100/2026-10-01_17-12-12/` | `20261001T171212Z` |

`tuning_baseline_2026-10-01.tar.gz` retains these baseline runs, the initial terminal failed runs, and the environment
audits under `2026-10-01/{baseline,initial,provenance}/`. `tuning_migration_2026-10-01.tar.gz` retains the original
goldens and the import audit. No user-owned GPU instance was stopped or deleted.

## H100 prefill profiling and bounded trials (2026-10-01)

Three schedule trials did not improve the whole layer. A fourth trial removed unused asynchronous copies and
measured a small gain, but the gain depended on measurement order and did not justify the shared codegen change.
The H100 prefill selections remain unchanged. All comparisons use the same pinned model revision and existing
correctness tolerance as the baseline above.

The K-tile and gate/up trials each alternated three baseline and three candidate processes, with 10 warmups and
100 iterations. Every process passed scaled eager correctness and launched 12 kernels. Ordered source hashes,
schedules and shared-memory sizes prove that only the intended kernel changed. Each process used a fresh tune
database and disabled new timing evidence. Whole-layer Emmy times below are microseconds; brackets contain all
three measurements in execution order.

| Change | Baseline | Candidate | Baseline median | Candidate median |
| --- | --- | --- | ---: | ---: |
| K tile width 128 to 64 | [87.048, 86.803, 87.160] | [87.061, 87.773, 87.125] | 87.048 | 87.125 |
| Gate/up plain TMA staging | [86.891, 86.288, 87.568] | [87.749, 88.221, 87.658] | 86.891 | 87.749 |

The K trial doubled its launch from 64 to 128 blocks, holding work, staging and rasterization fixed. It produced
no reliable gain. Gate/up changed from the recorded asynchronous staging to plain two-stage TMA while keeping
its tile, work and rasterization choices. Its isolated kernel became faster, but the layer became slower in all
three pairs. A separate Q-projection TMA probe changed only Q, passed scaled correctness, and measured Emmy at
89.795 µs against `torch.compile` at 82.274 µs with 5 warmups and 20 iterations. Its isolated Q kernel also became
slower, so this candidate stopped before repeated pairs. No schedule candidate was recorded into a golden.

The paired Nsight Systems trace locates costs across several parts of the layer. Q/K/V kernel durations total
about 19.9 µs for Emmy and 16.3 µs for the vendor path. Attention contributes another roughly 2.4 µs difference.
The remaining projections, normalization, MLP and output work account for about 5 µs. Vendor gate/up spans two
kernels, about 7.3 and 10.5 µs; comparing Emmy's roughly 19.3 µs fused kernel with only the latter would overstate
the gap. Launch gaps favor Emmy by about 8 µs in this trace. These diagnostic timings include profiler overhead;
the unprofiled, same-input whole-layer results remain the performance comparison.

An exact-source Nsight Compute diagnostic matched all 12 baseline source hashes and schedules. Its Q kernel
launches 256 blocks of 128 threads, with 64 registers per thread and 64 KiB of shared memory. The counters report
244,736 LSU instructions and 65,536 tensor-pipe instructions, about 29% SM throughput and 16% DRAM throughput,
and no shared-memory bank conflicts. The source issues 24 asynchronous copies per thread during the final three
loop iterations whose results are never consumed. Older counter captures lack exact source proof and are retained
as unattributed diagnostics. Counter-run durations around 10 µs, including a repeat without cache flushing, must
not be substituted for the roughly 7 µs warm Q duration in the Systems trace.

The copy-removal prototype guarded those unused transfers while preserving every commit and wait. Six paired
whole-layer comparisons passed scaled correctness, kept all 12 schedules and changed only the seven eligible
kernel sources. Five pairs favored the candidate; one differed by only 0.024 µs in the other direction. Pooled
medians were 86.981 µs for baseline and 86.697 µs for the candidate, a 0.284 µs improvement (0.33%). The first
three baseline-first pairs showed a 0.717 µs median difference; three additional candidate-first pairs showed
0.149 µs. Those reversed runs recorded an idle GPU before each process, the same 1980/2619 MHz clocks, 35–37°C,
and 124–127 W. Both experiment goldens remained fresh. This small, order-sensitive gain is preserved as a finding;
the code change was reverted.

The K trial used source `5694af721`; later trials used `2daed32f`, the H100 cherry-pick of the partial-pin repair.
Every control reproduced the validated baseline kernels. The H100 archive's `tuning_h100x1_2026-10-01` directory,
under `2026-10-01-h100/`, preserves the profiles, exact commands, source proofs, trial JSON/log/database files, failed
probes, copy-removal patch and system snapshots. A tile-only pin initially hit the partial-pin bug, another probe
lacked nvcc on its SSH path, and a separate work/staging probe failed strict accuracy before timing. Those failed
probes supply no performance result and do not change the headline tolerance.

The prefill K/V producers need a different coordinate alignment from decode: V sweeps `(1024, 512)`, while K
sweeps `(8, 512, 128)`. Their shared row coordinate appears in different positions. Flattening both in stored
order would mix row and channel coordinates, so the decode reform does not apply. A future shared-producer trial
must preserve that correspondence explicitly and beat the complete layer, including attention and MLP costs.
Persistent kernels remain outside this experiment's scope. The current evidence does not close the H100 prefill
gap or bring V100 decode to parity.

## Post-cut producer fusion compatibility (#1003)

All ten golden files have been updated for producer fusion after a cut. Twenty ordinary output-cut routing rows
recover the previous kernel sets. Every retained row strictly decodes, and each complete CUDA kernel set matches
its pre-feature source, argument order and launch geometry; the comparison ignores only generated function names.
All 185 measurements are retained. These are source comparisons, with no new GPU timing or correctness run.

The paired whole-layer results below remain the baseline. They do not measure the new fused producer alternative.
The next V100 s1 experiment should compare fused K/V producers with their ordinary cut alternatives, then compare
the whole layer against the 16-launch baseline and the same-input Hugging Face `torch.compile` layer. H100 s512
needs profiling before another schedule sweep: the earlier gate/up and staging trials below lost.

## Five-card whole-layer check after #988 (2026-09-30)

The target is Qwen3-0.6B layer 0 at sequence lengths 1 and 512 on the five named GPUs. Every comparison times the
Hugging Face layer with eager PyTorch, `torch.compile`, and Emmy in one process and on the same inputs. The source is
main after #973 plus the golden replay fix in this PR. Each card used its exact GPU, a fresh tune DB, deployable O3,
and `EMMY_FAST_MATH=0`. A cold compile sometimes exceeded the repository's two-minute development limit; the table
distinguishes a current paired result from an earlier paired result whose compiled Emmy kernel sources still match
the current strict replay. The warmup and iteration counts are given per cell.

Exact GPU UUIDs: RTX 5090 `GPU-bb78f2c5-11d6-02d6-f124-08b719623110`, RTX 4090
`GPU-33b961c9-1af0-1175-6135-a3f5f3f94940`, A100 `GPU-dc5ba098-1a7a-08ea-d5be-fc71f0046c7f`, H100
`GPU-c8195d4e-59c4-29d9-ed2f-0cd0b54bba73`, V100 `GPU-fb047284-9557-a127-0787-70f97e92826a`.

Captured whole-forward latency in microseconds. All s1 rows used 10 warmups and 100 iterations. A ratio above 1
means Emmy is faster. The s512 protocol is listed because cold compilation forced shorter bounded runs on two cards.

| Card | s1 Emmy | s1 `torch.compile` | s1 ratio | s512 Emmy | s512 `torch.compile` | s512 ratio | s512 protocol |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RTX 5090 | 20.474 | 36.859 | 1.80× | 129.388 | 136.882 | 1.06× | 10/100, greedy |
| RTX 4090 | 26.388 | 30.522 | 1.16× | 158.005 | 163.215 | 1.03× | 10/100 before #973; same kernel sources after |
| A100 40GB | 53.571 | 54.675 | 1.02× | 179.541 | 181.008 | 1.01× | 10/100, 11 stored cuts pinned |
| H100 | 29.026 | 31.383 | 1.08× | 87.173 | 83.485 | 0.96× | 5/20, 11 stored cuts pinned |
| V100 SXM2 | 71.753 | 60.957 | 0.85× | 502.784 | 641.653 | 1.28× | 5/20, 20 stored cuts pinned |

The 4090 s512 paired run predates #973. A strict golden replay on the rebased source passed at 156.672 us, and all
12 CUDA source hashes matched that earlier paired run. Rebased model-form attempts exceeded the two-minute cold
compile limit, so no post-rebase same-process `torch.compile` number is claimed for that cell. The current A100 s512
5/20 repeat was 179.405 versus 181.098 us. Two current H100 s512 5/20 runs measured Emmy at 87.456 and 87.173 us
against `torch.compile` at 83.335 and 83.485 us. The V100 s512 model run used the stored route pins; two unpinned
attempts exceeded the development limit before producing JSON.

Across the pre-rebase and rebased checks, strict whole-golden replay passed on all five cards at s1 and s512. Some
named stored rows now differ substantially from the greedy pick, as detailed for V100 below. Model-form s512 uses
the normal scaled correctness check. A strict repeat can differ from eager by a few one-step f16 outputs, including
six of 524,288 on RTX 5090 and H100. The golden-form strict checks passed, and no numerical tolerance was changed.
Selected CUDA source hashes matched pre-rebase records on all five cards; the rebase did not change those kernels.

Bare `run --golden FILE --bench` used to name the fastest child routing row when an inventory row was absent. That
row cannot replay the whole layer without its parent's cuts. The command now selects the fastest root routing row
first. The focused CLI test and exact-card strict replays cover this change.

The remaining losses survived targeted schedule and cut trials. On H100 s512, a wider gate/up warp group increased
the whole layer from 88.4 to 169.2 us. An A-only `cp.async` cache-policy change increased it from 88.229 to 89.525
us. A smaller gate/up schedule increased it from 88.283 to 95.851 us. The retained 12-kernel route passes strict
correctness. None of these changes closes its gap to `torch.compile`. The smaller gate/up trial is in the raw archive
below. Other H100 schedule trials and raw records are in [#992](https://github.com/cloudrift-ai/emmy/pull/992).

On V100 s1, removing projection cuts repeats too much work. A consumer-summed Q split saved one launch and tied at
about 72 us; adding the V split slowed the layer. A task-local two-output K/V cut saved two launches. Its paired
partial and final kernels took about 8.3 us in isolation, but the full 14-launch graph measured 73.2–73.5 us versus
the 16-launch baseline's 74.4–78.3 us in two short runs with clock variation. It passed a same-input strict output
comparison, yet remained about 13 us behind the model-form `torch.compile` result. No compiler or golden schedule
from that experiment was retained. The full CUDA graphs and timing records are in the raw archive.

The stored V100 s1 route also changed under #973. Its strict replay can realize an extra gate/up partial at about
208 us, making a correct 17-launch layer take about 271 us. The fresh model-form greedy result remains 16 launches
near 72 us. Strict evidence rejects the new partial because it has no measured schedule row. This is a separate
stored-route coverage problem; it does not make the model-form loss disappear. The current golden targets all pass
`emmy golden check`, which checks their Loop IR identity, not the completeness of child schedule evidence.

The task has eight favorable cells and two remaining losses. The A100 s512 lead is small and should be treated as
timing-sensitive. The V100 needs a faster way to combine the K/V projections or execute the short chain; the tested
two-launch combined schedule is insufficient. The H100 needs a reliable whole-layer gain beyond the tested gate/up
and staging schedules. Both require more compiler work before claiming universal parity.

The raw model comparisons, strict replays, timeout logs, candidate CUDA graphs, and trial findings are in
`tuning_universal_2026-09-30.tar.gz` (Git LFS). This archive omits tune DBs, cubins, and temporary prototype code.

## H100 s512 schedule check after #988 (2026-09-30)

The remaining H100 prefill gap was tested on Qwen3-0.6B layer 0 at sequence length 512. The exact H100 was an
80GB HBM3 card (sm_90, driver 580.178.04, CUDA 12.9). The compiler was main at `5193d2e0` (#988), with deployable
`-O3` and `EMMY_FAST_MATH=0`. The checked-in H100 golden supplied the route and measured schedules. Each changed
schedule was compared with that route in the same strict, whole-layer golden replay. All timed trials passed
correctness against eager. No tested schedule won reliably, so the golden is unchanged.

| Current model-form baseline, one layer | eager | torch.compile | Emmy | Emmy / torch.compile |
| --- | ---: | ---: | ---: | ---: |
| H100, s512, warmup 10, 100 iterations | 199.55 µs | 84.74 µs | 86.52 µs | 1.02× |

Both compiled model-form paths passed the scaled accuracy check. A separate strict golden-form replay passed at
88.50 µs; it has different timing semantics from the model-form comparison. The current model-form gap is 1.78 µs.
The #967 model-form row was Emmy 87.3 µs against `torch.compile` 79.0 µs. Most of the apparent gap change is the
slower `torch.compile` baseline in this current run, not a demonstrated Emmy improvement.

| Changed schedule | Touched kernel, before → after | Paired layer, before → after |
| --- | ---: | ---: |
| Gate/up producer band `+p4` | 19.31 → 97.98 µs | 88.43 → 169.17 µs |
| Attention stages `d4/smem-async/p2` | 12.67 → 13.18 µs | 88.08 → 88.71 µs |
| Attention value tile `m64n64` | 12.80 → 14.43 µs | 88.23 → 89.56 µs |
| Down stage `d4/smem-async/p2` | 10.88 → 10.89 µs | 87.97 → 92.77 µs |
| Q tile `m64n128` | 6.77 → 6.81 µs | 89.20 → 88.85 µs |
| Attention TMA value stage, `d1/smem-async/p2` score stage | 12.28 → 12.82 µs | 83.53 → 84.80 µs |

These are single paired runs, not repeat distributions. The 0.35 µs layer advantage of the Q tile came with a
slower Q kernel and is within observed run variation. The score tile `m64n128` was refused before timing because its
width disagreed with the 64-wide carrier chunk. A constrained 24-candidate attention staging search measured 20
configurations; its best isolated result was 12.8 µs, close to the recorded row near 12.7 µs. Its top distinct
candidate is the TMA trial above and lost in the layer. The temporary `+p4` compiler offer was reverted after its
loss. No compiler, route, or canonical golden change survived the pass.

The raw JSON, logs, working golden, search DB snapshot, and fuller findings are in the H100 archive's
`tuning_h100x1_2026-09-30` directory. The #967 pass identified the projections and attention as contributors to its
larger gap. This pass did not localize the current 1.78 µs gap further. Hardware-counter profiling did not finish
within the development time limit, so this pass makes no new counter claim.

## CSE re-record on exact cards

The CSE change shifted kernel identities. All ten Qwen3-0.6B layer goldens have been restamped and measured
again on their named cards. These current numbers are strict, whole-program golden replays in microseconds. The #967
numbers are model-form runs with a different warmup and iteration count, so the side-by-side values show recovery of
the earlier schedules rather than a controlled speed comparison.

| card | s1 #967 | s1 current | s512 #967 | s512 current |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 20 | 20.47 | 130 | 129.47 |
| RTX 4090 | 26.4 | 26.51 | 157.2 | 156.13 |
| H100 | 28.8 | 25.24 | 87.3 | 87.4 |
| A100 40GB | 53 | 52.6 | 179 | 179.7 |
| V100 SXM2 | 72 | 71.7 | 513 | 503.3 |

The H100 s1 route needed five additional reduction splits after CSE; without them its replay took 947 us. The recorded
16-kernel route takes 25.24 us and passes strict correctness. The V100 s512 restamp demoted 20 measured rows after its
CUDA sources changed. Its old route had no receipt for one GEMM, which took 657 us under the prior after CSE. Manually
pinning that GEMM to a measured warp and tile schedule reduced it to 112 us. The complete 21-kernel route passes strict
correctness at 500.2 us under the recording pins. A fresh-DB strict-evidence replay without hand pins takes 503.3 us
and picks the 21 measured route and kernel rows.

## Five cards — margin over torch.compile on the Hugging Face layer (2026-09-29)

### Question and scope

After #930 every cell was at or near `torch.compile`. This round asks how much margin the compiler can add without
persistent kernels, and measures it against a stricter baseline: `torch.compile` on the Hugging Face layer itself
(`emmy run Qwen/Qwen3-0.6B --layer 0 --seq-len N`, "model form"), not on the golden's replayed program ("golden form"),
which runs `torch.compile` slower (V100 s512: 637 vs 729 us). The corpus is unchanged: layer 0 at s1 and s512.

### Protocol

One lane run per card (`emmy bench` on this recipe) on the close-out compiler of #967. The lane's first step times
eager, `torch.compile` and Emmy on the Hugging Face layer in one process, with the committed golden as the only
evidence (fresh tune DB, `-O3`, `EMMY_FAST_MATH=0`, warmup 10, iters 100). Five fresh-process golden-form replays under
`--strict` against eager follow as the correctness gate. The s1 goldens were re-traced first: #871 changed how
`is_causal` traces at one token, so the old s1 programs no longer matched a fresh trace. Every golden row was recorded
from a sweep and confirmed by a whole-layer replay before recording.

### Result summary

Model form, from the lane, us. Ratio is `torch.compile` / Emmy (above 1: Emmy faster).

| card | s1 #930 | s1 Emmy | s1 t.c | s1 ratio | s512 #930 | s512 Emmy | s512 t.c | s512 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5090 | 24.5 | 20 | 33 | 1.65× | 132 | 130 | 135 | 1.04× |
| RTX 4090 | 26.5 | 26.4 | 28.9 | 1.09× | 158.7 | 157.2 | 162.6 | 1.03× |
| H100 | 36.7 | 28.8 | 35.1 | 1.22× | 95.8 | 87.3 | 79.0 | 0.90× |
| A100 40GB | 55–56 | 53 | 56 | 1.06× | 182–184 | 179 | 181 | 1.01× |
| V100 SXM2 | 72 | 72 | 62 | 0.86× | 520 | 513 | 636 | 1.24× |

The #930 column is that PR's scoreboard (golden form at s1). All ten lane cells pass their five `--strict` repeats. In
model form s512 misses `--strict` on 2-6 of 524,288 outputs by one f16 step, the accepted approximate match. The s1
`torch.compile` time is noisy on the desktop-shared RTX 5090 (25-39 us across runs) and moved 30.9 → 35.1 between two
H100 runs.

### What the pass found

- **mma.sync GEMMs (A100, 4090, 5090) were not bank-conflict bound.** ncu's conflict count on the A100 q projection is
  fill/drain port contention. The costs were the epilogue's 4-byte global stores, which throttle the load/store queue,
  and wave count. A store staged through shared memory took an A100 gate/up-sized GEMM from 38.9 to 32.1 us (torch
  31), but it moved the whole layer by under 1 us on the A100 and 4090, tied on the 5090, and as a schedule choice
  doubled the candidates every tensor-core compile prices, so it was dropped. A 96-row M tile (`f3x<N>`) fits one
  wave on the A100's 108 SMs, once a masked-M cp.async fill stopped re-reading row 511 for 64 rows (q 23 → 16.3 us). On
  the 4090 (128 SMs) and 5090 (170 SMs) the 96-row tile loses to wave quantization.
- **wgmma GEMMs (H100)** keep one MMA group in flight and store 16-byte rows; every plain GEMM now matches or beats
  `torch.mm`. The s512 cell is still 0.90×: `torch.compile`'s kernels sum to 73 us against Emmy's 86, the gap sits in
  the QKV block with cold weights, gate/up and attention, and TMA multicast across a CTA pair was slower in the layer
  (q 7.2 → 10.6 us).
- **Programmatic dependent launch** on sm_90+ is worth about 5 us at s1 on the H100 and 2 us on the 5090.
- **Isolated piece time misleads the layer.** On the A100 the isolated winners for k/v/o/down summed 7.5 us faster and
  made the layer 7 us slower; on the H100 the producer band and m64n192 did the same. Rows here were chosen by
  whole-layer replay.
- **s1 routes** keep #930's shape (8 cuts, split GEMVs with coop-t partials); the re-traced routes compile to 14
  launches. The prior's own s1 picks run 131 us on the A100 against 53 recorded.

### Systems and provenance

RTX 5090 (dev box, shared with the desktop), RTX 4090 (CloudRift `118.163.199.138:60011`), H100 80GB
(`bench-gb-h100-0924-1252-99aa`), A100 40GB (`bench-keep-a100-0921-1621-6784`), V100 SXM2 16GB (CloudRift
`185.165.50.75`). Compiler: #967 (`feature/golden-bench-margin`) after its close-out re-records, before its rebase
onto #969 (which changes recurrences only; every golden-bench target is still the fresh lowering after it).

### Durable files

`results_rtx5090x1.tar.gz`, `results_rtx4090x1.tar.gz`, `results_h100x1.tar.gz`, `results_a100x1.tar.gz`,
`results_v100x1.tar.gz` (the lane runs), and the ten `golden/qwen3-06b-s{1,512}_<card>.golden.json` files.

## H100 — the s512 layer's projections on `wgmma` (2026-09-27)

Every GEMM piece of the H100 s512 route above was recorded on `mma.sync`. The `wgmma` tier is offered for all six of
them (seven launches: gate and up share one kernel identity), including the pieces that read the RMSNorm output. One
row wins on every piece: `w4x1`, `wgmma_m64n64k16_f16_f32/f1x8/k4`, `d4/smem-async`, `gm8`.

| H100, s512 | GEMM pieces | kernel sum | end to end | eager |
| --- | ---: | ---: | ---: | ---: |
| before (`mma.sync` rows) | 135.6 | 199.8 | 233.5 | 200.9 |
| after (`wgmma` rows) | 65.8 | 129.4 | **141.5** | 201.4 |

Protocol: the same unpinned model-level replay as below, both files in one session on
`bench-gb-h100-0924-1252-99aa`. The sweep and the record ran as golden replays under `--strict`, which matched eager
(max abs error 0.00098). Swapped onto every GEMM receipt at once, the rows measured end to end: `d3` 148.9, `d4` 141.8,
`d2` 165.3, `d4` without `gm8` 146.0, `m64n128` 155.1, `w8x1` 158.7. `d3/smem-tma` is refused on the first
projection. `d5` and a `+p4` producer band do not replay on these pieces: their receipts fall back to prior picks
(366 us) without a message. The s1 file is unchanged: its pieces are one-row GEMVs.

**Where end to end goes past the kernel sum.** An `nsys --cuda-graph-trace=node` trace of each golden replay shows one
CUDA graph per replay holding every launch, about 0.1 us between kernels inside it, and a boundary between back-to-back
replays. The rest is kernels running slower in the chain than alone:

| H100 | launches | isolated sum | in-graph kernel sum | replay span | between replays | end to end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s512, `mma.sync` rows | 22 | 210.6 | 226.5 | 228.6 | 4.3 | 239.0 |
| s512, `wgmma` rows | 22 | 140.2 | 133.4 | 135.5 | 4.3 | 144.5 |
| s1 | 25 | 99.6 | 83.3 | 85.1 | 7.8 | 96.9 |

(Traced runs; nsys adds a few us to each isolated time.) On the old s512 file three single-stage `mma.sync`
projections ran 7-10 us slower each inside the graph than alone (for example 20.9 → 31.4 us), which is most of that
file's 34 us gap. Of the `d4` `wgmma` rows, two lose 1-2 us in the chain and the rest hold. At s1 the isolated times
overstate the eighteen 1-2 us kernels (one-kernel replays), and the boundary holds a memset. That memset is the
zero-init of an atomic accumulator in the program's first launch, which has no earlier kernel to carry it. It costs
about 4 us per replay of a single layer, and would be paid once per graph in a whole-model graph. None of this is
launch overhead that a runtime change would remove.

## Three cards — the whole layer as one fused kernel, with a flash-shaped route (2026-09-26)

### Question and scope

Since #897 the Qwen3-0.6B layer lowers to one fused kernel, and the recorded route cut all 31 of its seams: it
materialized the whole attention matrix, and its P.V piece ran as scalar code over 1,048,576 blocks. The layer took
1026 us on the H100 and 1467 us on the A100. This pass asks what keeps that route slow and whether a flash-shaped route
is reachable. It covers Qwen3-0.6B layer 0 at s512 on an A100, an H100 and a V100 SXM2, and at s1 on the A100 and
H100.

### Protocol

One card each: A100-SXM4-40GB (GCP `a2-highgpu-1g`), H100 80GB HBM3 (GCP `a3-highgpu-1g`, SPOT), V100-SXM2-16GB
(CloudRift). Deployable `-O3`, `EMMY_FAST_MATH=0`. Each golden is a fresh `emmy trace` of the layer. Its route was
benched under a `WORK` sweep into one tune DB per card, then recorded with `--record-greedy`, so every piece takes the
fastest measured row. The s1 files are the exception: their kernel set is the prior's, recorded from an empty tune DB,
because that is the set that splits the GEMV pieces (see below). Timings below are an UNPINNED replay of the committed
file at model level (`emmy run Qwen/Qwen3-0.6B --layer 0`, `EMMY_GOLDEN_FILE` the file), eager and `torch.compile` in
the same process, 10 warmups, 100 iterations. The same replay under `--strict` matches eager on all but 28-40 of
524,288 outputs at s512, each one f16 ulp off, which main shows too; the s1 files match it exactly.

An orphaned bench worker from another checkout held the A100 at full load for the first half of this pass. Every A100
number below was measured after it was stopped; the earlier A100 baselines (eager 785, `torch.compile` 429 at s512)
were more than twice too slow, and the A100 "before" figure was measured under the same load.

### Result summary

| card | s512 before | s512 after | eager | torch.compile |
| --- | ---: | ---: | ---: | ---: |
| A100 | 1467 (loaded GPU) | **323** | 352 | 193 |
| H100 | 1026 | **231** | 200 | 80 |
| V100 | 3471 | **1153** | 1116 | 636 |

| card | s1 after | without splits | eager | torch.compile |
| --- | ---: | ---: | ---: | ---: |
| A100 | **205** | 271 | 150 | 52 |
| H100 | **88** | 133 | 110 | 32 |

The s512 route cuts every seam except the score, the probabilities and the softmax statistics, so attention is one
twisted-carrier piece: 42.1 us on the A100 and 22.1 on the H100 on tensor cores with a staged key and value. The V100
takes the same cuts on its own evidence; its attention piece is 391 us, the Volta chunk tier without cp.async.

### What the pass found

- **P.V ran scalar because its V read the fused channel plainly.** A cut V projection stores its workspace at the
  o_proj's flat channel, so the fused-pair split, which asked for `i % d`, declined, and the piece re-walked the keys per
  channel. It now splits on a plain read too: 839 us to 95.
- **Rotate-half computed q and k three times, and the post-attention statistic computed o_proj again.** Seam clustering
  now abstracts a coordinate read through one expression, and compares forms by substitution instead of renaming, so a
  loop inside a cone that re-binds a captured name keeps its own variable.
- **A delegated zero-init ran on CTA 0 alone** and cost the V projection piece 130 us for a 4 MB accumulator; every
  thread of the grid now writes a stride.
- **The RoPE pieces were uncoalesced** (39 us on the H100): a re-formed piece with no contraction now takes its store's
  write order as its grid order.
- **Fusion dropped the f16 rounding of the V projection and attention output** where a reshape composed into the
  consumer. With it spelled, and a reducing seam stored at the dtype its readers convert to, V and the attention output
  are f16 workspaces, which the flash tier stages.
- **The flash tier read its hoisted query at the wrong row stride.** A query stored `[seq, head, dim]` strides its rows
  by every head's dim; the tier took the trailing extent and disagreed with eager on 487k of 524k outputs.
- **Cross-CTA split GEMV pieces returned wrong answers at s1, on main too** (#918). The old s1 files replayed at 55 us
  on the H100 but disagreed with eager on 1015 of 1024 outputs: a re-formed split partial lost its unit row and tiled
  the partition coordinate as M. With that fixed, five GEMV pieces split in each s1 file.
- **The evidence pick never takes a split it has measured.** Recorded from the sweep's tune DB, the s1 route chose no
  split (A100 271 us, H100 133) although the same sweep measured split pieces faster (a 52 us GEMV as a 21 us partial
  plus a 1.4 us finalize). The tune DB holds perf rows for the pieces but no routing row, so nothing prices the split
  arm. The committed s1 files take the prior's set instead; forcing `g8k` or `g16k` on every piece is slower.
- **s1 still loses to `torch.compile`** mostly on one piece: the q and k projections and their rotate-half copies lower
  as ONE six-channel GEMV that reads its weights three times (81 us on the A100 under the prior's schedule).

### Systems and provenance

- A100-SXM4-40GB `bench-keep-a100-0921-1621-6784`, H100 80GB HBM3 `bench-gb-h100-0924-1252-99aa` (GCP, driver 580.173,
  nvcc 12.9); V100-SXM2-16GB at `185.165.50.75` (CloudRift, driver 580.178, nvcc 12.9).
- Source: PR #914 on main `2c016a3a` with the #918 split fix.

### Durable files

- Goldens: `golden/qwen3-06b-s512_{a100,h100,v100}.golden.json`, `golden/qwen3-06b-s1_{a100,h100}.golden.json`.
- No archive: the recipe was not re-run; the sections below remain the lane evidence for everything else.

## Three cards — the score target computed its operands (2026-09-24)

### Question and scope

`q/k norm + RoPE + score statistics` was the corpus's slowest target on every card — 94 us on the H100, 161 on the
A100, 361 on the V100, against 29, 64 and 285 for Inductor. The 2026-09-11 pass called its route space exhausted and
its remaining gap structural: two pieces at 12% occupancy with 176 and 204 registers. This pass asks what those
pieces are actually doing, and covers only this target: Qwen3-0.6B layer 0 at sequence length 512, on a V100, an
A100 and an H100. It retunes no other target.

The answer is that the dominant piece was computing its operands instead of reading them. It is a flash kernel whose
q and k are the RMSNorm + RoPE cones of an f32 input, and a computed operand takes the synchronous compute fill: the
kernel re-evaluated both cones per A- and B-slab cell, three strided f32 loads at a time, and staged nothing.
Cutting the two cones into their own kernels leaves it two materialized f16 slabs, which is what lets it stage at
all.

### Protocol

One card each: a V100-SXM2-16GB (CloudRift, driver 580.178.04, nvcc 12.9) — NOT the SXM3 32GB part the recipe names;
an A100-SXM4-40GB and an H100 80GB HBM3 (GCP, driver 580.173.02, nvcc 12.9). Every number is deployable `-O3`,
`EMMY_FAST_MATH=0`, 10 warmups, 100 iterations, eager, Inductor and Emmy in one process. Tuning was manual: about 45
pinned `emmy run --ab` rows in four rounds per card, each round one process so the eager reference slice is built
once. Each winner was re-recorded with `--record-greedy --strict` into a copy of the committed file whose loop-8
rows had been stripped, merged back, and the committed file then replayed UNPINNED from a fresh tune DB — the
deploy contract, and the numbers below.

### Result summary

Today's walk of the committed file, all nine targets. `before` is the same file on the same card before this pass;
only the last row changed, so the other eight are one measurement printed once.

| target | eager | Inductor | before | after |
| --- | ---: | ---: | ---: | ---: |
| **H100** | | | | |
| input RMSNorm | 53 | 5 |  | 2.6 |
| value projection | 5 | 5 |  | 6.2 |
| query projection | 11 | 7 |  | 7.6 |
| key projection | 9 | 6 |  | 6.0 |
| softmax x V | 18 | 18 |  | 64.6 |
| SDPA + o_proj + residual | 25 | 21 |  | 46.8 |
| post-attn norm + gate/up | 79 | 22 |  | 18.3 |
| down_proj + residual | 12 | 11 |  | 15.8 |
| q/k norm + RoPE + score statistics | 168 | 29 | 94.2 | **29.5** |
| H100 total | | 124 | 262.1 | **197.4** |
| **A100** | | | | |
| input RMSNorm | 65 | 6 |  | 3.9 |
| value projection | 11 | 13 |  | 13.9 |
| query projection | 22 | 21 |  | 19.2 |
| key projection | 16 | 13 |  | 13.7 |
| softmax x V | 49 | 50 |  | 163.2 |
| SDPA + o_proj + residual | 65 | 63 |  | 67.0 |
| post-attn norm + gate/up | 125 | 52 |  | 49.5 |
| down_proj + residual | 27 | 32 |  | 36.3 |
| q/k norm + RoPE + score statistics | 236 | 64 | 161.0 | **49.2** |
| A100 total | | 314 | 527.7 | **415.9** |
| **V100** | | | | |
| input RMSNorm | 78 | 7 |  | 4.3 |
| value projection | 42 | 41 |  | 33.0 |
| query projection | 46 | 42 |  | 47.3 |
| key projection | 48 | 46 |  | 32.0 |
| softmax x V | 533 | 300 |  | 349.5 |
| SDPA + o_proj + residual | 560 | 321 |  | 281.3 |
| post-attn norm + gate/up | 217 | 130 |  | 105.5 |
| down_proj + residual | 70 | 67 |  | 110.0 |
| q/k norm + RoPE + score statistics | 822 | 285 | 360.7 | **229.3** |
| V100 total | | 1239 | 1323.6 | **1192.2** |

The retuned target's three kernels, after:

| card | q cone | k cone | flash kernel |
| --- | ---: | ---: | --- |
| H100 | 6.6 | 4.1 | 17.7 us `w4x1`, `mma_m16n8k16/f1x8/k4` on both contractions, `d2/smem-async`, `gm8` |
| A100 | 10.4 | 6.3 | 32.8 us `w4x1`, `mma_m16n8k16/f1x8/k4`, `d3/smem-async` |
| V100 | 14.3 | 8.1 | 205.0 us `w4x1`, `mma_m8n8k4/f2x2/k8`, `d2/smem` |

### What the pass found

- **The cut the target wanted was not reachable, because a cut piece re-folded its row statistic per cell.** A piece
  was minted with one free axis per workspace dimension, which binds the sweep the statistic is invariant in. The
  materialized q cone therefore launched 1048576 cooperative blocks — one per output element, each re-reducing the
  whole 128-wide row and writing one value: 317 us for a 1 MB pass on the H100. The piece now asks the same rank
  rule the fused kernel and the cut's own peel test already ask, and runs 6.6 us. That fix is what makes every
  number above reachable.
- **The win is the transport, not the tile.** Fused, the flash kernel's operands are computed, so only the
  synchronous compute fill resolves — every committed row on all three cards is `d1/smem`. With both cones
  materialized it takes `d2/smem-async` on the H100 and V100 and `d3` on the A100. The tile alone buys little: the
  first staged pick on the H100 still ran 69 us, and the same `f1x8/k4` tile measures 17.7 once the transport
  follows. On the H100 the two cones cost 10.7 us and the set totals 28.4 against 94.2.
- **The corpus's remaining gap on the datacenter cards is `softmax x V`, and it is the same defect one level down.**
  That target's value projection is computed inside the attention sweep. Cutting it materializes the projection into
  an f32 workspace shaped (head, dim, key), which the consumer reads back one fragment at a time through
  `mma_load_b_gmem_trans<float, __half>` at a 2 KB stride with no staging: 150 us on the A100 against 33 for the
  same kernel over `transpose_2`. Storing that workspace at the atom dtype instead was tried and measured nothing
  (151.8 us), so the layout is what binds, not the width. A workspace whose axis order follows the consuming
  contraction's own slab would close it; this pass does not.
- **The V100's flash kernel still spills.** 205 us at 255 registers, 8 bytes of local, 12% occupancy; every tile,
  warp split and staging depth the sweep reached lands between 197 and 636. That is the V100's remaining gap, and it
  is register pressure, not placement.
- **The V100 corpus now beats Inductor overall** (1192 against 1239) while the H100 and A100 still trail (197
  against 124, 416 against 314). Both remaining gaps are `softmax x V` and `SDPA + o_proj + residual`.

### Systems and provenance

- V100-SXM2-16GB at `185.165.50.75` (CloudRift), A100-SXM4-40GB `bench-keep-a100-0921-1621-6784` (GCP
  `a2-highgpu-1g`, us-central1-f), H100 80GB HBM3 `bench-gb-h100-0924-1252-99aa` (GCP `a3-highgpu-1g`, SPOT,
  us-east4-a) — a replacement for `bench-attn-h100-0923-1047-4b3d`, preempted mid-pass; us-central1 had no H100
  capacity to restart it in.
- Source: this branch, on top of main `8ca20b12`. The tuning rows are host-local under `~/gb-work/` on each box.

### Durable files

- Goldens: loop 8 re-recorded in `golden/qwen3-06b-s512_h100.golden.yaml`, `golden/qwen3-06b-s512_a100.golden.yaml`
  and `golden/qwen3-06b-s512_v100.golden.yaml`. Every other target is untouched.
- No archive: this pass tuned one target by hand and did not run the recipe, so it replaces no platform archive and
  the sections below remain the current lane evidence for every other target.
- The V100 walk still fails its strict gate on `down_proj + residual` — 7 of 524288 elements at flat index 413453,
  the same row and the same values #889 recorded. Nothing in this pass touches it.
- The compiler half costs 29 recorded rows elsewhere: 19 in `DeepSeek-V4-Flash-0731/v100_sm70.yaml`, 6 in
  `gemma-4-12B-it/rtx5090_sm120.yaml`, 4 in `Qwen3.8-27B-FP8/v100_sm70.yaml`. All are cut pieces whose schedule was
  composed against the per-element grid the rank rule now declines, so the piece is a different kernel and carries
  no rows. They need their cards to re-record; the strict decode names each one.

## Three cards — the gated MLP's staging lockout (2026-09-23)

### Question and scope

The `post-norm + gate/up + SiLU` target loses to Inductor on every card in the sections below — 0.36x and 0.29x on
the A100, 0.47x and 0.56x on the H100 — while the single-channel projections beside it beat Inductor on the same
runs. This pass asks why that one target is different, and covers only it: Qwen3-0.6B layer 0, sequence lengths 1
and 512, on a V100, an A100 and an H100. It retunes no other target and supports no claim about them.

The answer is structural, not a tuning shortfall. The gate and up projections share one A operand, so the term folds
TWO channels. The staging catalog refused the prefetching transports for any multi-channel fold and confined it to
the synchronous compute fill, which on Hopper also put the wgmma tier out of reach, because wgmma reads its operands
through shared-memory descriptors a TMA box fills. Every transport already deposits one slab per operand and the
drain already reads them in order, so the confinement bought nothing; lifting it is what the compiler half of this
pass does. A second, narrower refusal barred the fill's depth-2 ring on Volta — written for the ring under a compute
fill, it also caught the case where nothing is computed and the ring is an ordinary blocking-copy double-buffer.

### Protocol

One card each: a V100-SXM2-16GB (CloudRift, driver 580.178.04, nvcc 12.9) — NOT the SXM3 32GB part the recipe names,
so its numbers are not comparable to a recipe row; an A100-SXM4-40GB and an H100 80GB HBM3 (GCP, driver 580.173.02,
nvcc 12.9). Every number is deployable `-O3`, 10 warmups, 100 iterations, eager, Inductor and Emmy in one process.
Tuning was manual, as in the sections below: about 45 pinned `emmy run --ab` rows in three rounds per length, each
round one process so the eager reference slice is built once, seeding a task-owned tune DB. Each winner was then
re-recorded with `--record-greedy` into a stripped copy of the inventory, and the committed file replayed UNPINNED
from a fresh tune DB under `--strict --strict-evidence` — the deploy contract, and the numbers reported here.

### Result summary

`before` is the committed golden replayed on this card today; the V100 has no golden in this recipe, so its before is
the cold greedy at its best route. Inductor is the torch.compile lane of the same process.

| target | eager | Inductor | before | after | Inductor / after |
| --- | ---: | ---: | ---: | ---: | ---: |
| V100 prefill (512) | 221 | 130 | 328.7 | **106.1** | **1.23** |
| V100 decode (1) | 77 | 19 | 34.2 | 34.2 | 0.56 |
| A100 prefill (512) | 125 | 53 | 950.7 | **50.1** | **1.06** |
| A100 decode (1) | 63 | 13 | 31.9 | 19.5 | 0.67 |
| H100 prefill (512) | 79 | 23 | 44.1 | **18.5** | **1.24** |
| H100 decode (1) | 50 | 9 | 15.7 | 10.2 | 0.88 |

Both kernels of the cut, after:

| card | statistic piece | GEMM piece |
| --- | --- | --- |
| V100 prefill | 5.0 µs `t128/coop` | 101.1 µs `w4x2`, `mma_m8n8k4/f2x2/k8`, `d2/smem`, `gm8` |
| A100 prefill | 4.3 µs `t128/coop` | 45.7 µs `w2x2`, `mma_m16n8k16/f2x4/k4`, `d2/smem-async` |
| H100 prefill | 3.8 µs `t128/coop` | 14.8 µs `w4x1`, `wgmma_m64n64k16/f1x8/k4`, `d2/smem-tma`, `gm8` |
| A100 decode | 3.5 µs `t128/coop` | 16.0 µs `w1x1`, `mma_m16n8k16/f1x8/k4`, `d2/smem-async`, `gm8` |
| H100 decode | 3.0 µs `t128/coop` | 7.2 µs `w1x1`, `mma_m16n8k16/f1x2/k4`, `d2/smem-tma` |

### What the pass found

- **The committed A100 rows recorded an unscheduled kernel.** Both A100 files gave the cut's norm producer an EMPTY
  knob row — what the recorder writes when the term falls unmapped — and the replay honours it: 905 µs for a
  512×1024 norm the same card runs standalone in 3.4. The producer forks normally on main, so this is a stale
  recording and not a live defect; re-recording is the whole of the A100 prefill gap.
- **The H100's deeper cut was a workaround that outlived its defect.** `PLACE@map.2/inner.1/map.3/map=cut` keeps only
  the statistic in the producer and folds the scale into the GEMM's A, which the 09-11 section took because the
  shallower seam left the whole norm in an unforked producer. That producer forks now (`t128/coop`, 3.8 µs prefill),
  and the shallower seam is better on all three cards: it leaves the GEMM a MATERIALIZED A, which is what lets the
  copy transports carry it at all.
- **TMA is worth 2.3x on the H100 GEMM, and it was unreachable.** Under the compute fill the gate/up GEMM ran 33.9 µs
  on mma.sync. One A box and two weight boxes over a two-deep ring, drained by two wgmma chains off the ONE shared A
  descriptor, run 14.8 µs — and the `gm8` raster is a third of that on its own (20.1 µs without it). The single
  linear of the same 512×6144×1024 shape measures 17.7 µs pinned to the same row, and cuBLAS 12; the fused form is
  now faster than either because it never writes the gate and up products to memory.
- **The cp.async ring is worth little on the A100 and a lot on Volta.** The A100 GEMM moves 47.2 → 45.7 µs, because
  its compute fill already put the peers on cp.async; the whole A100 gap was the producer row. The V100 has no
  cp.async at all, so its fill had no peers to fly and ran single-buffered: 178.0 µs. With nothing computed, the
  ring is an ordinary blocking-copy double-buffer, and the same row over it runs 100.8.
- **Decode closes by less, and the reason is not staging.** The gate/up decode is a GEMV reading 12.6 MB of weights:
  8.1 µs of A100 bandwidth, 3.8 of H100. Emmy runs 16.0 and 7.2. What the shape wants is a cross-CTA split filling
  the card, and the split declines on this term — *the head fold is nested inside the projection's sweep loop; the
  split cannot strip it* — so the kernel stays on 48 and 192 CTAs. That refusal is the decode gap and this pass does
  not touch it.
- **The cold prior ranks the new options badly.** Widening the catalog cost the H100's unpinned cold pick, which went
  from 37.7 µs to 85 on the same target: the prior has never seen a multi-channel row carrying a copy transport. The
  deploy path is the recorded golden, which is unaffected, but a lane that searches instead of replaying — the
  recipe's V100, RTX and H200/B200 rows — may need a retune before it sees the win.

### Systems and provenance

- V100-SXM2-16GB at `185.165.50.75` (CloudRift), driver 580.178.04, nvcc 12.9, torch cu126.
- A100-SXM4-40GB `bench-keep-a100-0921-1621-6784` and H100 80GB HBM3 `bench-keep-h100-0921-1621-1fa1` (GCP
  `a2-highgpu-1g` / `a3-highgpu-1g`), driver 580.173.02, nvcc 12.9.
- Source: this branch, on top of main `1d5a7a73`. The tuning rows and their `--json` records are host-local under
  `~/gmlp-out/` on each box; no recipe lane was run, so the `results_*.tar.gz` archives are unchanged.

### Durable files

- Goldens: loop 6 re-recorded in `golden/qwen3-06b-s1_a100.golden.yaml`, `golden/qwen3-06b-s512_a100.golden.yaml`,
  `golden/qwen3-06b-s1_h100.golden.yaml`, `golden/qwen3-06b-s512_h100.golden.yaml`. Every other target is untouched.
- No archive: this pass tuned one target by hand and did not re-run the recipe, so it replaces no platform archive
  and the sections below remain the current lane evidence for every other target.

## Platform a10040x1 — hand-found common corpus on the 40GB part (2026-09-11)

### Question and scope

The committed A100 goldens no longer replayed: eight of the nine decode rows and most prefill rows spelled schedules
the current compiler does not enumerate (`WORK=t512` with the transposed cooperative band, whose catalog stops at
256 threads; the retired `PLACE@b` seam spelling; `REDUCE=coop/r2` on the fused norm), so the recipe's A100 lane
benched nothing and reported every target as an unreproducible pin. This pass re-finds the schedules by hand on an
A100-SXM4-40GB — the same die as the archived 80GB rows at 1555 GB/s instead of 2039 — and asks the same question
as the H100 section: which of the nine Qwen3-0.6B layer-0 targets beat Inductor, and where does each loss come from.
The goldens keep the embedded programs of the 80GB files with every knob and measurement stripped, so the compared
programs are identical to the H100 and 80GB rows. Search was manual: about 110 pinned `emmy run --strict` runs in
five rounds, seeded from the H100 rows and the sm_80 tiers, then `--record-greedy` under each winning pin. Committed
as `golden/qwen3-06b-s1_a100.golden.yaml` and `golden/qwen3-06b-s512_a100.golden.yaml`, now labeled
`NVIDIA A100 40GB`; the recipe gains a 40GB replay row beside the 80GB one, selected with `--filter deploy.gpu=`.

### Protocol

One `a2-highgpu-1g` VM (A100-SXM4-40GB, GPU-be299b90, driver 580.173.02, nvcc 12.9, PyTorch 2.11.0+cu130, triton
3.6.0, Ubuntu 24.04.4). Every number is deployable `-O3`, 10 warmups, 100 iterations, eager and Emmy in one process;
Inductor is the separate `torch.compile` lane the recipe runs once per task. The lane is one `emmy bench
experiments/golden-bench-2026/kernels --ssh … --filter "deploy.gpu=NVIDIA A100 40GB"` invocation at source
`79eef22c`, run `20260911T135801Z` (13:58-14:24 UTC), five strict repeats per sequence length, with a task-owned tune
DB and cubin cache.

### Result summary

Medians of the five strict repeats; Inductor from the torch-compile lane of the same task.

Decode: nine of nine targets correct on every repeat, task status succeeded. Inductor returned no timing for the
q/k-norm form (as on the H100).

| decode (sequence length 1) | eager | Inductor | Emmy | Inductor / Emmy | launches |
| --- | ---: | ---: | ---: | ---: | ---: |
| input RMSNorm | 131.1 | 3.13 | **2.60** | **1.20** | 1 |
| q_proj + q_norm statistic | 5.47 | 7.41 | **5.84** | **1.27** | 2 |
| k_proj + cast | 37.6 | 7.01 | **6.83** | **1.03** | 2 |
| v_proj | 36.2 | 6.82 | **5.84** | **1.17** | 2 |
| v_proj + 1-key SDPA | 13.1 | 12.5 | 14.5 | 0.87 | 1 |
| 1-key SDPA + o_proj + residual | 15.3 | 13.8 | **11.8** | **1.16** | 3 |
| post-norm + gate/up + SiLU | 149.6 | 10.9 | 30.4 | 0.36 | 2 |
| down_proj + residual | 9.88 | 7.73 | 96.8 | 0.08 | 2 |
| q/k norm + RoPE + score statistics | 359.3 | no timing | 3.11 | — | 1 |

Prefill: eight of nine targets measured and correct on every repeat; the q/k-norm form fails to lower, so the task
status is failed by design. Two attention forms carry no recorded row and are reported at the lane's own pick.

| prefill (sequence length 512) | eager | Inductor | Emmy | Inductor / Emmy | launches |
| --- | ---: | ---: | ---: | ---: | ---: |
| input RMSNorm | 208.6 | 10.2 | **3.43** | **2.98** | 1 |
| q_proj + statistic | 11.6 | 11.6 | 14.5 | 0.80 | 1 |
| k_proj + cast | 111.2 | 18.6 | 19.9 | 0.94 | 1 |
| v_proj | 63.5 | 14.3 | 14.4 | 0.99 | 1 |
| softmax × V (no recorded row) | 49.9 | no timing | 1846 | — | 3 |
| SDPA + o_proj + residual (no recorded row) | 94.2 | no timing | 85601 | — | 2 |
| post-norm + gate/up + SiLU | 267.4 | 56.2 | 194.3 | 0.29 | 2 |
| down_proj + residual | 27.3 | 31.7 | 39.0 | 0.81 | 1 |
| q/k norm + RoPE + score statistics | — | no timing | fails to lower | — | — |

Run-to-run spread over the five repeats stayed within 3% on every target except two whose first repeat ran 28%
slower than the other four (v_proj + 1-key SDPA at 18.6 versus 14.4 µs, prefill q_proj at 18.6 versus 14.5); the
medians above are the four agreeing repeats' value.

### What the sweep found

- **The old rows were not stale measurements but retired spellings.** `WORK=t512` with `REDUCE=coop-t` decoded on
  no card: the transposed cooperative band's catalog offers 32 to 256 threads and never offered 512, so those rows
  can only have been recorded by a codec that read the two knobs differently. `PLACE@b` names a seam by axis, which
  the route grammar retired. Neither is a regression to fix; the rows are re-found below.
- **Decode GEMVs want a cross-CTA split under the transposed band.** The bare 256-thread band lands 32 CTAs per
  1024 outputs (k_proj 7.8 µs); `g8k` with the same band per piece fills the card and takes k_proj to 6.9, v_proj to
  5.9 and q_proj to 5.9 µs, each within 3% of Inductor's single kernel — the finalize launch is the whole gap. The
  fragment-tile splits the cold greedy prefers (`w1x2`, `f2x8/k2`) are 1.3-2x slower here, and the plain cooperative
  band (`coop`) is 2-3x slower than its transposed twin at every width.
- **The decode down-projection's fast rows fail the strict gate, as on V100 and H100.** `g8k/coop-t` measures 8.4 µs
  against Inductor's 7.8, and every cooperative row (split or not, transposed or not) is one element in 1024 off eager
  by four fp16 ulps: the rounding boundary before the residual add. None may be recorded, so the file keeps the
  cold pick, a `g2k` split on the fragment tile at 97 µs.
- **The two fused decode attention forms.** SDPA + o_proj + residual beats Inductor through the materializing cut
  (`PLACE@map.1/inner.2/map=cut`, 11.9 µs, three launches). v_proj + 1-key SDPA has no cut route that helps here: the
  fused kernel on the `f1` tile with one-stage staging (14.5 µs in the sweep, 16.8 µs median in the lane with a 25%
  spread across repeats) is the best correct row, 1.3x behind Inductor.
- **The decode MLP halves with a cut and a transposed band on the gate/up child.** Fused, the best row is 61.7 µs;
  `PLACE@map.2/inner.1/map=cut` with `WORK=t256,REDUCE=coop-t` reaches 30.4 µs (norm child 14.8 µs as a direct
  kernel, gate/up 14.2). The norm child is the remaining loss against Inductor's 14.5 µs: the bare pin reaches both
  children, and a child-scoped pin is the same gap the H100 section records.
- **Prefill GEMMs stop at the 64x64 tile with a three-deep cp.async ring**, exactly the H100 finding: `w2x2`,
  `f2x4/k4`, `d3/smem-async` is the best row for q_proj, k_proj and v_proj, within 7% of Inductor on k_proj and
  v_proj and 1.25x behind on q_proj; the wider `f4x4/k8` fragment and the `w4x1` tile are 5-15% slower. The
  down-projection (K=3072) prefers `w4x1`, `f1x4/k8`, two-stage cp.async at 39 µs against Inductor's 31.7; the
  three-deep ring is 60% slower there, and a `g2k` split fails the strict gate by the same residual rounding.
- **The prefill MLP's tensor-core rows cannot stage.** Every `d3/smem-async` pin on the fused kernel and on the cut's
  GEMM child raises "STAGE pin does not resolve for this contraction"; `d2/smem` is the only staging the cut child
  accepts, and it lands the pair at 194 µs against Inductor's 53.6 (norm child 152 µs as a direct kernel, GEMM 41).
- **Three prefill attention forms have no realizable row on sm_80, one of them by a wrong answer.** Softmax x V: the
  fused greedy hangs past the 2 s kernel watchdog, the corpus cut route (`PLACE@map.1/twist.2/inner=cut`) and the
  `g4k` split of its twisted band both return wrong answers on a million of a million elements, and the row-wise
  `w8x1` tile exceeds the 60 s bench budget; the lane's evidence-driven pick is a correct three-launch route at 1.85
  ms. SDPA + o_proj + residual: every route exceeds the 60 s budget under a pin, and the lane's pick measures 85.6 ms.
  The q/k-norm + RoPE + score-statistics form fails to lower in the cold greedy (`no extent for coordinates
  ['in6']`, the H100 failure), its tensor-core pins fail nvcc with a duplicate accumulator declaration, and its
  scalar pins exceed the budget. All three stay inventory rows, and the prefill task's status is failed by design.
- **A base row that lists its kernel set is pinned by its routing arm alone.** `--record-greedy` writes a
  `kernel_set` listing on the base realization, which makes the replay lane pin that row with only the split
  (`REDUCE=g8k`) and leave the pieces' schedules to the planner; on this card that pick was a cut + split on a
  fragment tile whose answer was wrong on 2045 of 2048 elements. The listings are dropped from both files, which is
  the shape the H100 goldens already have: the lane benches the evidence-driven pick, which reaches the recorded rows.

### Systems and provenance

- Host `bench-codex-a100-0908-0933-d43f` (GCP `a2-highgpu-1g`, FLEX_START, us-central1-b), one A100-SXM4-40GB
  (`GPU-be299b90-0ff5-e1b4-db53-28465b6f874b`), driver 580.173.02, nvcc 12.9, Ubuntu 24.04.4, PyTorch 2.11.0+cu130,
  triton 3.6.0.
- Source `79eef22c`, clean staged tree; run directory `2026-09-11_13-58-01`; both task records and both
  `artifacts.tar.gz` are in the archive.

### Durable files

- Goldens: `golden/qwen3-06b-s1_a100.golden.yaml`, `golden/qwen3-06b-s512_a100.golden.yaml`.
- Raw-results archive: `results_a10040x1.tar.gz` (root member `2026-09-11_13-58-01/` with the two `*.experiment.yaml`
  records and the two `*_artifacts.tar.gz`). The 80GB archive `results_a100x1.tar.gz` is unchanged and not comparable.

## Platform h100x1 — retune on main after the wgmma fixes (2026-09-11)

### Question and scope

The 2026-09-10 rows below were recorded at `2d4f9510`, before the atom-major wgmma B (#782) and the H100 hardware
golden (#787) landed. Replayed on main `f01a23e7` they no longer all deploy: the decode v_proj + 1-key SDPA target
picks a 43 µs tensor-core child where its recorded route measured 10.9, because the child receipts name kernel
identities the current lowering no longer produces, and the decode down_proj still runs the 223 µs cold pick. This
pass re-tunes every losing target by hand (`emmy tune` was not used: eight rounds of pinned `emmy run --strict`,
about 190 measured rows, fresh tune DB per row) and re-records all eighteen targets on main from a stripped
inventory, so every row in the two files carries a main identity and a strict verdict. Same goldens, same lane form,
same host as the section below.

### Protocol

The same `a3-highgpu-1g` SPOT VM as the section below (H100 80GB HBM3, driver 580.173.02, nvcc 12.9, PyTorch
2.14.0+cu130). Tuning: every candidate is one `emmy run --golden <working copy> --realization <target> --bench
--strict --bench-backends eager,emmy` process under an `EMMY_KNOBS` pin, against its own empty tune DB (a
`bench_fail` row left by an earlier candidate silently disqualifies later pins of the same kernel), at deployable
`-O3`, 10 warmups and 100 iterations. Recording: the two committed files were stripped to their inventory rows and
every target re-recorded with `--record-greedy` under its winning pin, one run per kernel set and two for the MLP and
softmax × V cuts; receipts that a later run of the same target superseded were pruned so each kernel identity keeps
its fastest strict-correct receipt. Both files then replayed unpinned from a fresh DB (`--strict`, eager, Inductor
and Emmy in one process) and deployed the intended kernel set on every target. The lane is one
`emmy bench experiments/golden-bench-2026/kernels --ssh … --filter deploy.gpu=…` invocation from a clean checkout at
`f8debb51`, run `20260911T091850Z`, five strict repeats per sequence length.

### Result summary

Medians of the five strict repeats; Inductor from the torch-compile lane of the same task; the 09-10 ratio is the
section below. Decode: nine of nine targets correct on every repeat (the task status is failed because Inductor cannot
compile the score-statistics target and the strict walk counts that). Prefill: seven of nine measured and correct on
every repeat; softmax × V deploys a correct route but its inventory row's own pinned replay picks a wrong split, and
the two attention forms below it have no runnable row, so the task status is failed by design.

| decode (sequence length 1) | eager | Inductor | Emmy | Inductor / Emmy | 09-10 | launches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| input RMSNorm | 114.1 | 2.77 | **2.25** | **1.23** | 1.25 | 1 |
| q_proj + q_norm statistic | 4.22 | 2.73 | 4.21 | 0.65 | 0.59 | 2 |
| k_proj + cast | 29.6 | 4.62 | 5.38 | 0.86 | 0.84 | 2 |
| v_proj | 30.0 | 4.16 | 4.69 | 0.89 | 0.83 | 2 |
| v_proj + 1-key SDPA (softmax-reduce cut) | 9.18 | 8.29 | 9.88 | 0.84 | 0.75 | 2 |
| 1-key SDPA + o_proj + residual | 12.4 | 9.12 | **8.40** | **1.09** | 1.04 | 4 |
| post-norm + gate/up + SiLU (statistic cut) | 124.5 | 7.48 | 15.97 | 0.47 | 0.30 | 2 |
| down_proj + residual (tensor-core split) | 7.82 | 3.66 | 5.72 | 0.64 | 0.02 | 2 |
| q/k norm + RoPE + score statistics | 307.4 | Inductor compile failed | 2.58 | — | — | 1 |

| prefill (sequence length 512) | eager | Inductor | Emmy | Inductor / Emmy | 09-10 | launches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| input RMSNorm | 175.0 | 6.02 | **2.87** | **2.10** | 2.20 | 1 |
| q_proj + statistic (wgmma n64) | 5.08 | 5.00 | 5.66 | 0.88 | 0.70 | 1 |
| k_proj + cast (wgmma n64) | 66.6 | 9.15 | **7.70** | **1.19** | 1.03 | 1 |
| v_proj (wgmma n64) | 42.3 | 7.50 | **5.76** | **1.30** | 0.99 | 1 |
| softmax × V (V-projection cut) | 16.9 | 17.1 | 110.4 | 0.15 | 0.12 (wrong) | 2 |
| post-norm + gate/up + SiLU (statistic cut) | 194.8 | 23.4 | 41.5 | 0.56 | 0.17 | 2 |
| down_proj + residual (wgmma n64, producer band) | 11.4 | 11.5 | 13.25 | 0.87 | 0.63 | 1 |
| SDPA + o_proj + residual | — | — | every form hangs | — | — | — |
| q/k norm + RoPE + score statistics | — | — | fails to lower | — | — | — |

Run-to-run spread over the five repeats stayed within 3% on every measured target.

### What the retune found

- **A one-warpgroup wgmma tile over a two-deep TMA ring is the prefill linear row.** `w4x1`, `wgmma_m64n64k16`,
  `f1x8/k4`, `d2/smem-tma`: 256 CTAs for q_proj, 128 for k_proj and v_proj, two waves or one. The 8-warp form, the
  128- and 256-wide instruction tiles, deeper rings and a producer band all lose on these short grids (q_proj n128
  9.3 µs, n256 13.8, `w8x1` 8.2-11.1; v_proj `d3` 7.0 vs `d2` 5.8). k_proj and v_proj now beat Inductor; q_proj, which
  carries the q_norm statistic in its epilogue, closes from 0.70 to about 0.9.
- **The prefill down_proj wants the producer band.** K=3072 on 128 CTAs: `w4x1+p2` over `d4/smem-tma` runs 13.3 µs
  against 15.9 for the uniform 4-warp form and 18.7 for the recorded mma.sync row. Every split (`g2k`, `g4k`, with and
  without the residual cut) fails strict by 8-12 elements at 4 fp16 ulps: the fused-residual rounding defect below.
- **Both MLP forms cut at the norm's map, not at the norm.** The recorded `PLACE@map.2/inner.1/map=cut` seam of the
  09-10 files leaves the whole RMSNorm in the producer piece, which lowers as a direct kernel with no schedule fork
  (12 µs decode, 92 µs prefill for a 1024-wide norm). One level deeper, `PLACE@map.2/inner.1/map.3/map=cut` keeps only
  the statistic in the producer, which schedules as an ordinary cooperative reduce (1.6 µs decode, 2.3 prefill), and
  folds the scale into the GEMM's A operand. That A is computed, so only the synchronous `smem` fill resolves for the
  GEMM piece (`smem-async` and TMA raise `STAGE pin does not resolve`), which keeps wgmma at 28.7 µs for the
  512×6144×1024 gate/up GEMM (n128 33.2). The decode GEMV piece cannot split (`the head fold is nested inside the
  projection's sweep loop`), so it runs a two-warp mma.sync strip at 13.4 µs; the cooperative form sits on 24 CTAs at
  19 µs.
- **A cut whose pieces want different rows is recorded in two pinned runs.** A bare `WORK` / `REDUCE` pin reaches
  every piece of a cut, and a value one piece cannot spell drops that piece to its unscheduled form. Recording the
  statistic piece under `WORK=t128,REDUCE=coop` and the GEMM piece under its tensor-core pin, into the same file,
  leaves one routing row and two receipts per piece; the unpinned strict-evidence replay then takes the fastest
  measured receipt per kernel (verified on the decode MLP: 1.6 + 19.0 µs from a file whose other receipts read
  46.5 and 9.1).
- **The decode down_proj's cooperative rows fail strict; its tensor-core rows pass.** Every `coop-t` row, split or
  not, cut or not, is off at the same element (index 225, four fp16 ulps) — so the difference is in the GEMV's own
  accumulation, not in where the residual rounds. An mma.sync strip accumulates like the reference and passes:
  `w2x1`, `f1x8/k4`, `d2/smem-tma`, `g16k` runs 3.6 + 1.1 µs, from the 223 µs cold pick, at 0.64× Inductor.
- **The decode v_proj + 1-key SDPA recovers through the softmax reduce.** `PLACE@map.1/map.1/reduce=cut`
  materializes the one-key softmax statistic (3.2 µs) and leaves the projection and the weighted sum to one
  tensor-core kernel (6.5 µs): 10.0 µs, against the recorded route's 10.9 and Inductor's 8.1. Pins on top of that
  seam only hurt (a `coop-t` GEMV piece there runs 90 µs).
- **The decode GEMVs stay at the two-launch floor.** The single-launch cooperative forms sit on 4-8 CTAs (6.7-6.9
  µs); a cross-CTA split fills the card at the price of a combine launch (q_proj 2.0 + 1.1, k_proj 2.9 + 1.3, v_proj
  2.2 + 1.3 µs). `t512`, `t32x8` and `coop-t/r2` are not rows for these shapes; `t64` runs 22 µs.
- **The two prefill attention forms have no correct row on main.** Every route of softmax × V — the recorded
  `PLACE@map.1/twist.2/inner=cut` route, its split-KV variant, a further score/softmax cut, the fused kernel and the
  scalar fallback — returns a wholesale wrong answer at S=512 (97% of elements, max abs 3.2); the same failure
  reproduces at the #775 and #782 merge commits, and the archived 09-10 lane already reported this target as
  failing its strict repeats, so the 147.5 µs number in the section below was never a correct row. SDPA + o_proj +
  residual builds since #782 but every form hangs past the 2 s kernel watchdog, including the o_proj cut and the
  attention halves. Both stay inventory rows.
- **The prefill score-statistics form** still fails to lower (`no extent for coordinates ['in6']`); its root-most
  cut exceeds the 60 s bench watchdog. Inventory row.

### Systems and provenance

- Host `bench-h100-wgmma-0910-0903-212a` (GCP `a3-highgpu-1g`, SPOT, us-central1-a), one H100 80GB HBM3
  (`GPU-b9509f7c-dbfc-557b-daba-4eff2fb9420b`), driver 580.173.02, nvcc 12.9.41, PyTorch 2.14.0+cu130, triton 3.8.0,
  the same image as the 09-10 run; the `id_ed25519` key of the 09-10 memo is gone from the VM (Google rewrites
  `authorized_keys` from instance metadata), the lane used the `google_compute_engine` key.
- Tuning ran from a git worktree at main `f01a23e7` imported over the VM's shared venv; the lane staged the branch's
  tree into `~/.local/share/emmy/h100_x_1/repo`, whose editable install had to be re-pointed at that tree (it pointed
  at an rsync copy from the #787 session, which the recipe's `./venv/bin/emmy` would have imported instead).
- Source `f8debb51`, clean staged tree; run directory `2026-09-11_09-18-50`; both task records and both `artifacts.tar.gz` are
  in the archive. About 190 tuning rows and their `--json` records are host-local under `~/tune2/` on the VM.

### Durable files

- Goldens: `golden/qwen3-06b-s1_h100.golden.yaml`, `golden/qwen3-06b-s512_h100.golden.yaml`, every row recorded on
  main `f01a23e7`.
- Raw-results archive: `results_h100x1.tar.gz` (root member `2026-09-11_09-18-50/` with the two `*.experiment.yaml` records
  and the two `*_artifacts.tar.gz`); it replaces the 2026-09-10 archive.

## Platform h100x1 — earlier hand-tuned corpus and the wgmma tier (2026-09-10; goldens superseded above)

### Question and scope

Can the nine-target Qwen3-0.6B layer-0 corpus (sequence lengths 1 and 512, the same embedded programs as the A100
goldens) beat Inductor on an H100 with hand-found schedules, and what does each loss come from? The H100 goldens
were built from the A100 files with the card identity changed and every knob and measurement stripped, so the
compared programs are identical across the two cards. Search was manual: seven rounds of `emmy run --ab` and pinned
runs (about 150 measured rows), then `--record-greedy` under the winning pin into the working golden. The same PR
adds the Hopper `wgmma` tensor-core tier, and the k-projection's recorded row uses it. Committed as
`golden/qwen3-06b-s1_h100.golden.yaml` and `golden/qwen3-06b-s512_h100.golden.yaml`; the recipe replays them like
the A100 row.

### Protocol

One `a3-highgpu-1g` SPOT VM (H100 80GB HBM3, GPU-b9509f7c, driver 580.173.02, nvcc 12.9, PyTorch 2.14.0+cu130,
cupy 14.2). Every number is deployable `-O3`, 10 warmups, 100 iterations, eager and Emmy in one process; Inductor is
the separate `torch.compile` lane the recipe runs once per task. The tuning rounds ran with a task-owned tune DB; the
recorded rows were measured against a fresh DB (see the failure-row finding below). The lane is one
`emmy bench experiments/golden-bench-2026/kernels --filter deploy.gpu=…` invocation against that host over SSH at
source `2d4f9510`, run `20260910T210813Z` (21:08-22:06 UTC), five strict repeats per sequence length.

### Result summary

Medians of the five strict repeats; Inductor from the torch-compile lane of the same task. Decode: nine of nine
targets correct on every repeat, task status succeeded. Prefill: seven of nine measured and correct on every repeat;
the two attention forms below have no realizable row, so the task status is failed by design.

| decode (sequence length 1) | eager | Inductor | Emmy | Inductor / Emmy | launches |
| --- | ---: | ---: | ---: | ---: | ---: |
| input RMSNorm | 114.1 | 2.84 | **2.27** | **1.25** | 1 |
| q_proj + q_norm statistic | 4.30 | 2.57 | 4.39 | 0.59 | 2 |
| k_proj + cast | 29.6 | 4.47 | 5.31 | 0.84 | 2 |
| v_proj | 30.1 | 3.99 | 4.78 | 0.83 | 2 |
| v_proj + 1-key SDPA | 9.46 | 8.19 | 10.9 | 0.75 | 2 |
| 1-key SDPA + o_proj + residual | 12.7 | 9.08 | **8.70** | **1.04** | 4 |
| post-norm + gate/up + SiLU | 124.4 | 7.67 | 25.3 | 0.30 | 2 |
| down_proj + residual | 7.80 | 3.69 | 222.9 | 0.02 | 2 |
| q/k norm + RoPE + score statistics | 307.7 | Inductor compile failed | 2.58 | — | 1 |

| prefill (sequence length 512) | eager | Inductor | Emmy | Inductor / Emmy | launches |
| --- | ---: | ---: | ---: | ---: | ---: |
| input RMSNorm | 174.9 | 6.32 | **2.88** | **2.20** | 1 |
| q_proj + statistic | 5.02 | 5.05 | 7.19 | 0.70 | 1 |
| k_proj + cast (wgmma n64) | 66.5 | 8.97 | **8.70** | **1.03** | 1 |
| v_proj | 42.2 | 7.32 | 7.41 | 0.99 | 1 |
| softmax × V | 16.9 | 17.4 | 147.5 | 0.12 | 3 |
| post-norm + gate/up + SiLU | 193.8 | 24.2 | 143.0 | 0.17 | 2 |
| down_proj + residual | 11.5 | 11.8 | 18.7 | 0.63 | 1 |
| SDPA + o_proj + residual | — | — | no realizable row | — | — |
| q/k norm + RoPE + score statistics | — | — | fails to lower | — | — |

Run-to-run spread over the five repeats stayed within 3% on every measured target; the softmax × V and the two MLP
rows are the cold greedy's own picks (their recorded rows failed strict or did not exist), reported as measured.

### What the sweep found

- **Decode GEMVs are parallelism-bound, not bandwidth-bound.** With lanes on the output axis (`coop-t`) a
  1024-output projection runs on four CTAs (6.9 µs for 2 MB of weights). A cross-CTA K split (`g8k`, `g16k`) fills
  the 132 SMs and reaches 4.4-5.3 µs, but the finalize launch keeps the pair behind Inductor's single kernel
  (2.6-4.5 µs). Atomic splits (`g<n>a`) fail to bench: the accumulator-reset gap already recorded for split-K.
- **The decode down-projection's fast rows fail the strict gate.** Every cooperative row, split or not, is one element
  in 1024 off eager by four fp16 ulps, over the 1e-3 gate: the fp16 rounding boundary before the residual add, the
  defect recorded on V100. None may be promoted, so the file keeps an inventory row and the cold tensor-core pick
  runs at 223 µs.
- **Prefill GEMMs on mma.sync top out at the 64×64 CTA tile** (`w2x2`, `f2x4/k4`, three-deep cp.async ring): 128
  CTAs, one wave. Bigger tiles leave SMs idle, K splits add a launch, TMA and producer bands do not beat cp.async at
  these shapes. q_proj (N=2048) runs at 240 TFLOPS against a cuBLAS-class 433 TFLOPS Inductor kernel.
- **The wgmma tier lands the k-projection.** `wgmma_m64n64k16` on `w8x1`, `f1x8/k4`, `d3/smem-tma` takes k_proj from
  10.4 to 8.7 µs against Inductor's 9.0; on v_proj it measures 7.7 beside the mma.sync row's 7.4, so that row stays.
  The corpus weights are N-contiguous, which pins the tier to one 64-column swizzle atom per K row: every wider
  N-contiguous tile measured wrong wholesale, because the row-major slab does not store each further atom as its own
  eight K rows the way the descriptor's MN-major layout expects. A K-contiguous 2048³ f16 GEMM runs the n256 row at
  34.0 µs against cuBLAS's 24.6 (0.73×) and the best mma.sync row's 69.3, all strict. Follow-up, same day: the
  atom-major B slab lifted that restriction (the N-contiguous 2048³ GEMM goes from 48.2 µs at n64 to 32.9 at n128,
  strict), but the corpus rows above do not move — at S=512 a 128-wide N tile halves an already short grid (k_proj
  8.4 → 9.7 µs, v_proj 7.7 → 8.8, down_proj 18.6 → 23.8 at `w4x1`), and a `g2k` / `g4k` split lands at 19 µs.
- **The two fused decode attention forms recover with a materializing cut.** 1-key SDPA + o_proj + residual: 7.2 ms
  cold → 8.7 µs with `PLACE@map.1/inner.2/map=cut` and a split GEMV on the children, beating Inductor. v_proj + 1-key
  SDPA: 32.8 → 10.9 µs. The post-norm MLP cut (`PLACE@map.2/inner.1/map=cut`) helps at both lengths but its
  materialized norm still runs as a direct kernel: a bare `WORK`/`REDUCE` pin applies to both children of the cut,
  and editing the producer's child receipt by hand did not reach it either.
- **Failure rows poison pinned replays.** Once `bench_fail` rows for a kernel are in the tune DB, later pinned
  compiles of that kernel silently drop the pinned tensor-core tile and emit the scalar form; the same pin realizes
  against an empty DB. Every recorded row was measured against a fresh DB. The disqualification should be visible.
- **Two prefill attention forms have no realizable row.** SDPA + o_proj + residual emits duplicate accumulator
  declarations for every tensor-core pin: the online-softmax state is emitted twice inside the o_proj's computed-A
  operand fill (the chain-form recompute already diagnosed on A100). The score-statistics form fails to lower in the
  cold greedy (`lower: no extent for coordinates ['in6']`) and its cut routes exceed the 60 s bench watchdog. Both
  stay inventory rows without measurements, as in the A100 file, and the walk now reports them without hiding the
  targets after them.
- **Softmax × V realizes on tensor cores through the corpus cut route** (`PLACE@map.1/twist.2/inner=cut`) at
  148 µs against 17 µs Inductor; the hand-recorded version of that route failed strict, so the row stays the cold
  pick's and is not promoted.

### Systems and provenance

- Host `bench-h100-wgmma-0910-0903-212a` (GCP `a3-highgpu-1g`, SPOT, us-central1-a), one H100 80GB HBM3
  (`GPU-b9509f7c-dbfc-557b-daba-4eff2fb9420b`, PCI `2330`), driver 580.173.02, nvcc 12.9.41, cuBLAS 12.9.0.13,
  Ubuntu 24.04.5, PyTorch 2.14.0+cu130, triton 3.8.0, cupy-cuda12x 14.2.0.
- The single pass-through H100 fails CUDA initialization until NVLink is disabled in the driver
  (`NVreg_NvLinkDisable=1`): the VM sees no NVSwitch for the fabric manager and the GPU's fabric state never leaves
  "In Progress". The image also lacks `make`, `g++`, `python3.12-venv` and `python3.12-dev`.
- Source `2d4f9510`, clean staged tree; run directory `2026-09-10_21-08-13`; both task records and both
  `artifacts.tar.gz` are in the archive.

### Durable files

- Goldens: `golden/qwen3-06b-s1_h100.golden.yaml`, `golden/qwen3-06b-s512_h100.golden.yaml`.
- Raw-results archive: `results_h100x1.tar.gz` (root member `2026-09-10_21-08-13/` with the two `*.experiment.yaml`
  records and the two `*_artifacts.tar.gz`).

## Current-head corpus requalification (2026-08-29)

The draft is based on current main `b88763fa`; the exact combined source for this pass is `857ba7e9`. Every hardware
run used deployable O3, the exact GPU capability, task-owned tuning and cubin state, the repository CLI, and strict
direct correctness where a timing was admissible. The retained A100 VM stayed running.

### A100 corpus

The exact sm80 lane collected 201 tests and found six applicable cases. It completed in 189 seconds with five passes,
one independent stat-fill watchdog failure, and 195 skips. The result JSON and full log are retained host-local on the retained A100 VM (untracked
`_tune/a100-corpus-857ba7e9/evidence/full-corpus/`; `_tune/` is not in the repository).

| A100 case | Emmy | `torch.compile` | launches | result |
| --- | ---: | ---: | ---: | --- |
| `attention/rmsnorm-gqa-b-cut.yaml` | 6.907 µs | 10.026 µs | 2 | Correct; 1.45x faster |
| `attention/sdpa-computed-value-cut-mma.yaml` | 6.193 µs | 8.039 µs | 2 | Correct; 1.30x faster |
| `attention/rmsnorm-qk-sdpa-workspace-chain.yaml` | 75.855 µs | 11.469 µs | 1 | Correct; 6.61x slower |
| `matmul/f16-cut-splitk-unit-row.yaml` | 9.183 µs | 2.839 µs | 3 | Correct; 3.23x slower |
| `matmul/f16-mma-broadcast-batched-pv-transpose.yaml` | 17.664 µs | 11.894 µs | 1 | Correct; 1.49x slower |
| `attention/rmsnorm-gqa-sdpa-stat-fill.yaml` | about 280 ms/iteration | 19.456 µs | — | Pinned row exceeds the aggregate 10-second watchdog |

The GQA win required both a better route and one replay fix. Materializing the clustered normalized-K value produces
one cooperative producer and one consumer. The A/B path then incorrectly dropped scoped OFF exceptions such as
`REDUCE@a7=''`, allowing bare `REDUCE=coop` to fan out and change the consumer source. Replay now preserves a scoped
OFF when it overrides a non-OFF bare family. The unchanged perf command fell from 10.193 to 6.857-7.100 µs and uses
the same fast source as direct full-row replay.

The PV golden now carries the strict `w8x1`, `f1x8/k8`, `d1/smem-async` schedule. This is about four times faster than
the prior checked row, but an eight-row neighbour search found no further schedule gain. The kernel uses 48 KiB shared
memory, 126 registers per thread, and 25% occupancy. Its transpose epilogue emits 32 stride-512 scalar stores per
thread; toggling vector stores produces byte-identical CUDA. Parity therefore needs a transpose-aware store path or a
different contraction orientation.

The workspace-chain golden improved 13.5% by using `WORK=t8` and cooperative reductions, with the combined lane
reaching 75.855 µs. Its CUDA still recomputes Q RMSNorm within output-key work, K RMSNorm within each dot product, and
the softmax score scan across value-output lanes. No offered schedule changes that structure; reuse or materialization
must move those cones outside the repeated loops.

The split-K case remains a launch-structure gap. Its fastest strict forced row with a genuine split used `g2k+w1x1`
at 4.681 µs versus 2.761 µs for `torch.compile`, but the corpus row remains the repeatable authored 9.183 µs result.
Deferred split-K requires partial, finalize, and cut-consumer kernels. Atomic split-K removes the finalize kernel but
needs a runtime accumulator reset because its first kernel has no predecessor to own zero initialization. A safe
improvement needs a cross-CTA last-arriver primitive or a proved consumer-side reset protocol.

The stat-fill case now builds and passes strict correctness after preserving provider evaluation domains, ordering
sibling operands by dependency, closing tiled provider cones, and retaining computed-B projection providers. Its
authored fused schedule is nevertheless about 0.28 seconds per iteration and trips the benchmark watchdog. A correct
six-seam route reached 3,342 µs versus 17.238 µs for `torch.compile`; its best measured factor-16 child was 184.525 µs.
The parent instead selected an unmeasured factor-32 split, so a 206.592 µs parent canary is not qualified evidence.
The remaining storage gap is ordering in the child-identity schedule receipts: record the split choice first, then join the identities
and measured schedules of the children that choice creates.

### Other exact platforms

| platform | exact corpus coverage | current result |
| --- | --- | --- |
| V100 sm70 | 2 cases | Volta MMA is correct at 3,130.368 µs versus 757.760 µs; linear-cut latency is inadmissible because strict correctness still has 26,418/524,288 mismatches |
| RTX 4090 sm89 | 0 cases | 201 collected, 201 skipped; no exact sm89 closed case and therefore no parity claim |

The promoted V100 row uses `WORK=w4x2` and a single `d1/smem` stage. It shares A tiles across two N warps and B tiles
across four M warps, reducing shared-memory requests, but remains 4.13x behind `torch.compile`. Volta lacks
`ldmatrix` and `cp.async`; the next useful schedule primitive is a producer/consumer warp band with named barriers,
not another depth or geometry row. The linear-cut target still differs in contraction accumulation order after the
public f16 boundary is restored, so its perf-harness number is reported only as diagnostic output.

### Retained fixes and conclusion

This pass retains small, separately tested fixes for provider evaluation domains, dependency-ordered operand splicing,
scalar-atom dump replay, direct tuning of persisted unscheduled Tile children, provider-cone closure, and scoped-OFF
A/B replay. It promotes the GQA, PV, workspace-chain, and V100 schedules. The combined local gate is 3,981 passed,
1,012 skipped, and five expected failures; lint is clean.

Parity is not achieved: A100 has two wins, three measurable losses, and one watchdog; V100 has one correct loss and
one correctness failure; RTX 4090 has no exact corpus coverage. No large serving experiment was started while those
kernel-level gaps remain. The next compiler work is structural reuse/materialization for attention, transpose-aware
PV stores, a cross-CTA completion/reset primitive for split-K, split-first ordering for stat-fill's child-identity
schedule receipts, and a Volta
producer/consumer staging primitive.

## Cut-pinned attention qualification after the computed-B changes (main through `d2950079`, 2026-08-28)

### Corrected protocol

The first screen put `PLACE` choices in proposal `knobs`. That measured one structural candidate whose new kernels
kept greedy schedules; it did not test the route the compiler is designed to tune. This follow-up instead froze the
kernel set in realization `pins`, ran the normal two-level tuner on every minted kernel identity, and replayed the
assembled route from the same isolated evidence state. A CPU regression test now protects that exact contract: a
pinned placement cut enrolls both children and the assembly replays different `WORK` and `STAGE` rows.

The full route has four distinct value seams: shared statistics (`PLACE@map.fold.a21`), Q
(`PLACE@a.map.a`), K (`PLACE@a.map.b`), and softmax weight (`PLACE@map.fold.a1`). The three previously tested K
spellings resolve to three different Fold occurrences, but value clustering groups them into one K-value `CutSite`;
pinning any occurrence replaces all three. They are one cut, not three composed cuts.

The first measurement lanes used exact `6e6181d5` source, deployable `-O3`, isolated tuning state, seed 0, at most 12
candidates per independent kernel, patience 4, and an outer wall bound. The receipt-aware follow-up rebased the draft
onto exact `a597f15d` and regenerated the working files before inspecting or measuring any schedule. A fresh trace
still produces one maximal whole-layer target, so these diagnostics use untrusted copies of the checked self-contained
score/statistics slice. They are host-local compiler qualification, not replacement publication evidence; no results
archive was changed.

### Hardware result

| platform | frozen route | result |
| --- | --- | --- |
| V100 | K-value cut; two children | Correct replay: 595,101 µs versus 1,920 µs eager. No child candidate finished inside the bounded search, so replay correctly used the offline fallback. |
| A100 | statistics + Q + K + softmax weight; five primary children plus one recursive statistics split | 67 clean benches in 219 s. Best children were 7-37 µs except the softmax-weight producer at 110,744 µs; tuned route 110,882 µs. Fresh replay was 110,723 µs versus 758 µs eager and passed direct correctness. |
| RTX 4090 | statistics + Q + K; four launches | Per-identity DB replay used different child schedules: 4,561 µs versus 494 µs eager, direct correctness passed. The remaining consumer was 4,396 µs. |
| RTX 4090 | add softmax weight; five launches | The consumer fell to 19-31 µs, but the new producer's best row was 100,416 µs; replay was 101,233 µs versus 504 µs eager and passed direct correctness. |
| RTX 5090 | statistics + Q + K + softmax weight; structural replay only | Exact lowering produced the expected five children. Timing was deferred because an unrelated task owned the host's only compatible GPU; it was not interrupted. |

### Receipt-aware current-main retune

The follow-up regenerated the working targets after rebasing. V100, A100, and RTX 4090 measurements used exact
`043f1f25`, which adds only the route-contract test to `a597f15d`; RTX 5090 used exact `5ddf7816`, whose receipt
decoder change does not alter kernel source. All rows used deployable O3 and bounded candidate or explicit-row
budgets.

| platform | child | best bounded row | result |
| --- | --- | --- | --- |
| V100 | K-cut cast producer | `TILE=f2`; other schedule families off | 2.924 µs |
| V100 | K-cut pointwise producer | `TILE=f4`; other schedule families off | 2.686 µs |
| V100 | K-cut attention consumer | only offered row: all schedule families off | exceeded the 15 s watchdog; no accepted latency |
| A100 | softmax materialization (`c3d`) | `WORK=t128, REDUCE@a3=coop, REDUCE@a4=coop` | qualifying repeats 110,768 and 110,786 µs |
| RTX 4090 | softmax materialization (`c3d`) | `WORK=t128, REDUCE@a3=coop, REDUCE@a4=coop` | search observations 100,351 and 100,335 µs; a noise-scale tie with the prior 100,416 µs row |
| RTX 4090 | other four-cut children | per-child rows: statistic `t32/coop`, Q all-off, K `t128/coop`, consumer f2x8 MMA with async stage | 4.15-27.89 µs |
| RTX 5090 | softmax materialization (`c3d`) | `WORK=t128, REDUCE@a3=coop, REDUCE@a4=coop` | after the compiler fix, qualifying repeats 72,380 and 71,966 µs |

The A100 c3d candidate-pool bound is 1,094,745,632 rows and the consumer bound is 1,066,670,432. Whole-target MCTS
spent 8m30s in first-candidate CPU descent without measuring a row. On RTX 4090, the 24-live-candidate MCTS-only arm
took 407.5 s and the equally bounded evidence-seeded refinement reached its 600 s wall; neither found a different c3d
schedule. Exhaustive child-row listing and strict receipt decoding were each stopped at 60 s. The useful schedules
are visible by deploy identity, but flattening these pools is not a usable listing or validation algorithm.

Current main's child-identity schedule receipts close the representation gap, and `5ddf7816` fixes strict decoding
when a regenerated target lowers to several kernels. Exact deployment is not closed yet. On RTX 4090 the canonical
t128 receipt joined the correct c3d identity but reported row DRIFT and fell back to t8: 397,303 µs for c3d and
397,477 µs for the four-cut route versus 480 µs eager, with direct correctness passing. The explicit working-file
path also treats receipt siblings as independent flat A/B rows rather than installing them together for base
lowering. No receipt was promoted; the remaining work is a child-directed exact-row descent shared by strict decode
and the verified tier, plus grouped working-file replay.

RTX 5090 exposed one independent built-stage gap: every screened row initially emitted ambiguous `float * __half`
expressions under readable CUDA rendering. The readability fold had inlined a mixed-dtype single-use `Assign` before
the target-aware renderer could insert `__half2float`. The compiler now keeps such assignments named; the new closed
sm120 realization case proves offered, realized, built, and correct. The repaired explicit rows measured 72,488 µs
at t32, 72,405 µs at t64, 71,969 µs at t128, 73,936 µs at t256, and 73,362 µs at t512. This closes compilation but
does not change the repeated 512×128 work.

### Statistics-sharing replay after #682

PR #682 (`d2950079`) directly closes the repeated-statistics gap identified above: Tile normalization restores object
sharing between structurally equal cones, and two provider-closed statistics seams make the shared row state
materializable. On the exact Qwen s512 target, adding those two cuts to the prior four-cut route produces six launches.
The statistics producer writes max and normalization state once per `(head, query)` row; the softmax-weight child
loads that workspace and no longer contains the 512-key scan. The consumer also loads the shared state rather than
recomputing it.

The retained A100 was replayed at exact branch source `31e7e629` (PR #682 plus this draft's receipt and readable-CUDA
fixes), deployable O3, isolated DB/prior/cubin state, five warmups, and 20 iterations. Both standard repeats passed the
strict direct eager check.

| lane | eager (µs) | Emmy route (µs) | dominant statistics child (µs) | result |
| --- | ---: | ---: | ---: | --- |
| standard repeat 1 | 744.653 | 11,717.632 | 11,501.568 | correct; 6 launches |
| standard repeat 2 | 744.795 | 11,720.704 | 11,505.664 | correct; 6 launches |
| `FAST_MATH` | 744.590 | 11,704.320 | 11,501.568 | correct; noise-scale 0.1% change |

The prior four-cut full replay was 110,723 µs, so the shared-statistics route is 9.45x faster. The old c3d child falls
from about 110,780 µs to 65-66 µs; the consumer is 81 µs and the other three producers are 7-37 µs. This is a real
algorithmic improvement, but the route remains 15.7x slower than eager. The matched `torch.compile` request again
produced no positive timing for this embedded target, so it still cannot supply a parity ratio.

The new bottleneck is the one correctly shared statistics producer, not duplicated work. Its greedy row is
`TILE@a4=f1x2, WORK=t32x8` and takes 11.50 ms. A bounded MCTS-only follow-up measured
`TILE@a4=f4x6, WORK=t32x16` at 11.535 ms and found no improvement; after 2m55s the next candidate remained in CPU
descent with the GPU idle, so the arm was stopped. The remaining performance work is to make the nested 128-channel
score contraction inside the online 512-key statistics reduction eligible for an efficient tensor-core schedule,
then reduce the still-large candidate descent. No receipt was promoted. Raw host-local evidence is retained under
`_tune/pr682-a100/remote/`; the task-owned remote scratch was removed while the A100 VM stayed running.

`torch.compile` produced no positive latency for these score/statistics strict replays, including the post-#682
attempt, so no parity ratio is claimed.
The earlier output-projection strict result remains a valid separate finding: 1,594,544 µs for Emmy versus 52.6 µs
for `torch.compile` on RTX 4090. The V100 down-projection also remains a direct correctness failure and was not
admitted as a performance result.

### Bottleneck and receipt-aware replay

Before #682, the correction changed the diagnosis. Placement worked, and resulting kernels were independently
schedulable. On A100 and RTX 4090, four children tuned into the tens-of-microseconds range; materializing the softmax
weight isolated one producer that remained about 100-111 ms. Its lowered loop had free query and output-key axes and,
for every output weight, recomputed the complete 512-key reduction whose body performs a 128-channel score
contraction. Ordinary `WORK` and `REDUCE` choices changed the constant factor but preserved that repeated scan. PR #682
closes that reuse gap; the statistics-sharing replay above supersedes this performance state. On V100, even the K-cut
consumer did not complete a candidate inside the original search wall.

The earlier cold-replay drift was a separate persistence gap: the DB keyed different rows by child structural
identity, but the old flat realization could not serialize conflicting child-global `WORK`, `TILE`, `REDUCE`, `STAGE`,
or `RASTER` values. Main `a597f15d` resolves that representation gap with child-identity schedule receipts. Each
sibling realization carries the route cuts in `pins`, one child's row in `knobs`, and that child's `deploy_identity`
in `identity`; strict decoding checks the row only against that child's candidate pool. Copying child rows into the
parent flat map remains invalid, but the sibling receipts make exact per-child replay representable in the schema.
The current-main retune above shows that strict enumeration and deploy equality still need a child-directed descent
before those receipts are promotion-ready for this large route.

No realization was promoted. Offered, realized, built, and correctness stages are closed for the composed route, so
there was no small compiler failure or new realization-corpus gap to patch. `FAST_MATH` was not promoted because the
standard route remained far behind eager and changing contraction math does not remove the isolated producer cost.

## Host-local exact-card qualification checkpoint (2026-08-28)

### Question and scope

Can the current Qwen3-0.6B FP16 layer-0 kernel inventory match or beat `torch.compile` on the exact V100, A100,
RTX 4090, and RTX 5090 cards after bounded retuning? This pass covered the same nine targets at sequence lengths 1
and 512 on each card: 18 targets per platform. It did not retune the historical FP8, 32B, H200, B200, or serving
lanes, so it supports no claim about those workloads.

This was bounded tuning and exact `emmy run --golden-file ... --bench` qualification, not a new `emmy bench` recipe
snapshot. The checked-in `results_*.tar.gz` files and the later platform sections therefore remain the earlier
archived runs. Current raw evidence is retained in the ignored `_tune` directories listed below; it is not presented
as a replacement experiment record or archive.

Consequently, this section is a host-local working checkpoint for compiler and golden review, not durable publication
evidence. The paper must not cite its exact counts or timing ranges until the recipe is rerun and the per-platform
raw-results archives are replaced through the normal experiment workflow.

### Protocol and acceptance rule

- Model: `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, layer 0, FP16, static sequence lengths
  1 and 512.
- All measurements used deployable `-O3`. Ordinary screens used five warmups and 20 iterations; promoted finalists
  used 10 warmups and 100 iterations in at least two fresh processes. Hard targets used one warmup and one iteration
  under a bounded watchdog rather than extending the run indefinitely.
- Direct Emmy-versus-eager correctness was the admission gate. `torch.compile` was compared only when it compiled
  the whole target and passed its own eager check; an unavailable baseline did not turn a correct Emmy target into a
  failure. A hung or incorrect Emmy target remained unresolved.
- Win/tie/loss follows the preregistered two-percent rule in the experiment README. Counts use only targets with a
  valid `torch.compile` result, and the denominator is reported explicitly.
- `FAST_MATH` was considered only when the standard row had empty compile flags and the fast-math row independently
  passed direct eager correctness. It was promoted only when it changed the performance conclusion.

### Platform summary

| platform | direct eager correct | comparable with `torch.compile` | win / tie / loss | baseline unavailable | unresolved | current golden status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| V100 | 16/18 | 14 | 1 / 0 / 13 | 2 | 2 | REPOSITORY validation passes 18/18; one row fails runtime correctness |
| A100 | 16/18 | 15 | 5 / 1 / 9 | 1 | 2 | sequence 1: REPOSITORY 9/9; sequence 512: WORKING 7/9 |
| RTX 4090 | 17/18 | 15 | 3 / 1 / 11 | 2 | 1 | REPOSITORY validation passes 18/18 |
| RTX 5090 | 18/18 | 16 | 3 / 1 / 12 | 2 | 0 | REPOSITORY validation passes 18/18 |

The result does not establish cross-platform parity with `torch.compile`. Emmy matches or beats it on 1/14
comparable V100 targets, 6/15 A100 targets, 4/15 RTX 4090 targets, and 4/16 RTX 5090 targets. Several targets,
especially attention targets, are orders of magnitude slower or hang; the V100 MLP down-projection remains
numerically incorrect.

The strongest new positive promotion is the RTX 4090 sequence-512 input RMSNorm: `WORK=t256, REDUCE=coop` measured
2.3641-2.3663 µs in the qualifying repeats, against 4.6793-4.6999 µs for `torch.compile` and
126.61-126.68 µs for eager. The repaired A100 sequence-1 score/statistics schedule repeated at 34.99 µs and passed
direct correctness, but it still loses to the 11.26-12.16 µs `torch.compile` result.

### Main unresolved targets and performance losses

| platform | target role | result |
| --- | --- | --- |
| V100 | sequence-512 down-projection + residual | all four bounded schedules are incorrect; the stored row has 40/524,288 mismatches |
| V100 | sequence-512 o-projection + residual | the selected kernel exceeds the internal watchdog |
| A100 | sequence-512 softmax times V | correct greedy execution is about 327 ms versus about 46 µs for `torch.compile`; no safe recorded route |
| A100 | sequence-512 o-projection + residual | bounded execution hangs, so no timing or correctness result is admitted |
| A100 | sequence-512 score/statistics | the corrected lowering path still hangs under the bounded greedy replay |
| RTX 4090 | sequence-512 score/statistics | correct, but 164-372 ms; `torch.compile` is unavailable |
| RTX 4090 | sequence-512 softmax times V | correct, but 202-235 ms versus 36.1 µs for `torch.compile` |
| RTX 4090 | sequence-512 o-projection + residual | warmups take about 3.69 s before the internal watchdog aborts |
| RTX 5090 | sequence-512 score/statistics | correct at about 397 ms; `torch.compile` is unavailable |
| RTX 5090 | sequence-512 softmax times V | correct at about 152 ms versus 30.8 µs for `torch.compile` |
| RTX 5090 | sequence-512 o-projection + residual | correct at about 3.35 s versus 40.1 µs for `torch.compile` |

The V100 correctness diagnosis found that fusion removes a public f16 rounding boundary before the residual add.
Restoring that boundary reduced mean error but did not reproduce eager matmul accumulation and rounding, and it
changed the kernel identity. The partial change and its provisional realization case were therefore reverted; the
target remains an explicit compiler correctness gap rather than a promoted schedule.

### `FAST_MATH`

Only the sequence-512 q-projection changed category: RTX 4090 improved from about 20.7 µs to 16.4 µs against a
20.4 µs `torch.compile` result, and RTX 5090 improved from about 19.1 µs to 16.6 µs against roughly 19-20 µs.
Other direct-correct fast-math rows on A100, V100, RTX 4090, and RTX 5090 were ties or losses and were not promoted.
Rows that changed outputs were rejected regardless of speed.

### Compiler changes established by this pass

- Singleton reduction collapse, computed-A statistic-fold selection, and guarded unit-row recovery close reduced
  correctness or realization failures seen in the Qwen targets.
- Golden feature derivation, exact path retry, measured-row latency recording, and bounded cold-pool descent repair
  replay and search without adding a separate benchmark harness.
- Placement-cut capture preservation, unit-axis preservation, scoped-cut consumption, and output-sweep promotion
  make the affected schedules realizable. The affected closed realization cases reached `built` with nvcc on their
  exact-capability cards.
- A proposed structural-route receipt was audited and reverted. The final design fails closed: whole-slice
  `PLACE` measurements are not written as deploy evidence, and both tuning-database and online-reservoir measured
  tiers reject legacy rows containing `PLACE` or `PLACE@...`. Search may retain those rows for ranking and training,
  but automatic deployment cannot use them without an exact child-schedule receipt.

The final compiler suite reports 3,911 passed, 990 skipped, and 5 xfailed; `make lint` passes. The route contract
is now a positive fail-closed test replacing one prior xfail.

### Systems and evidence

| platform | exact GPU and software | qualification source | ignored local evidence |
| --- | --- | --- | --- |
| V100 | Tesla V100-SXM3-32GB, `GPU-b415579d-cdad-42bb-23d1-32c20cdb729d`; driver 580.159.03, nvcc 12.9, PyTorch 2.13.0+cu126 | `0e4729d5` | `_tune/v100-current/current-head/` |
| A100 | A100-SXM4-80GB, `GPU-80df657e-2e14-421c-32a5-cb2429dc93e6`; driver 580.65.06, nvcc 12.9, PyTorch 2.13.0+cu130 | `15c27422` | `_tune/a100-current/` |
| RTX 4090 | GeForce RTX 4090, `GPU-81d79c00-868e-3ec5-2948-745283b756f6`; driver 580.159.03, nvcc 13.3, PyTorch 2.13.0+cu130 | `7b5161e8` | `_tune/rtx4090-current/safe-head/` |
| RTX 5090 | GeForce RTX 5090, `GPU-bb78f2c5-11d6-02d6-f124-08b719623110`; driver 580.173.02, nvcc 13.0, PyTorch 2.13.0+cu130 | `5e95d2de` | `_tune/rtx5090-current-source-revalidate/` |

The compiler tree after the final safety patch is `c9d8a19e`. Later changes relative to a card's qualification source
are search or safety changes, except for measurement-only golden promotions already qualified at the listed source.
They do not supply unmeasured performance claims for that card.

### End-to-end decision

Large serving experiments were intentionally skipped. The kernel gate is not healthy: the common corpus is
incomplete on V100 and A100, and every card has major `torch.compile` losses in attention. Running a long serving
matrix now would consume hardware without supporting the requested across-platform compiler claim.

## Earlier archived refresh: first fix wave (main @ 001d4f44)

## What this refresh is

Second measurement pass over the corpus, after the maintainer's fixes landed (#561 split-axis re-fusion, #556
conv1d/einsum lowering, #549/#547 FA restoration, #513-era search changes). Same three GPUs, fresh hosts,
recipe budgets, three `-O3` repeats per target. Each `goldens/<model>_<gpu>.yaml` was rebuilt from a fresh
trace + tune + 3-repeat verification on `001d4f44`; the previous values remain in git history for comparison.

Runs were driven by an updated benchmark flow — one row per committed golden on its exact GPU, three `-O3`
repeats of `emmy run --golden <file> --bench --bench-backends eager,tcompile,emmy`, no tracing or tuning at
measurement time — kept UNCOMMITTED in this PR per review direction; `recipe.yaml` in-tree is unchanged.

## Before/after (sum of measured kernel targets, median of 3 repeats, µs; old = previous committed goldens)

| platform | model/seq | old emmy | new emmy | gain | new vs eager |
| --- | --- | ---: | ---: | ---: | ---: |
| v100x1 | 0.6B s512 | 71714 | 58981 | 1.2x | 0.07x |
| v100x1 | 0.6B-FP8 s512 | 147160 | 71084 | 2.1x | 0.10x |
| v100x1 | 32B-FP8 s512 | 408241 | 395773 | 1.0x | 0.06x |
| rtx4090x1 | 0.6B s512 | 35305 | 34466 | 1.0x | 0.03x |
| rtx4090x1 | 0.6B-FP8 s512 | 34496 | 24525 | 1.4x | 0.04x |
| rtx4090x1 | 32B-FP8 s512 | 650248 | 117839 | **5.5x** | 0.08x |
| rtx5090x1 | 0.6B s512 | 32870 | 27620 | 1.2x | 0.03x |
| rtx5090x1 | 0.6B-FP8 s512 | 28830 | 17843 | 1.6x | 0.10x |

Decode (s1) rows improved 4-13x on the FP8 corpora but changed measurement coverage (19 -> 8-11 targets, from
new fusion identity plus bench failures), so their sums are not clean ratios; per-target values are in the
archives. Matched-kernel gains on the V100 0.6B corpus: geomean 2.6x, led by q_proj 4.5x (992 -> 219 µs, 0.88x
eager on Volta — #561's tensor-core unlock confirmed in silicon), k/v_proj 3.4x, and the SDPA matmul fusions
21.5x / 16.8x (FA restoration).

## Why layer totals still trail torch.compile: the remaining defects, diagnosed

1. **RoPE-fusion statistic replay (dominates every s512 total; unchanged).** `k_sdpa_mean_reduce`'s loop nest
   recomputes the k-norm statistic (a full 512x128 reduce) inside every q-row iteration — a 512x replay
   (~23000-29300 µs of each card's s512 total; the s1 variant is fine, replay factor 1). Consistent with the
   #513 guard removal enumerating this fused form and cold greedy deploying it. Fix directions: hoist
   loop-invariant statistics in loop/canonicalize, or make the placement-cut alternative evidence-reachable
   cold. Diagnosis-only here per review direction.
2. **Computed-A (fused norm+gate/up) misdeploys, worst at decode.** New extreme case: on the rtx4090,
   `k_linear_mean_reduce_549927.s1` deploys KNOBLESS at **116445 µs vs eager 108** (~1000x); the V100 s512
   sibling regressed 679 -> 1035 µs. The search reaches no schedule for this form and the fallback is
   catastrophic.
3. **Qwen3.6-27B capture advanced one op and is blocked again**: conv1d now lowers (#556), the trace now stops
   at `aten.masked_fill requires resolved self, mask, and fill inputs`. Still no 27B golden.
4. **Hung kernel under the 32B corpus on V100**: `k_mul_12__partial` exceeds the 2 s bench watchdog in the
   tuned deploy (16/19 targets measured around it).
5. **Unmeasured golden rows are real bench failures, kept as inventory**: fp8 files carry 11-22 unmeasured
   realizations each (hangs, compile failures, or the coverage change); only the 0.6B BF16 files validate at
   REPOSITORY level on all three cards, the rest at WORKING level.

## Environment caveats (hosts are rented and heterogeneous)

- The rtx4090 host ran an old driver (CUDA 12.2-era) and nvcc 12.1: the default cu130 torch cannot initialize
  (fixed with the cu126 wheel + matching cu12 libraries), and a subset of fp8 kernels fail to COMPILE under
  nvcc 12.1 that compiled under CUDA 13.3 on the first pass — its fp8 numbers carry that asterisk.
- V100 requires torch cu126 and `cupy-cuda12x==13.6.0` + `fastrlock` (nvrtc 13 dropped Volta), as before.
- Pre-run canaries must check BOTH `cupy.full` AND `torch.cuda.is_available()`; a cupy-only canary passed on
  the old-driver 4090 while every torch-side measurement failed.

## Platform a100x1 — earlier routing measurements (2026-08-23; not current deploy evidence)

An earlier pass measured two positive routing realizations at deployable `-O3` on the exact
NVIDIA A100-SXM4-80GB (`GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`):

| seq | target | routing realization | Emmy | fused/cold Emmy | eager | result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 512 | score + softmax statistics | `PLACE@b=cut` | 299.69 µs | 21232 µs | 745.47 µs | 2.49x eager |
| 1 | post-attention norm + gate/up + SiLU | `PLACE@map=cut` | 66.97 µs | 4612 µs | 154.62 µs | 2.31x eager |

Both routing realizations passed strict Emmy-versus-eager correctness in that pass. They are not current automatic
deploy evidence: the 2026-08-28 audit showed that a `PLACE` total does not identify the exact child schedules it
measured, and the final compiler fails closed on such rows. The current qualification above supersedes this section.

At the 2026-08-23 boundary this was direct tuning evidence; no matching experiment snapshot was produced.
`results_a100x1.tar.gz` below remains the 2026-08-21 run and does not measure the two routing realizations.

## Platform a100x1 — historical chain-form replay (2026-08-21)

### Question

`main` now carries the chain root formation: a fold closes over the values its projection body defines, so the RoPE
gathers and the k-norm no longer survive as their own kernels — they become part of the score kernel's tree. That
re-keys every computed-A attention target. Which committed realizations survive the re-key, what do the re-keyed
targets cost once they are tuned again on this card, and — now that every fold is a node with a `PLACE` seam — does
any cut beat the fused form on the targets that lose?

### Identity diff

Both inventories were re-traced from `Qwen/Qwen3-0.6B@c1899de2` layer 0 and every target name was compared with the
committed golden. A surviving identity kept its committed knobs and measurements verbatim.

| seq | committed | re-traced | carried verbatim | re-keyed or unpinned | absorbed by the score kernel |
| --- | ---: | ---: | ---: | ---: | --- |
| 512 | 12 | 9 | 6 | 3 | RoPE cos gather, RoPE sin gather, k-norm + RoPE |
| 1 | 10 | 9 | 6 | 3 | q/k norm + RoPE |

The three re-keyed targets per sequence are the score/statistics kernel, softmax·V, and o_proj + residual. The last
two keep their names but had no committed schedule (their knob maps have never been recordable), so they were tuned
from scratch as well.

### Protocol

Each re-keyed target was hybrid-tuned on this card: agent proposals drawn from the card's own measured schedules plus
every `PLACE` seam the recognize rule enumerates, then `emmy tune --max-candidates 48 --patience 12 --seed 0` under a
per-target wall budget, with an isolated tuning DB, online checkpoint, and cubin cache. Every finalist was re-measured
at deployable `-O3` against the cold greedy pick, then verified in a fresh `emmy run --golden … --target … --bench
--strict` process. The recipe then replayed both committed goldens in five fresh
`emmy run --golden … --bench --strict --bench-backends eager,tcompile,emmy --warmup 10 --iters 100` processes with an
empty per-task tuning DB, online checkpoint, and cubin cache.

### Per-kernel result (median of five processes, µs)

`greedy` is the cold pick re-benched in the same process; `deployed` is the committed realization where one exists and
the greedy pick otherwise.

| seq | target | role | eager | torch.compile | greedy | deployed | vs eager | vs tcompile |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | k_mean_20f978 | input RMSNorm | 191.54 | 8.69 | 3.82 | 3.17 | 60.4x | 2.74x |
| 512 | k_linear_reduce_06a42b | v_proj | 12.97 | 13.70 | 83.03 | 17.50 | 0.74x | 0.78x |
| 512 | k_linear_1fd3d5 | q_proj + reshape to heads | 100.64 | 21.65 | 132.97 | 22.35 | 4.50x | 0.97x |
| 512 | k_linear_a09c5a | k_proj + reshape to heads | 59.86 | 14.73 | 71.21 | 17.28 | 3.46x | 0.85x |
| 512 | k_sdpa_linear_reduce_c0a378 | softmax·V, computed V | 45.78 | 46.04 | 159.57 | 159.57 | 0.29x | 0.29x |
| 512 | k_linear_sdpa_reduce_e24efe | o_proj + residual | 60.16 | 61.27 | 315.39 | 190.46 | 0.32x | 0.32x |
| 512 | k_linear_mean_reduce_dc067d | post-attn norm + gate/up + SiLU | 246.39 | 49.62 | 67.27 | 67.27 | 3.66x | 0.74x |
| 512 | k_linear_6b4b5f | down_proj + residual | 32.09 | 37.21 | 303.10 | 46.57 | 0.69x | 0.80x |
| 512 | k_sdpa_mean_reduce_29d3df | q/k norm + RoPE + scores + softmax stats | 745.29 | failed | 21286.91 | 19609.60 | 0.04x | — |
| 1 | k_mean_b8e46d | input RMSNorm | 121.85 | 2.89 | 3.05 | 2.38 | 51.2x | 1.21x |
| 1 | k_linear_reduce_7ef15d | v_proj | 7.42 | 7.21 | 12.39 | 4.74 | 1.57x | 1.52x |
| 1 | k_linear_49a16b | q_proj + reshape to heads | 35.09 | 6.79 | 21.45 | 4.84 | 7.25x | 1.40x |
| 1 | k_linear_dfb21f | k_proj + reshape to heads | 33.67 | 5.25 | 11.06 | 4.74 | 7.10x | 1.11x |
| 1 | k_sdpa_linear_reduce_d0f5c0 | softmax·V, computed V | 11.91 | 10.62 | 25.22 | 20.72 | 0.57x | 0.51x |
| 1 | k_linear_sdpa_reduce_14c8c7 | o_proj + residual | 14.06 | 12.16 | 36.86 | 24.44 | 0.58x | 0.50x |
| 1 | k_linear_mean_reduce_549927 | post-attn norm + gate/up + SiLU | 154.62 | 16.38 | 4605.95 | 393.22 | 0.39x | 0.04x |
| 1 | k_linear_2dcd0c | down_proj + residual | 9.57 | 7.64 | 35.13 | 9.46 | 1.01x | 0.81x |
| 1 | k_sdpa_mean_reduce_0a2624 | q/k norm + RoPE + scores + softmax stats | 334.55 | failed | 39.82 | 36.53 | 9.16x | — |

Every target measured in all five repeats of both rows: the two `torch.compile`-less targets are ordered last in their
golden, so their strict failure no longer costs the later targets their measurement.

Layer totals as the sum of those medians: sequence 512 is 1494.7 eager, 252.9 Inductor (eight of nine targets), 22423
untuned Emmy and 20134 deployed Emmy; sequence 1 is 722.7 eager, 68.9 Inductor (eight of nine), 4791 untuned and 501
deployed. Sequence 1 improves on the previous revision (522 → 501 µs, 1.44x eager). Sequence 512 does not: one target,
the fused score/statistics kernel, is 19610 µs of the 20134 µs total. Over the other eight targets the sequence-512
layer is 524 µs against 749 µs eager, or 1.43x.

### Why the fused score kernel costs 19.6 ms

Its tile IR is unambiguous. The kernel places `free=(head, query)` and sweeps the key axis on the store; inside that
sweep it runs the whole k cone per cell — a 128-element k-norm fold over `to_4`, then the k RoPE — so every k vector is
recomputed once per query row rather than once per key. At sequence 512 that is 512x redundant arithmetic, and it is
the whole cost: 21243.9 µs of the 21286.9 µs untuned total is that single kernel, at 94% occupancy and 34 registers.
No thread tier changes it (`WORK=t256/t512/t1024` measure 21558 / 21835 / 22126 µs; `coop-t` measures 37873 µs).

### Cut options

The recognize rule enumerates three seams on the score root — bare `PLACE` (the score dot `acc2`), `PLACE@a2` (the
q-norm fold), and `PLACE@fold.fold.a4` (the k-norm fold) — plus one on the softmax·V root. Every one was measured at
deployable `-O3` against the fused form in the same process.

| seq | target | fused | cut option | cut | verdict |
| --- | --- | ---: | --- | ---: | --- |
| 512 | k_sdpa_mean_reduce_29d3df | 21289.98 | `PLACE@fold.fold.a4=cut` (k-norm fold) | 19625.98 | 1.08x — committed |
| 512 | k_sdpa_mean_reduce_29d3df | 21289.98 | `PLACE@a2=cut` (q-norm fold) | 335166.47 | 0.06x |
| 512 | k_sdpa_mean_reduce_29d3df | 21289.98 | both seams | 335165.44 | 0.06x |
| 512 | k_sdpa_linear_reduce_c0a378 | 159.57 | `PLACE=cut` | 4466.69 | 0.04x |
| 512 | k_linear_sdpa_reduce_e24efe | 315.39 | `PLACE=cut` | 366.59 | 0.86x |
| 1 | k_sdpa_linear_reduce_d0f5c0 | 25.17 | `PLACE=cut` | 26.22 | 0.96x |
| 1 | k_linear_sdpa_reduce_14c8c7 | 36.82 | `PLACE=cut` | 37.56 | 0.98x |
| 1 | k_sdpa_mean_reduce_0a2624 | 39.58 | no legal seam on this tree | — | — |

One cut won in this historical run, and it was the k-norm fold: materializing that reduction once removed 8% of the
redundant work. It is not current automatic deploy evidence for the reason stated above. The seam that would matter
is not in the set. Cutting `a2`
promotes the key axis to a free axis of the residue, whose grid becomes 4.2M blocks and whose cost rises 16x, because
the k cone is then recomputed with no reuse at all. What the target needs is the RoPE'd k vector materialized once as
the dot's B operand; that is a binding the contraction binder still declines, not a seam the placement fork can spell.
For the softmax·V and o_proj forms the cut is a straight loss: it splits a working mma contraction into a scalar
producer plus a workspace zero-fill (`__zp524288`, 48.5 µs on its own in the o_proj cut).

### Schedules that measure but cannot be recorded

Four targets measured a deployable win that the golden's one-knob-map-per-realization format rejects. A multi-kernel
target whose kernels include a knob-free one (an elementwise epilogue such as `k_add_5`, or `k_sdpa_reduce_fe4eb9` at
sequence 1) always fails the merge: that kernel records the empty value for every schedule family while the others
record the pinned value, so `realized_tuning_knobs` sees `WORK: '' != 't512'` and returns nothing. The pin itself is
uniform and replays, so these four realizations record the exact `--ab` pin instead of the merged realized map, and
each was re-verified by replaying the committed golden in a fresh strict process.

| seq | target | greedy | recorded pin | deployed |
| --- | --- | ---: | --- | ---: |
| 512 | k_linear_sdpa_reduce_e24efe | 315.39 | `WORK=w8x2,TILE=mma_m16n8k16_f16_f32/f1x8/k8,STAGE=d2/smem` | 190.46 |
| 1 | k_linear_sdpa_reduce_14c8c7 | 36.86 | `WORK=t512,REDUCE=coop-t` | 24.44 |
| 1 | k_sdpa_linear_reduce_d0f5c0 | 25.22 | `WORK=t512,REDUCE=coop-t` | 20.72 |
| 1 | k_sdpa_mean_reduce_0a2624 | 39.82 | `WORK=t128,REDUCE=coop/r2` | 36.53 |

### Repeat variation

Every target's five paired latencies agree to within 0.6% of their median (0.57% at sequence 1, 0.52% at 512), and
every committed realization reproduces its tuning-time `-O3` measurement to within 0.5%.

### Defects this round surfaced

1. **The online-prior refit aborts the tune.** `emmy tune` raises `_catboost.CatBoostError: All features are either
   constant or ignored` from `OnlinePrior.fit`, reached through `measure_proposals`' `prior.maybe_refit()`, when the
   first measured proposal contributes a run of rows whose feature vectors are identical. It killed the whole
   invocation for four of the six re-keyed targets on a cold online checkpoint. Re-running the same command against a
   checkpoint that already carries a varied dataset succeeds, which is the workaround used here.
2. **The tune-lane bench watchdog censors an expensive target completely.** Every candidate of the 21 ms fused score
   kernel exceeds the 2 s accumulated-GPU-time budget and is marked `bench_fail`, so a full 1800 s search ranked
   nothing at all and wrote no `ranking` block. `EMMY_BENCH_RUN_TIMEOUT_S` raises the budget; at 60 s the search
   instead spent its whole 2400 s in re-lowering without completing a candidate, so this target's schedules were
   priced by direct `emmy run --ab` instead.
3. **`torch.compile` still cannot compile the RoPE-bearing attention reference** on PyTorch 2.13.0, so `--strict`
   rejects those two targets in every repeat and both rows report `failed`. This is the same Inductor limitation the
   previous run recorded.
4. **A bare `PLACE=cut` pin re-cuts every piece it produces.** Because the pin is authoritative on each freshly
   recognized tree, the resolution recurses through the fragments; on the sequence-512 score tree a single
   `emmy compile --ir tile` under that pin had not terminated after ten minutes. Named seams resolve promptly.

### Conclusion

The re-key is mostly benign: six of nine realizations per sequence carried over verbatim and reproduce their committed
numbers, and sequence 1 is faster than the previous revision (1.44x eager). Sequence 512 regressed by construction —
folding the k cone into the score kernel made it recompute that cone once per (query, key) pair, which costs 19.6 ms
against 745 µs eager and turns a 1.56x layer win into a 0.07x loss. The placement fork is real and usable: its seams
enumerate, resolve by name, and one of them is now committed evidence, but the seam that would undo this particular
regression is a contraction binding rather than a placement.

### Limitations

- Layer-0 evidence only, one model, one card; never a whole-model claim.
- Both rows report `failed` because `--strict` requires a `torch.compile` latency the two RoPE-bearing targets cannot
  produce. Every other target passed strict Emmy-vs-eager correctness in all five repeats.
- The Inductor column is missing for those two targets, so no geometric mean over the full corpus is available; the
  measured denominators are stated above.
- A target's program includes the producers its output needs, so the attention targets overlap and the layer total is
  a sum of overlapping sub-programs on both the Emmy and the eager side, not a disjoint decomposition.
- The five repeats share one deployed host and run back to back, so they capture process-level, not day-level,
  variation.

### Run and system

- Status: failed (2/2 rows, `torch.compile` reference unavailable on the two RoPE-bearing targets; every target
  measured in every repeat)
- Result timestamp: 2026-08-21T23:03:34Z; run ID: `20260821T230334Z`
- Rows: `…sl1_scommon` (row ID `551082cef77b`, 440.74 s) and `…sl512_scommon` (row ID `3a4d139974b8`, 2131.30 s)
- Git revision: `213c443a`; dirty: false
- Host: `riftvm`; Ubuntu 24.04.1 LTS; AMD EPYC 7742 64-Core Processor
- GPU: NVIDIA A100-SXM4-80GB, UUID `GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`; PyTorch 2.13.0+cu130

### Durable files

- Raw-results archive: `results_a100x1.tar.gz`; archived root `2026-08-21_23-03-34/`
- Members: both `*.experiment.yaml` records, both `*_artifacts.tar.gz` task archives (per-repeat verification JSON per
  target, per-repeat logs and exit statuses, package freeze, replayed working golden), and the two runner logs
- Committed goldens: `golden/qwen3-06b-s1_a100.golden.yaml`, `golden/qwen3-06b-s512_a100.golden.yaml`

## Platform sections

### v100x1 — full pipeline (rebench + retune, 4 models attempted, 27B blocked at trace)
Goldens: `qwen3-06b_v100.yaml` (18/18, REPOSITORY), `qwen3-06b-fp8_v100.yaml` (27/38), `qwen3-32b-fp8_v100.yaml`
(23/38). Archive: `results_v100x1.tar.gz`.

### rtx4090x1 — full pipeline; measurements re-run after the driver fix
Goldens: `qwen3-06b_rtx4090.yaml` (18/18, REPOSITORY), `qwen3-06b-fp8_rtx4090.yaml` (16/38),
`qwen3-32b-fp8_rtx4090.yaml` (19/38). Archive: `results_rtx4090x1.tar.gz`.

### rtx5090x1 — full pipeline on a replacement host (first instance had unstable SSH and a failing toolchain)
Goldens: `qwen3-06b_rtx5090.yaml` (18/18, REPOSITORY), `qwen3-06b-fp8_rtx5090.yaml` (27/38).
Archive: `results_rtx5090x1.tar.gz`. Large models excluded by RAM fit (30 GB host), as before.

## Limitations

Layer-0 evidence only; `-O3` numbers throughout; tcompile per-target values live in the archives (its lane
fails on some SDPA targets); s1 sums are not comparable across passes due to coverage changes; multi-kernel
targets record `knobs: {}` with per-kernel `record_knobs` in the archives.

## Platform h200x8 — earlier failed attempt (retained)

### Conclusion

The latest non-dry invocation failed before tracing, tuning, or kernel benchmarking began. It produced no latency,
correctness, or coverage measurements and supports no kernel-performance claim. The failure is retained because
dry-run validation is not a result.

### Protocol and failure

The invocation selected one `common` row on a pre-allocated host detected as eight NVIDIA H200 141GB GPUs. The task
used one GPU and targeted `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, layer 0, sequence length 512,
with search budget 12, patience 4, and seed 0.

Remote setup reached the command after staging a clean source tree. The command then failed because `make` was not
installed. Its exit trap encountered a second error because `task_dir` was unset, so the intended `artifacts.tar.gz`
was never created or retrieved. The command result records exit code 1 and the missing-result transfer error. The
runner summary is 0/1 successful tasks.

### System and provenance

- Hardware detected: NVIDIA H200 141GB x8; the task requested one GPU.
- Timestamp: `2026-08-13_16-08-20_e1c8d16a`.
- Git revision: `030b6d58182bb3da1748c4954d7d2fd0211e8d3b`; staged source was clean.
- Workload status: failed before measurement.
- The legacy command result has no assembled experiment record and no complete typed system-information section.

### Durable files

- Raw-results archive: `results.tar.gz`.
- Archived root: `2026-08-13_16-08-20_e1c8d16a/`.
- Supporting evidence: runner logs, the executed recipe snapshot, the task manifest, and the command result JSON are
  in the archive.
