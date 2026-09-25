# PR #884: conservative tuning and qualification

Requested review artifact, 2026-09-25. Keep this file with the PR at the author's request. Results below concern
stored model targets and selected kernel sets, not end-to-end serving throughput.

## Goal and stopping rule

The ambition is conservative. The minimum is parity with main's measured results and evidence that the PR has not
made the best available knobs worse. Start with existing knobs and preserve the techniques that made them fast:
placement cuts, enough parallel work, cooperative reductions, matrix tiles that avoid spills, and split reductions
where useful. Adapt identities and site names before changing schedules. Take an easy improvement when the emitted
code or a nearby measured configuration suggests one; otherwise stop at parity. Do not pursue a broad model tune.

The author reported that tuning with the prior is broken. After that instruction, all candidate selection is manual.
Small earlier prior-driven attempts did not resolve the regressions and are not evidence for the final choices.
This is not the tune skill's equal-budget hybrid-versus-MCTS experiment; that procedure was explicitly superseded by
the author's manual, conservative scope.

## State at the requested pause

The author paused tuning and requested documentation of findings and PR state. Finish only measurements already
running; do not add candidate ideas or expand the search. PR #884 remains a draft. Compiler fixes, the main merge,
Gemma/Qwen FP8 golden repairs, and the AWQ identity repair are reviewable changes. DeepSeek's fresh-target
recordings remain working artifacts; its canonical file still contains the minimal routing repair only.

GitHub reports conflicts with the latest main; the previous merge was at `c1ba9e5b`. Conflict resolution is
deferred at the author's request. Remaining qualification work is to select the retained DeepSeek recordings,
verify the complete canonical edit
with row decoding and fresh-lowering tests, and resolve or explicitly account for the remaining baseline/CI
failures. No claim is made that the full suite is green or that all best-available schedules have proven parity.
Further tuning is paused, not implicitly authorized by this remaining-work list.

## Incidence in the stored model inventories

Fresh-lowering comparison with only the new sibling-free-loop merge disabled versus enabled:

| Inventory | Emitted final Loop kernels | Changed final Loop kernels |
| --- | ---: | ---: |
| DeepSeek-V4-Flash-0731, V100 | 152 | 8 |
| Gemma-4-12B-it, RTX 5090 | 40 | 0 |
| Qwen3.8-27B-FP8, V100 | 129 | 6 |
| Qwen3.8-27B-AWQ-INT4, V100 | 46 | 3 |
| Qwen3.8-27B-GPTQ-Int4, V100 | 40 | 2 |
| Qwen3.8-27B-EXL3, V100 | 38 | 2 |

The comparison lowers each distinct stored frontend program through `LOOP_PASSES` under the golden's declared
capability, pairs final kernels by output buffers, and compares their serialized Loop IR. The listed inventories
have no added or removed final kernels in this comparison. Counts include emitted kernels that have no configured
stored golden entry; DeepSeek has 149 configured entries but emits 152 kernels across its 12 stored programs.

This measures changed final bodies, not every intermediate application, affected schedule identity, or execution
frequency. Gemma demonstrates the distinction: its final Loop bodies remain the same while 26 root routing
identities need repair. Multiple shape variants or timing rows can describe one target. No serving-throughput
improvement follows from these counts alone, and no NVFP4 performance claim is made.

## Comparison and hardware

- Main baseline: `c1ba9e5b`, merged into the PR by `be423aff`.
- Compiler fix for wholly overhanging Volta fragments: `e15ddbfb`. Later commits only update documentation or goldens
  unless specifically recorded below.
- NVIDIA Tesla V100 SXM2 16 GB: Qwen3.8-27B-FP8. Rental completed and terminated after collecting results.
- NVIDIA Tesla V100 SXM3 32 GB: DeepSeek-V4-Flash-0731. CUDA 12.9, PyTorch 2.13 with CUDA 12.6, capability 7.0.
  Rental terminated after the final archive was collected and checksum-verified; the lease audit reports absent or terminal.
- NVIDIA GeForce RTX 5090: Gemma-4-12B-it. The supplied machine is shared; task data stays in a private subdirectory.
- Correctness suite: at most four pytest workers. Timed runs: sequential GPU jobs, deployable O3, normally 10 warmups
  and 100 iterations. Preserve the stored shapes, symbolic hints, bindings and precision pins; standard rows use
  `FAST_MATH=false`. Use the CLI's fixed default seed consistently.

All measurements use the native `emmy run` CLI. Task scripts only prepare structural inputs or orchestrate those
commands; they do not implement timers. Compare PR and main on the same card. Stored historical timings are useful
starting points, but a fresh main replay takes precedence when it disagrees with the stored number.

## Scope and findings

### Gemma: preserve the schedules

The change invalidated root routing identities while the child schedules remained valid. Repairing 26 root
identities removed 182 PR-specific decode failures. Exact-card decode then reported 512 passed, 18 expected failures
and six failures also present on main. A representative pre-attention replay passed strict O3 validation at about
36.9 microseconds across eight launches. Child schedules and their historical timing receipts were preserved.

This is an identity repair, not a measured whole-model speedup. The 5090 subsequently had a driver/library
mismatch: loaded module 580.95.05, installed module and libraries 580.178.04. After the author authorized repair,
a standard Ubuntu reboot loaded the installed module; NVML and a CUDA computation passed. No reinstall was needed.

### Qwen FP8: avoid spills in the changed matrix kernel

The larger merged matrix schedule initially selected an `f8x1` register tile and took roughly 42 ms. A small manual
tile change, using the same Volta MMA family with `f2x1/k8`, reduced register pressure to 127 registers with no spills.
The changed kernel took about 1.70 ms. The whole selected target measured 6.532 ms and 6.531 ms on repeated O3 runs,
versus 7.850 ms on main on the same V100 16 GB: about 17% lower latency for this target.

The measured schedules and routes are promoted. One obsolete expected failure is removed; none are added. The final
Qwen decode/fresh-lowering subset reported 169 passed, two expected failures and two failures also present on main.

### Qwen AWQ INT4: one identity repair

The final full-suite comparison found one additional PR-specific row, which passed on main. Its existing all-off
reduction settings and `WORK=t32` still decode after updating the identity. Exact-card O3 replays on the V100 SXM3
32 GB passed: main measured 280.576 and 283.136 microseconds; the PR measured 276.736 and 279.296 microseconds.
No new schedule search was needed. The identity-only edit preserves its knobs and historical receipt; the fresh
paired measurements are separate validation evidence. Local validation: 90 passed, one inherited expected failure.
This is parity, not a whole-model performance claim.

### DeepSeek: qualify eight fresh targets and eleven shape variants

A minimal repair of 12 root identities and their cut spellings first removed 11 PR-specific decode failures.
The resulting subset reported 441 passed, six expected failures and 19 failures also present on main. Eight stored
targets still differed from fresh lowering. Their eleven shape variants are the remaining performance scope.

Old cuts were matched to fresh cuts by the values they represent. Most variants already approached or beat main's
stored latency without new search. Three findings needed more work:

1. **Wholly overhanging Volta fragments.** A four-column operand exposed an out-of-bounds fragment read in the
   4096-token case. Clamping an offset could not repair a fragment whose base was already outside the operand.
   Three guards zero-fill such fragments. Compute-sanitizer reported 5,090 errors before and zero after the fix;
   strict same-input replay passed. All eleven fresh variants then passed strict replay.
2. **Decode copy parallelism.** Reusing the old 256-thread setting reduced the initial 694 microseconds to roughly
   163 microseconds, still behind main's fresh 37.7 microseconds. Trying 128 and 512 threads did not close the gap.
   Generated CUDA showed that the merged loop left only four threads copying 4,096 elements each. One additional
   output-owning cut restored parallel copying. The complete target then measured about 33.8 microseconds with eight
   launches, versus main's 37.7 microseconds. This is a cut adjustment, not a new compiler optimization.
3. **Dynamic narrow matrix.** The new large Volta tile took about 5.7 ms. Main's cooperative scalar reduction took
   about 389 microseconds, but applying that exact schedule to the new matrix failed both strict same-input and
   independent eager comparisons. It was rejected. Three nearby small MMA tiles passed strict eager checks; the best
   unsplit candidate took roughly 0.74-0.79 ms. One four-way split of that candidate took about 169 microseconds plus
   a 1.7-microsecond finalize kernel, also passing strict eager comparison. No broader sweep was attempted.

The selected dynamic matrix knobs are `WORK=w2x1`, `TILE=mma_m8n8k4_f16_f32/f1x1/k8`, `REDUCE=g4k`, empty `STAGE`,
`LOOPIFY=0`, and empty `RASTER`. The decode candidate retains the mapped old cuts, adds `PLACE@map.2/map=cut`, and
uses `WORK=t256`. These are kernel-local spellings; a split intended for one child must not be pinned on every child.

One other apparent regression disappeared on a fair fresh baseline measurement: a target with a historical
3.625 ms receipt took 4.261 and 4.239 ms on main, versus 4.224 and 4.224 ms on the PR. An earlier 5.521 ms
main run had omitted its child split routes; it is superseded by the comparison that imports those routes first.
Do not compare a new receipt only to a stale historical number, or omit the baseline's recorded child schedules.

All eleven DeepSeek variants passed two strict O3 replay rounds. Recording the verified dynamic matrix split
restored complete-target parity: 10.729 ms and 10.727 ms, versus 10.914 ms on main. Retained-schedule checks and
reuse of that same matrix setting at the matching 4096-token target close the large prefill gap too: the first
4096-token target repeats at 28.758 and 28.859 ms, within 1% of fresh main replays at 28.644 and 28.632 ms.
Promotion remains pending final golden validation. The measurements below are complete-target pinned replay
latencies, not individual kernel times or end-to-end model throughput. Fresh same-card main results are labeled;
a historical receipt alone cannot establish a regression or a speedup.

| Case / stored target / regime | Baseline, µs | PR repeats, µs | Interpretation |
| --- | ---: | ---: | --- |
| 00 / loop 4 / M=1 | 37.7 fresh | 33.57 / 33.57 | Extra copy cut restores parallelism. |
| 01 / loop 4 / M=16 | 2960.89 historical | 2918.40 / 2893.82 | Existing mapped cuts; no fresh paired baseline. |
| 02 / loop 4 / dynamic | 10914 fresh | 10729.47 / 10727.42 | Small MMA tile and four-way split. |
| 03 / loop 4 / M=4096 | 28644.35 / 28632.06 fresh | 28758.02 / 28859.39 | Within 1%; not a demonstrated win. |
| 04 / loop 23 / dynamic | 4260.86 / 4239.36 fresh | 4224.00 / 4224.00 | Preserve incumbent child split routes. |
| 05 / loop 40 / M=1 | 735.38 historical | 167.25 / 167.94 | Correct repeated candidate; fresh main not paired. |
| 06 / loop 59 / M=1 | 405.86 historical | 370.18 / 371.20 | Correct repeated candidate; fresh main not paired. |
| 07 / loop 75 / M=16 | 2958.73 historical | 2941.95 / 2916.35 | Existing mapped cuts; no fresh paired baseline. |
| 08 / loop 94 / M=16 | 2683.90 / 2677.76 fresh | 2673.66 / 2677.76 | Parity after restoring recorded child routes. |
| 09 / loop 111 / M=4096 | 28680.48 historical | 28765.18 / 28753.92 | Same matrix setting as case 03. |
| 10 / loop 130 / M=4096 | 45242.92 historical | 44249.09 / 44177.41 | No fresh main pair; do not claim proven parity. |

Case 08 previously selected different child routes between recordings (2.808 and 4.408 ms). Importing its existing
recorded child splits before the two final replays restored 2.674-2.678 ms, matching fresh main. These final replays
were already running when tuning was paused. The earlier 4.408 ms candidate must not be promoted. Case 10 also finished both queued retained-route replays
with strict passes at 44.249 and 44.177 ms; its previous 46.047/46.378 ms recordings are superseded.
Its historical receipt is 45.243 ms, but there is no fresh paired main measurement for this case.

## Reproduction and evidence

Representative full-target recording command, using a working copy of the relevant golden:

```bash
emmy run --golden working.yaml --realization TARGET --pin-route \
  --bench --strict --bench-backends eager,emmy --record-greedy --json result.json
```

Representative manual matrix comparison, on the standalone extracted target with a matching eager reference:

```bash
emmy run --golden matrix-reference.yaml --realization pr884-deepseek-matrix-dynamic \
  --bench --strict --bench-backends eager,emmy \
  --ab 'WORK=w2x1,TILE=mma_m8n8k4_f16_f32/f1x1/k8,REDUCE=g4k,STAGE=,LOOPIFY=0,RASTER=' \
  --json matrix-split.json
```

Working YAML, exact commands, logs, O3 JSON, tune DB and prior snapshots are retained under
`_tune/pr884-20260925/`, including archives collected from each remote task. The DB is used for measured evidence;
its presence does not mean that prior-driven search selected the manual candidates. Correctness failures are not
promoted. Repeated winners must retain positive paired O3 measurements, actual realized knobs and live references.

The final V100 32 GB archive, `_tune/pr884-20260925/v100-32-final-results.tar.gz`, contains the per-case inputs, exact native CLI logs and JSON, source dumps,
`autotune-backup.db`, and the online-prior snapshot. Its verified SHA-256 is
`dbe2991aa003cc198ebbe464e780e4252f6408171898a40dc3d38a4b153a989b`.
Use these result prefixes for any later promotion review:

- Cases 00, 01, 05, 06, 07: `deepseek-verified-NN-r1/r2`.
- Case 02: `deepseek-split-whole-r1/r2`.
- Cases 03 and 09: `deepseek-static-NN-r1/r2`.
- Cases 04, 08 and 10: `deepseek-retained-NN-r1/r2`.
- Fresh main comparisons: `deepseek-main-baseline-00/02`, `deepseek-main-retained-04-r1/r2`, and
  `deepseek-main-final-03/08-r1/r2`.
- AWQ comparisons: `awq-main-r1/r2` and `awq-pr-r1/r2`.

An initial local aggregate, `deepseek-pre-final-candidate.yaml`, predates the retained 08/10 results and is not a
promotion-ready file. Its per-packet row decode completed, but the final assembled canonical file has not been
validated. Select the final recordings above before doing that validation; do not reuse the superseded aggregate.

Embedded Loop targets without a runnable frontend reference use `same-input-greedy`. That checks agreement between
schedules on identical inputs, not independent mathematical correctness. The standalone dynamic matrix received a
matching eager reference specifically to resolve the scalar-versus-MMA numerical disagreement. The isolated fragment
failure additionally received compute-sanitizer coverage.

## Validation and remaining work

- Dependency/fusion subset: 33 passed. Volta lowering subset after the bounds fix: 62 passed.
- Full suite at four workers after the fixes: 7,731 passed, 353 skipped, 37 expected failures, 52 failed and seven
  setup errors. This includes inherited golden failures, the pending DeepSeek fresh-target promotion, and local
  environment failures. It is not a green result.
- Exact-main rerun of 30 non-golden failing nodes: 22 failed, seven setup errors, one passed. The one difference,
  program rebinding after CUDA graph capture, also failed on main when run in isolation. Disabling the new loop
  rule on the PR did not fix it either.
- Final local lint passed across 845 files. At pre-documentation head `e9c9b667`, native, lint and package CI
  checks passed; the test job was still running. The final documentation/AWQ commit needs its own CI result.
- DeepSeek: two strict O3 rounds completed for all eleven variants. Fresh-target promotion is still pending;
  preserve the final retained child routes, strictly decode the complete edit and rerun its fresh-lowering check
  before calling the PR ready. No new tuning is started at this pause.
- Both V100 rentals are terminated and audited. The final 32 GB archive is collected and checksum-verified.
  The repaired supplied 5090 remains running; the task has no GPU processes on it and other agents' data is intact.

## Workflow limitations

The checked-in tune skill still describes prior-based search, while the author required manual sweeps because that
path is broken. Its candidate reasoning, exact-card validation and evidence-retention requirements still apply;
its search comparison protocol does not describe this run.

The existing CLI can flatten child routing pins onto unrelated kernels when a working file contains several routing
rows. This produced invalid split widths on a reduction of extent four. Qualification inputs therefore scope the
named root route and let each child select its own measured evidence. Final recordings retain kernel-local route
ownership. Fixing the generic CLI pin scope is separate work, not a reason to broaden this PR's tuning.

Fresh normalization can change GPU parallelism even when it removes duplicate work. The decode example shows why
identity repair and successful decoding alone do not establish performance parity. The manual checks stop after
the existing techniques recover parity or expose a cheap improvement.
