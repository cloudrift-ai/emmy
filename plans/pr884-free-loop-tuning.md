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
- NVIDIA Tesla V100 16 GB: Qwen3.8-27B-FP8. Rental completed and terminated after collecting results.
- NVIDIA Tesla V100 32 GB: DeepSeek-V4-Flash-0731. CUDA 12.9, PyTorch 2.13 with CUDA 12.6, capability 7.0.
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

This is an identity repair, not a measured whole-model speedup. The 5090 subsequently developed a driver/library
mismatch, preventing additional GPU comparisons; the task did not change drivers or reboot the shared machine.

### Qwen FP8: avoid spills in the changed matrix kernel

The larger merged matrix schedule initially selected an `f8x1` register tile and took roughly 42 ms. A small manual
tile change, using the same Volta MMA family with `f2x1/k8`, reduced register pressure to 127 registers with no spills.
The changed kernel took about 1.70 ms. The whole selected target measured 6.532 ms and 6.531 ms on repeated O3 runs,
versus 7.850 ms on main on the same V100 16 GB: about 17% lower latency for this target.

The measured schedules and routes are promoted. One obsolete expected failure is removed; none are added. The final
Qwen decode/fresh-lowering subset reported 169 passed, two expected failures and two failures also present on main.

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

One other apparent regression disappeared on fresh baseline measurement: a target with a historical 3.625 ms receipt
took 5.521 ms on main and 4.262 ms on the PR. Do not compare a new receipt only to a stale historical number.

Two strict full-target O3 replays of all eleven DeepSeek variants are in progress. Final measurements and promotion
status must replace this paragraph before claiming parity or merge readiness.

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
- Lint passed across 845 files before this requested documentation addition.
- Complete repeated DeepSeek replay, promote only validated rows, strictly decode the proposed golden, and rerun its
  fresh-lowering check. Record any remaining regression explicitly rather than restamping it away.
- Collect the V100 32 GB results, terminate that rental, and audit the lease. Leave the supplied 5090 running with
  other agents' data untouched.

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
