# DeltaNet investigation and proposed fix plan

Status: 🚧 Implementation approved and in progress. PR #973 was stacked on #969 in native GitHub stack #974;
GitHub automatically retargeted it to main after #969 merged. The original implementation baseline combines
main `81af0892` with #969 `57667193`. The bugtracker row is marked 🚧 on this branch.
Change it to ✅ with the fix PR link only when the agreed scope is ready for review; disclose any remaining support gap.

Stage status: 🚧 means in progress; ✅ means its deliverables have passed their checks. Pending stages have not started.
The initial investigation below records the pre-implementation revisions. Later sections record validation results.

## Revisions and recommendation

- Report baseline: `a98fd4f8`, investigated September 28, 2026; report merged in #961.
- Current GitHub main checked September 29: `81af0892d97a17ffdad07056406141228020cf31`.
- PR #969: OPEN, `5766719382129f0459f6dfbafcb1df4cf116bc2a`, based on `567a42ad`.
- Main was tested from a clean archive; #969 from its existing clean checkout. Separate temporary tune DBs were used.
- Saved checkpoint Loop IR inventories supply the exact named kernels. This avoids another checkpoint download but
  does not establish that a new full-model trace produces identical kernels. Fresh model tracing is an acceptance gate.

Recommend building the recurrence fix on the merged #969 plus current main. If work starts before it merges, stack
on #969 and incorporate current main, then rebase after merge. Its explicit carrying Fold is the right representation
for state, seed and previous-state reads. Reimplementing the old state-buffer lift would conflict with that direction.
The repack, padding and output-domain fixes do not intrinsically depend on #969 and can be separated for review.

However, #969 is not a fix for the report. It also rejects the real checkpoint's register program because of a
size-one batch coordinate. That regression should be addressed in #969 or explicitly included in our first patch.
No comment or message has been posted to the PR.

## What changed on main since the report

There are ten main commits after the report's tested revision. Relevant changes include:

- #963 (`0d147f67`): iterative deep schedule walks, plus Qwen3.8 V100 findings. It improves compiler traversal, not
  the reproduced fragment role loss or output-domain nesting.
- #930 (`567a42ad`): Qwen3-0.6B parity work, with substantial cut, scheduling, fragment/load lowering and evidence
  changes. It is already in #969's base. Old schedule selections and performance numbers cannot be assumed current.
- #968 (`889242cd`): TMA for stored f16 weights beside computed activations. Relevant once GDN projections can reach
  tensor-core schedules; it does not itself orient the multi-channel GDN projection correctly.
- #971 (`81af0892`): native fp4 TMA. It fixes another tracker row, not these bf16 GDN input projections or recurrence.
- #954 and #871 improve native sampling precision and paged KV-cache handling. Neither supplies GDN recurrent state
  or removes serving-twin capture's explicit `self_attn` requirement.
- #962, #964, #961 and #972 cover workflow, MoE precision, the report and plan consolidation respectively.

The padding and serving refusals remain. The existing term lowering already describes sibling loop trees, so the
output fix should repair how real output terms reach that mechanism rather than introduce another loop-placement API.
The report's `_sweep_start` explanation remains a hypothesis, not an established sole root cause.

## Fresh reproduction results

| Failure | Current main | #969 head |
| --- | --- | --- |
| Nonzero zero-fill padding | Small trace raises the reported `aten.pad` refusal | Same |
| GDN serving capture | Focused `_layer_signatures` call raises the explicit non-attention refusal | Same |
| Fragment role preservation | Isolated rename fails; real 512-token checkpoint kernel crashes in CUDA rendering | Isolated rename still fails |
| Checkpoint register scheduling | Register program accepted; default reaches malformed repack | Register program rejected on singleton batch coordinate; default emits scalar CUDA |
| Independent output sweeps | Fresh 64-token checkpoint CUDA nests all six domains | Same, in the original report's axis order |
| Input projection tensor cores | Not exhaustively re-enumerated | Fresh qkv/convolution piece remains weight-first with `REDUCE=coop`, no MMA TILE |

The #969 default replay's lack of an assertion is not recovery. It selected a different path. Explicitly pinning the
reported register schedule also did not produce a register kernel. The emitted scalar sources were not executed or
validated. They additionally warrant inspection: the default source has a serial `a0` argument and an inner loop
shadowing it; the pinned fallback source contains `std::vector` state storage inside the device kernel.

One existing #969 test was run on the local RTX 5080 Laptop GPU:
`test_register_state_starts_from_the_seed_tensor_on_cuda`: 1 passed, 20 deselected (0.41 s reported by pytest).
This validates that focused supported shape, not the real checkpoint or full GDN layer. No new performance claim
is made, and the known runaway kernel was not launched.

## IR examples: actual and intended

### 1. Loop IR carries the previous state correctly

Excerpt from the saved checkpoint target, freshly decoded on both revisions:

```text
for a0 in 0..7  # carries acc2[1, 48, 128, 128]
    for a1 in 0..48
        ...
        for a5 in 0..128
            in1 = load matmul_2[0, a1, a0, a4, a5]
            v1 = pre acc2[0, a1, a5, a3]
            v2 = multiply(in1, v1)
            acc0 <- add(acc0, v2)
        ...
        acc2[0, a1, a2, a3] <- v10  (seed 0)
        add_70__steps0[a0, 0, a1, a2, a3] = v10
    # commit acc2
```

#969 preserves this algebra in Tile IR instead of turning the state into a buffer during lift:

```text
Fold[a0 in 0..7]
    ... v1 = pre acc2[a1, a5, a3]
    init: (0)
    cells: (a1, a2, a3)
    combine: acc2 = next(acc2, acc2__o)
outputs
    add_70__steps0[a0, 0, a1, a2, a3] = v10__obs
    sweep(a6.a7) add_70__steps1[a0, 0, a1, a6, a7] = v13
```

This is useful. But `RegisterProgram.from_tile` compares the output's batch slice `(0, a1)` with the carrier's
`(a1,)` and returns `None` at line 80. The literal zero represents the original size-one batch dimension.

Intended result: keep the correct external output indices and the carrying Fold, derive physical batch ownership
consistently after accounting for proven size-one dimensions, and offer a register schedule again. Do not simply
drop every literal coordinate: validate it against the corresponding tensor dimensions and carrier mapping.

### 2. The fragment crash is a semantic field lost during renaming

Instrumentation of the actual main checkpoint compile captured this transition in `100_loopify`:

```text
# Before the rewrite:
FragmentRepack _rf203 <- ('_rf61',) (f16, m16n8k16, part=0, role=b)
# After the rewrite, and in final Kernel IR:
FragmentRepack _rf[203] <- ('_rf[61]',) (f16, m16n8k16, part=0)
```

The constructor defaults the missing role to A. An A repack expects two source fragments, whereas this B repack
correctly has one. Rendering fails at `assert len(self.srcs) == 2`.

Intended Kernel IR and CUDA, illustrative rather than emitted by a fix:

```text
FragmentRepack _rf[203] <- ('_rf[61]',) (f16, m16n8k16, part=0, role=b)
```

```cuda
emmy_c_to_b_f16(_rf[203], _rf[61]);
```

Use a field-preserving replacement when renaming the node, rather than rebuilding it with a partial field list.
Keep the renderer's assertion: the malformed node is the problem.

### 3. Separate Tile output domains become a Cartesian product in CUDA

Fresh #969 Tile IR, values abbreviated:

```text
outputs
├─ sweep(a0.a1)    reshape_9[0, a0, 0, a1] = ...
├─ sweep(a0.a1)    to_7[0, a0, a1] = ...
├─ sweep(a0.a1.a6) reshape_5[0, a0, 0, a1, a6] = ...
├─ sweep(a0.a1.a6) reshape_7[0, a0, 0, a1, a6] = ...
├─ sweep(a8.a10)   type_as[0, a8, a10] = ...
└─ sweep(a8.a11)   transpose_1[0, a8, a11] = ...
```

Fresh #969 CUDA, braces and calculations abbreviated:

```cuda
for (int a0 = 0; a0 < 48; a0++)
  for (int a1 = 0; a1 < 64; a1++)
    for (int a8 = 0; a8 < 64; a8++)
      for (int a6 = 0; a6 < 128; a6++)
        for (int a10 = 0; a10 < 5120; a10++) {
          type_as[a8 * 5120 + a10] = ...;
          for (int a11 = 0; a11 < 10240; a11++)
            transpose_1[a8 * 10240 + a11] = ...;
        }
```

That multiplies all six extents: 131,941,395,333,120,000 innermost iterations. Fresh main uses the order
`a8 → a11 → a0 → a1 → a6 → a10`, retaining the same product.

Intended lowered structure, schematic, before distributing work across threads:

```text
for a0 in 0..48
    for a1 in 0..64
        store reshape_9, to_7
        for a6 in 0..128
            store reshape_5, reshape_7
for a8 in 0..64
    for a10 in 0..5120
        compute/store type_as
    for a11 in 0..10240
        load/store transpose_1
```

The projection calculations must be placed with the outputs that depend on them; moving only the stores is
insufficient. Shared normalization statistics should remain shared. Keep maximal legal fusion and expose cuts and
worker schedules through the existing mechanisms; do not forbid fusion to hide the bad lowering.

### 4. The qkv/convolution contraction still has an unsuitable operand orientation

Actual #969 scheduled Tile IR:

```text
operand[acc4, acc5, acc6, acc7]: Fold[a3 in 0..5120] contraction  <REDUCE=coop>
    operand[in15]: load linear_wt[a3, a2]
    operand[v35, v37, v33, v31]: Fold free  <computed>
```

The stored K×N weight is first; it is shared across the four convolution-tap products. The desired scheduling
shape is conceptually `A[tap, token, k] × B[k, channel]`, with the normalized activation/tap values on A and the
stored weights on B, and an offered f16 MMA TILE. This is not necessarily a two-operand swap: the existing
multi-channel representation gives its shared operand a specific position. First evaluate existing cuts to
materialize the projection before convolution, then extend generic channel/orientation lowering only if necessary.
Do not hardcode this model or force the greedy scheduler to choose an unmeasured MMA candidate.

## Proposed implementation sequence

1. **Rebase the evidence on #969 plus current main.** Preserve the exact checkpoint kernel reproductions and add
   small regression cases. Include the singleton external batch dimension, nonzero seed, multiple chunks, and the
   separate corrected-value output. Recheck current PR head before starting; coordinate the singleton regression
   with its author only if authorized to post.
2. **Repair recurrence lowering.** Preserve `FragmentRepack` fields through rename/loopification. Fix the carrier
   matcher coordinate correspondence introduced by #969. Inspect both register and classic serial paths so a
   fallback cannot conceal lost scheduling coverage or a misplaced time loop. Keep the carrying Fold semantics.
3. **Repair output-domain placement.** Reduce the six-output checkpoint writer to a small executable case with
   independent domains and shared prefixes. Trace the boundary from scheduled Tile through `Fold.lower`, projection
   peeling and `apply_output_specs`; locate the first point that introduces unrelated axes. Reuse the existing
   term-owned placement to emit sibling computation/store nests. Cover load-bearing siblings, reordered outputs,
   shared intermediate values and worker distribution. Do not replace the bug with duplicated normalization.
4. **Add zero-fill padding support.** Lower constant padding through existing `IndexMapOp`/`IndexSource` machinery:
   guarded input coordinates plus a zero source. Preserve dtype and logical output shape; prove out-of-range lanes
   never read outside the input. Test sequence lengths 1, 16, 63, 64 and 65, including chunk-boundary behavior.
   Nonconstant padding modes are outside this model's requirement and should retain explicit refusals.
5. **Recover projection tensor-core options.** Re-evaluate cut choices on the new baseline, including projection
   materialization before the four-tap convolution. If necessary, repair general operand orientation and shared
   multi-channel handling. Require an offered, renderable, numerically valid MMA route for qkv and z projections.
   Use the now-available computed-f16 weight TMA route where legal, comparing it with existing transport choices.
6. **Address serving as an explicit integration stage.** The compiler repairs alone do not create a serving
   program. Add GDN layer classification/capture and a recurrent-state contract covering convolution history,
   matrix state, prefill-to-decode handoff, request reset, and batched request isolation. A mixed-model boot using
   existing validated GDN kernels with Emmy full-attention layers can be an intermediate milestone, labelled as
   such. Full native GDN completion requires its own stateful prefill/decode programs and reference validation.
   Keep this as a separate reviewable change if its size exceeds the compiler fixes.
7. **Validate, measure and prepare review.** Use the local 5080 for focused correctness and watchdog completion;
   the user-provided remote 5090 for repeatable deployment measurements and capacity where sufficient. Keep its
   checkout, caches and artifacts in a new dedicated directory alongside other agents' work; leave their files and
   running jobs untouched. Check for existing GPU activity before measuring. Rent hardware only when
   a concrete capacity or architecture gap requires it. V100 validates the distinct Volta fragment path and the
   existing serving recipes; it cannot validate native NVFP4 tensor-core execution. Full model capacity must be
   checked separately from individual-layer tests. Complete the repository's required finalization gates, document
   limitations, and mark the tracker ✅ with the ready PR only for the scope actually completed.

## Acceptance evidence

Stage deliverables are cumulative. The following sketches are intended results, not outputs from an implemented fix.

| Stage | Reviewable deliverable | Observable completion condition |
| --- | --- | --- |
| ✅ 1. Baseline | Pinned main + #969 reproduction matrix and small regression inputs | The singleton batch case demonstrably reaches the same recurrence algebra as the checkpoint; each known failure has a bounded reproducer. |
| ✅ 2. Recurrence | Field-preserving repack rewrite, consistent carrier/output coordinates, register and serial regressions | Kernel IR retains `FragmentRepack … role=b`; the checkpoint offers `STAGE=d1/reg`; classic launch-per-step CUDA does not reopen the time axis inside each launch; focused GPU states and corrected values match the reference. |
| ✅ 3. Output domains | One correct placement of computation and stores, with sibling-domain regression tests | Lowered IR has sibling `(a0,a1[,a6])` and `(a8,a10)/(a8,a11)` nests. Store counts are proportional to the sum of the output sizes. The real kernel set completes under the watchdog with correct outputs. |
| ✅ 4. Padding | Constant zero-fill padding through existing index maps | Tensor IR expresses `y[t,d] = x[t,d] if t < T else 0`; guarded Loop/Kernel loads are in bounds; short GDN traces succeed and returned sequence length remains T. |
| ✅ 5. Projections | Legal tensor-core routes for qkv and z, with measured cut/schedule alternatives | Tile IR offers activation-A / weight-B contractions with MMA TILE; emitted CUDA contains the expected MMA instructions; reference comparisons pass and measured latency is reported. |
| ✅ 6. Capture and state handoff | Static GDN capture and explicit state contract; native request dispatch remains the separate follow-up below | `prefill(x,S0,H0) → (y,S1,H1)` followed by `decode(x1,S1,H1) → (y1,S2,H2)` matches an independent reference; H is convolution history; reset and request isolation pass. This proves the block contract, not whole-model serving. |
| 🚧 7. Review | Validated PR(s), measurements, updated docs and tracker | Required finalization checks pass, scope and remaining gaps are explicit, and the tracker changes to ✅ only with the ready-for-review fix PR. |

- CPU tests preserve repack roles for both A/B and modern/Volta layouts; the checkpoint's actual loopification
  path renders correctly. Enumerate the recurrence's offered register schedules for the checkpoint shape.
- Loop/Tile/Kernel/CUDA dumps agree on the state update order, seed, output domains and batch coordinates.
  Independent sweep extents do not multiply one another; shared prefixes remain legal.
- GPU comparisons use identical inputs and independently computed outputs, including corrected values and final
  state. Start with existing recurrence tolerances (`rtol=3e-3`, `atol=1e-3`, relative norm error below `2e-3`) for
  the focused small-input tests; justify tolerances separately for real checkpoint data and accumulation modes.
- Fresh NVFP4 and bf16 layer tracing works at short and full-chunk lengths. Compile the formerly crashing 512-token
  recurrence and complete the 64-token projection kernel set within the existing watchdog, then record latency.
- Check checkpoint-reference binding before relying on `--strict`; the inline quantization bug is not proof that
  this reference is invalid. An unusable scalar reference is not a pass. Use focused eager references as needed.
- End-to-end serving validates prefill followed by decode, request reset and multi-request state isolation. Do not
  mark native GDN support complete from standalone chunk tests or from a mixed serving fallback.
- Run focused tests during development; at finalization run the required full suite, lint, golden/corpus checks,
  duration accounting and documentation review. Restamp only when required, retaining the distinction between
  measured evidence and schedules that need remeasurement.

## Evidence files

The excerpts above are the review evidence retained in this plan. The full local diagnostic files are under
`/home/io/cr/emmy/deltanet-investigation/` and are not part of this PR:

- `main-probe.txt`, `pr969-probe.txt`: fresh small reproductions and register matcher acceptance.
- `main-lift.loop.txt`, `pr969-lift.tile.txt`: checkpoint recurrence Loop IR and #969 carrying Fold.
- `main-trace.rewrite.txt`: actual checkpoint repack field loss, with the `100_loopify` call stack.
- `main-recurrence.kernel.txt`, `main-recurrence.err`: malformed final repacks and CUDA-render assertion.
- `pr969-recurrence.cuda.txt`, `pr969-pinned-recurrence.cuda.txt`: emitted scalar alternatives, not validated kernels.
- `pr969-projection.tile.txt`: qkv/convolution contraction and six independent output specifications.
- `main-projection.cuda.txt`, `pr969-projection.cuda.txt`: fresh runaway output nests.
- `probe.py`, `trace_lowering.py`: temporary diagnostic scripts; no compiler source changes.

## Implementation evidence (in progress)

- Recurrence: field-preserving repacks, singleton output-coordinate matching, serial-axis binding, and a late-bound
  register-schedule closure are fixed. The saved 512-token recurrence now emits CUDA with B-fragment conversion.
  On the isolated RTX 5090 checkout, the 11 focused recurrence GPU tests pass. On the local 5080 Laptop, numerical
  comparisons pass but the existing f32-accumulator 128-wide no-spill assertion reports 24 local bytes; the unchanged
  #969 checkout fails identically.
- Output domains: an unused root scalar kept otherwise independent operands inside the union of their axes.
  Removing that dead computation preserves live internal stores and lets the operand computations lower as siblings.
  Both output orders pass the structural regression; all 48 normalization tests pass. The real projection emits
  CUDA, but its end-to-end CLI run exceeded the 110-second process budget before producing a result. No latency or
  checkpoint numerical pass is claimed yet.
- Padding: constant zero fill uses guarded IndexMap sources, including safe inactive input coordinates. All 36
  focused backend checks pass, covering chunk lengths 1, 16, 63, 64 and 65 plus left padding and empty input.
  Five actual Transformers chunk-rule traces pass and retain the requested sequence length and final-state shape.
- Projections: fresh default CUDA contains qkv MMA and TMA kernels using current main's capabilities; numerical
  validation, z-projection coverage and measured alternatives remain pending.
- GitHub native stack #974 contains #969 then #973. The extra current-main commits appear in #973 while its base
  remains #969's older branch; they are not DeltaNet changes.

### Checkpoint-shape validation after preserving all outputs

Index-map composition was also deleting returned intermediate tensors. Keeping those producers restores all six
outputs of the saved projection frontend, which now lowers to the original `k_conv1d_linear_mean_reduce_c4b163`
identity. A new three-backend regression covers this output contract. The Torch reference now supports Conv1d,
with four CPU reference cases covering groups, bias, stride, padding and dilation.

- The six-output projection passes `emmy run --ir projection-reference.json --bench --strict` on the local 5080
  Laptop. This is the exact saved frontend with synthetic boundary tensors, not checkpoint-loaded weight numerics.
  A short 1-warmup/3-iteration run reports 881 µs Emmy vs 421 µs eager; this is smoke evidence, not a stable speed
  claim. The selected kernel set contains two qkv projection copies and is slower than eager.
- The original 512-token recurrence target executes on its full 48-head shape against independently written
  float64 matrix algebra over synthetic boundary tensors. Its offered float32-accumulator register schedule passes
  `rtol=3e-3, atol=1e-3`; relative norm errors are 0.000572 for states and 0.000524 for corrected values.
  The default float16-accumulator schedule has state relative norm error 0.000845, but 27/5,505,024 state values
  exceed that elementwise bound (maximum absolute error 0.001844). This is explicitly not a strict numerical pass
  for the default precision. The float32 schedule keeps the same recurrence algebra and passes the original bound.
- Seeded recurrence GPU coverage now includes the singleton external output-batch coordinate.

Actual fixed CUDA contains `emmy_c_to_b_f16` for the repack and legal `mma_m16n8k16_f16_f32` register recurrence.
The projection's emitted kernels include `mma_m16n8k16_f16_f16` with `d2/smem-tma` and `d2/smem-tma/p2`.
Its independent outputs are distributed into separate kernels; no six-axis Cartesian output loop is required.

Stages 2–4 are marked complete for the compiler contracts above. Stage 5 still needs stable measurement coverage.
Serving and final review are unfinished; the overall bugtracker row remains 🚧.

### Interleaved z-projection weights

The two existing z-projection cuts exposed a further correctness bug: one producer reads alternating weight
columns, but shared-memory staging copied contiguous bytes. The logical B operands are `B[k, 2*n]` and
`B[k, 2*n+1]`. A contiguous copy starting at either address silently supplies neighboring columns to MMA.
The fix uses the existing per-element operand fill when neither contraction dimension is contiguous.

The following is schematic lowering, not a verbatim dump:

```text
Tile IR:       even[k,n] = B[k,2*n]; odd[k,n] = B[k,2*n+1]
Before:        cp.async(shared_even[k,n:n+8], &B[k,2*n], 16 bytes)
After:         shared_even[k,n] = B[k,2*n]
               shared_odd[k,n]  = B[k,2*n+1]
               barrier; ldmatrix; mma.sync
```

Eight GPU regression cases pass on the local 5080 Laptop: one/two channels, 13/16 columns, and single/double
buffering. They also check that the generated B operand does not use the incorrect contiguous asynchronous copy.
Fifteen focused existing transport checks pass; the broader transport run exceeded the development time budget
and is not counted as a pass.

Both saved-shape z-projection producers now pass against float32 matrix multiplication on synthetic float16
inputs: A is 64×5120 and B is 5120×6144. The first producer returns the full projection; the second returns its
even and odd channels. Maximum absolute errors are 0.000993, 0.000992 and 0.000993, respectively, within the
original `rtol=3e-3, atol=1e-3` bounds. This validates the isolated producers, not the complete core/z consumer
or end-to-end serving. Stable latency measurements remain pending.

On September 30, the PR state check confirmed that #969 had merged (September 29, 16:07 UTC) and GitHub had
automatically retargeted #973 to main. The fix PR remains draft and the overall tracker remains 🚧.

### Main update after the parent merged

The branch now includes main `a5b8263c`, including #975's affine carried-state sequence split (`REDUCE=g<n>k`).
That change adds a probe/prefix/partitioned-walk route; it does not replace the fragment-role, singleton-coordinate,
output-domain, padding or strided-weight fixes here. Merge conflicts retained those fixes and main's new split.
The combined recurrence/split/weight-gather selection has 42 passing tests. Its first run exited unsuccessfully
only because two existing tests exceeded the duration inventory threshold without their worker-group suffix.
The required `-n 2 --dist=loadgroup --durations=0 --durations-min=0.5` run passes all 42 tests in 17.32 seconds,
including the duration gate. Existing slow tests already have grouped inventory entries; no newly added test in
this selection exceeds half a second.
The longer six-output projection measurement hit the 110-second development budget before reporting results,
so the earlier smoke measurement remains the only measured evidence for that complete projection.

### September 30: stateful capture and projection measurements

The local workstation is shared with another agent. Its timing results are diagnostic only and must not be used
for performance comparisons. Correctness tests continue locally. The remote 5090 is checked for other GPU jobs
before each measurement; our checkout and caches stay under `/root/deltanet-pr973`.

Pinned z-projection measurements on the idle 5090 use a synthetic 64×5120 activation and 5120×6144 weights,
10 warmups and 100 iterations through `emmy run --bench --strict`. Both schedules pass the strict check:

| Schedule | Emmy | Eager in the same run |
| --- | ---: | ---: |
| `mma_m16n8k16_f16_f32/f4x4/k4`, `w2x2`, `d2/smem-async` | 129.9 µs | 29 µs |
| Same TILE/WORK, `d2/smem-tma` | 67.4 µs | 27 µs |

These are isolated projection measurements, not checkpoint-weight or whole-layer performance. Neither reaches
eager parity. Profiling the complete six-output projection's timeout locates the cost in greedy cut/schedule
selection before GPU execution; pinning an individual projection resolves its compile in under one second.

Stateful GDN capture now exposes `(x, state, history) -> (y, next_state, next_history)` through the installed
Hugging Face forward. It clones history so decode cannot mutate the caller's input; zero states start/reset a
request. Negative constant padding, used to crop convolution history, now shares the guarded index-map lowering.
Twelve padding/cropping backend tests pass. Five CPU wrapper checks cover prefill lengths 1/16/65, subsequent
decode, reset, two-request isolation and both trace shapes. Two traced-IR evaluations also match the eager output
and both returned state tensors with nonzero input states. Mixed static GDN/full-attention capture passes, including
the full-attention projection's output gate. Symbolic GDN widths remain explicitly unsupported.

Full-block GPU validation found another output-domain defect. After a computation moves into an operand term,
an unused load could remain inside the output sweep. The writer then cannot extract output specifications:

```text
Before (actual residual structure, abbreviated):
  for a20: write add_10[a0,0,a20] = v477
  for a37:
    in224 = load history[a0,a37+4,0]  # no remaining statement reads in224
    for a39: write add_5[a0,0,a37,a39] = v534
  for a43: for a45: write copy_[a0,a43,a45] = v539

After lifting:
  pure computation belongs to closed operand terms
  output specifications retain only their three independent sweep paths
```

The lift now retains only effects in each sweep and exposes every scalar those effects read. Its small regression
fails on the prior implementation; all 17 operand-edge tests pass after the repair.

The misaligned-address failure came from vectorizing adjacent elements while checking only the last index.
For the three-tap convolution history, an odd row starts at an odd float address:

```text
Before (CUDA address, abbreviated):
  float2 pair = *(float2 *)&history[row * 3];  // invalid alignment for odd row
After:
  float first = history[row * 3];
  float second = history[row * 3 + 1];
```

Both load and store vectorization now prove alignment and adjacency from the complete flattened address.
Seven CPU checks cover odd/even strides, fixed rows, split coordinates and unknown layouts. The tiny one-token
block matches traced IR, Loop IR and CUDA with bound transformed weights: output error is zero, both state
errors are below 8e-9. A GPU regression checks three decode steps, seeded state, two independent batch rows,
fresh requests and reset; it passes under the grouped test runner.

Multi-token validation exposed two further bugs. A two-token prefill agrees with eager in traced and Loop IR. Its first CUDA
compile exposed a dropped time-coordinate binding when a carried fold becomes a register-program root. The
root now closes over that coordinate, preserving operand bindings. The next mismatch exposed lost seed strides:

```text
Seed tensor shape: [2, 1, 1, 64, 64]
Carried state after unit-axis removal: [batch, row, column]
Before (actual CUDA, renamed coordinates): seed[batch + row + column]
After: seed[batch * 4096 + row * 64 + column]
```

Classic and register schedules now restore unit coordinates against the bound seed tensor before lowering its
address. Four focused GPU checks pass, including both schedule families with nonzero seeds containing singleton
dimensions. The complete two-token block now passes: output maximum error 1.86e-9, recurrence-state error
7.45e-9, convolution-history error zero. Full-block GPU tests now pass both one-token and two-token prefill
followed by two decode steps, seeded requests, independent batch rows and reset. Native serving dispatch remains
pending; explicit-state block execution is not an end-to-end serving claim.

The 5090 isolated QKV projection (`64 × 5120` by `5120 × 10240`) now has direct strict-check evidence:

| Schedule | Emmy | Eager |
| --- | ---: | ---: |
| `mma_m16n8k16_f16_f32/f4x4/k4`, `w2x2`, `d2/smem-tma` | 70.2 µs | 72 µs |

The earlier paired run measured async at 209.8 µs, but its TMA comparison lacked the CLI's strict eager
correctness check; the direct TMA rerun above passed. These remain synthetic isolated projections, not
checkpoint or whole-layer results. Local 5080 runs are used for correctness because the workstation is shared.

### Finalization in progress

The 5090 passed five focused checks at `12b88c6b`: full-block one/two-token prefill followed by decode and reset,
both singleton-seed schedule families, and the time-binding regression (95.24 seconds including collection and
compilation). Its GPU is free again. Static ordinary and NVFP4 mixed-layer capture also pass; parameter identity
retargets the GDN wrapper's paths before checkpoint spelling. NVFP4 capture verifies actual packed weight constants.

A fresh main fetch remains at `a5b8263c`, already merged here. Final lint passes. The required full suite is running
locally with four workers; the shared 5080 is used only for correctness. The tracker remains 🚧 until finalization
is complete. Native request dispatch and whole-model qualification remain separate integration work.

Fresh model validation uses the report's cached revisions, not the saved frontend inventory alone. The existing CLI
traces `Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462` at 16 and 64 tokens, spelling four
quantized weights and four calibrated activation paths and writing eight distinct Loop kernels for each length.
The unquantized `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` also traces at 64 tokens in FP16;
the shared selected-layer/inventory APIs trace BF16 at 16 and 64 tokens (16 and 24 Loop kernels respectively).
A fresh 512-token NVFP4 trace also succeeds and writes 12 distinct Loop kernels.
These are fresh structural captures, not checkpoint-value GPU correctness or whole-model serving tests.

The first full-suite attempt found a stale installed Rust runtime and Triton's hard-coded `/sbin/ldconfig` path.
Both reproduced on unchanged main. An isolated runtime built from this branch and the supported
`TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` override fix all four focused environment checks. The suite restarted
with those settings. An existing source assertion also depended on the prior choosing a rolled scalar reduction;
its test now explicitly pins the scalar serial path with unrolling disabled. The focused assertion passes.

All 64 added CPU checks pass under the grouped runner, with slow cases entered into the duration inventory.
An older convolution test expected nonzero padding to fail; it now compares convolution plus empty/nonempty padding
against eager on all three backends (six passing cases). No unsupported-mode rejection was removed.

All 22 added CUDA checks also pass under the grouped runner with the matching runtime. The latest source revision
passes six focused checks on the 5090, including ordinary/NVFP4 stateful capture, one/two-token prefill handoff and
both seed-address schedules. That ungrouped invocation still exited unsuccessfully because the duration gate
looked for node IDs without the recorded `@cuda` suffix. The grouped rerun passes all seven checks in 89.82 seconds,
including the 128-wide FP32-accumulator no-spill assertion that fails on the 5080 baseline.

The local full suite also exposed the host's missing `/bin/bash` path in serving-image shell tests. All 55 tests in
that file pass with Bash supplied inside a temporary mount namespace; the shared workstation is unchanged.
Seven staging tests likewise pass when the namespace supplies Git on their fixed `/usr/bin:/bin` PATH.
The local full run completed with 8,085 passed, 390 skipped, 16 failed and seven setup errors in 1,518.16 seconds:
two obsolete assertions already repaired, the baseline 5080 spill assertion, thirteen shell-path failures and seven
Git-path setup errors. The existing FP8 expert check passed after 185.64 seconds of compilation/execution.
A separate invocation of that check on unchanged main also exceeded a 110-second diagnostic budget. These
shared-workstation runs do not establish a before/after compile-time comparison.

The final full suite is now running on the 5090 with eight workers, the current tracked source and a freshly built
Rust extension. Its first recipe-history check required adding Git history to the previously archive-only checkout;
that check now passes independently. No other agent's remote files or environments were changed.

Main advanced during finalization to `e702de5d` (#966, packed NVFP4 staging in cut matmul pieces). It is merged into
this branch at `af63a978`. The change re-forms computed-operand contractions inside their output grid so activation
remains A and packed weight becomes B, restoring async/TMA byte-slab staging. It complements the projection fixes;
it does not replace the recurrence, seed-stride, output-domain or padding repairs. All six packed-cut regression
checks and all 111 Qwen3.8 golden checks pass on the combined branch. The latter include row decoding and fresh
target lowering and completed in 105.11 seconds. The superseded remote full run was stopped and the final gate restarted on
this revision with Git history already present.

After the workstation restart, the 5090 SSH endpoint refused connections, so that remote run's final result is
not yet available. The local checkout, matching runtime and completed checks survived. The full combined-branch
suite restarted locally with four workers and temporary Bash/Git mounts; it collected 8,516 tests. Local runs
remain correctness checks, not performance measurements.

### Replacement 5090 and further main integration

The final local run on `71f76c09` completed with **8,125 passed, 392 skipped and one failed** in 1,527.67 seconds.
The sole failure is the existing 128-wide FP32-accumulator resource assertion: numerics pass, but ptxas reports
255 registers and 24 local bytes. An isolated checkout of unchanged `e702de5d` reproduces those exact numbers.
On the replacement 5090, all five capture, seed-address and no-spill checks pass in 19.69 seconds on `71f76c09`.
The two-case block handoff run completed one case before its 110-second limit; it is not counted as a passing run.
The remote environment is isolated under `/root/deltanet-pr973`, with CUDA 13.0, Python 3.12 and a matching runtime.

Main then advanced to `73a19d81`. Of its changes, #978 independently repairs singleton seed addressing and preserves
carried-kernel identity; #980 rejects incomplete buffer indices; #976 and #981 change normalization and keep reduction
subroutines compact. The merge reuses main's `seed_index` implementation and removes this branch's duplicate
`restore_unit_indices`. Tests combine main's singleton seed case with this branch's singleton output-batch case,
and retain the larger multi-batch, two-singleton seed regression. The remaining fragment-role, output-domain,
padding, vector-alignment and capture fixes are still part of this PR. The new runtime contract includes dependent
launches, so the native extension must be rebuilt before GPU validation on the merged revision.

### Native serving follow-up boundary

PR #973 supplies the static programs and validates their explicit state contract. Connecting them to native request
execution is a separate change because the existing native exporter accepts only unquantized dense Qwen3, and the
generation runner assumes every layer exposes full attention. The remaining deliverables are concrete:

1. Classify each layer when building the runner. GDN owns matrix state and convolution history; full attention owns
   its KV cache and retains its output gate. Derive each state shape and dtype from the captured program contract.
2. Allocate two state/history buffer sets per active request and GDN layer. A program reads the old set and writes
   the new set; swap only after completion. Reset both sets when a request slot is reused. This avoids aliasing an
   input that another output computation still reads.
3. Consume exactly the prompt's tokens. The current native attention path may compute a prefill chunk past the
   prompt end because later decoding overwrites those KV positions. GDN cannot reuse that policy: extra steps
   change its recurrent state. Use full static chunks followed by single-token steps for the tail initially.
4. Validate mixed-layer logits, prompt continuation and interleaved request isolation against the installed HF
   implementation at lengths `1`, `chunk-1`, `chunk`, `chunk+1` and `2*chunk+3`. Cover cancellation and slot reuse.
   Qualify packed checkpoint loading and memory use before claiming full NVFP4 model serving or its performance.

Expected execution-plan structure, **a draft, not implemented native dispatch**:

```text
request r, GDN layer l:
  zero S[r,l,0], S[r,l,1], H[r,l,0], H[r,l,1]       # request admission/reset
  GDN_chunk(x[0:C], S[r,l,0], H[r,l,0])
      -> y[0:C], S[r,l,1], H[r,l,1]
  GDN_decode(x[C:C+1], S[r,l,1], H[r,l,1])
      -> y[C:C+1], S[r,l,0], H[r,l,0]               # a real tail token
  # No launch for padded positions beyond the prompt.
```

The follow-up's review evidence should include the emitted buffer bindings, exact consumed-token counts, reference
logits, reset/isolation results and checkpoint memory accounting. The block-level checks in this PR establish the
program seam; they do not substitute for those runner-level checks.
