# DeltaNet investigation and proposed fix plan

Status: 🚧 Investigation complete enough to propose implementation. No compiler or serving changes made.
Implementation awaits the user's review of this plan. The bugtracker row is marked 🚧 on this branch.
Change it to ✅ with the fix PR link only when the agreed scope is ready for review; disclose any remaining support gap.

Stage status: 🚧 means in progress; ✅ means its deliverables have passed their checks. Pending stages have not started.
The investigation below is complete; stage 1 still needs the combined main + #969 baseline before implementation.

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
   the remote 5090 for repeatable deployment measurements and capacity where sufficient. Rent hardware only when
   a concrete capacity or architecture gap requires it. V100 validates the distinct Volta fragment path and the
   existing serving recipes; it cannot validate native NVFP4 tensor-core execution. Full model capacity must be
   checked separately from individual-layer tests. Complete the repository's required finalization gates, document
   limitations, and mark the tracker ✅ with the ready PR only for the scope actually completed.

## Acceptance evidence

Stage deliverables are cumulative. The following sketches are intended results, not outputs from an implemented fix.

| Stage | Reviewable deliverable | Observable completion condition |
| --- | --- | --- |
| 🚧 1. Baseline | Pinned main + #969 reproduction matrix and small regression inputs | The singleton batch case demonstrably reaches the same recurrence algebra as the checkpoint; each known failure has a bounded reproducer. |
| Pending: 2. Recurrence | Field-preserving repack rewrite, consistent carrier/output coordinates, register and serial regressions | Kernel IR retains `FragmentRepack … role=b`; the checkpoint offers `STAGE=d1/reg`; classic launch-per-step CUDA does not reopen the time axis inside each launch; focused GPU states and corrected values match the reference. |
| Pending: 3. Output domains | One correct placement of computation and stores, with sibling-domain regression tests | Lowered IR has sibling `(a0,a1[,a6])` and `(a8,a10)/(a8,a11)` nests. Store counts are proportional to the sum of the output sizes. The real kernel set completes under the watchdog with correct outputs. |
| Pending: 4. Padding | Constant zero-fill padding through existing index maps | Tensor IR expresses `y[t,d] = x[t,d] if t < T else 0`; guarded Loop/Kernel loads are in bounds; short GDN traces succeed and returned sequence length remains T. |
| Pending: 5. Projections | Legal tensor-core routes for qkv and z, with measured cut/schedule alternatives | Tile IR offers activation-A / weight-B contractions with MMA TILE; emitted CUDA contains the expected MMA instructions; reference comparisons pass and measured latency is reported. |
| Pending: 6. Serving | GDN capture and explicit persistent state, delivered separately if needed | `prefill(x,S0,H0) → (y,S1,H1)` followed by `decode(x1,S1,H1) → (y1,S2,H2)` matches an independent reference; H is convolution history; reset and request isolation pass. Mixed fallback is labelled separately. |
| Pending: 7. Review | Validated PR(s), measurements, updated docs and tracker | Required finalization checks pass, scope and remaining gaps are explicit, and the tracker changes to ✅ only with the ready-for-review fix PR. |

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
