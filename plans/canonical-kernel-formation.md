# One kernel formation: a cut piece is the kernel its own program forms

Status: open, 2026-10-06, branch `fix/canonical-cut-pieces`. Blocks `emmy run --kernel` (PR #1076) for cut pieces.

## Problem

A kernel's identity should be what the compiler forms from its own program. Whole-kernel targets meet that; cut
pieces do not. The cut mints a piece as a tile, lowers it to a loop nest and lifts it again (`reformed` in
`passes/tile/_row.py`), and `loop/canonicalize` never sees it. Compiled from its stored body, the same piece passes
through `loop/canonicalize`, which fuses its split free axes (a 16 × 128 head split into one 2048 axis), and comes out
as another kernel. Measured over the 4090 and 5090 repository goldens: 167 of 671 cut pieces change, carrying 170
golden rows; no target changes. Every other Loop pass leaves the stored pieces alone.

Rows measured on a piece compiled alone (`--kernel`) therefore file under an identity the layer compile never looks
up, and a piece inside a layer may miss the tiers the canonical form unlocks (its space is larger: 106k rows against
47k on the 5090 q-projection piece).

## Rejected: send pieces back through the Loop passes

Making the engine re-run every Loop pass on a minted piece was scoped and does not hold:

- **Fusion undoes the cut.** `loop/fusion` is maximal: it merges a loop whose every reader sits in one other region,
  so a workspace producer read only by its consumer piece is merged straight back. Gating that breaks the maximal
  fusion invariant.
- **No re-entry exists.** The engine drives one cursor over the whole graph; passes before the cursor are never
  revisited, and no per-node pass state exists. Re-entry is a new engine mechanism.
- **Piece state has no home on a loop op.** `placement_decided`, `layout_decided`, `split_consumed`, the grid order and
  sweeps `reformed` sets, and the routing receipt read right after the splice all assume tile pieces.
- **Truncated pipelines** (`PLACEMENT_PASSES`, `TILE_LOWERING`, `LOWERING_PASSES`) carry no Loop passes, so pieces
  cut inside them would strand or skip them.

The distinction that matters: fusion is a graph-level pass (which kernels exist); canonicalization is a per-kernel
transform (the form one kernel takes). A piece's kernel set is already decided by the cut, so only the per-kernel
form should apply to it.

## Design: kernel formation is one function

Every kernel, a fused region or a cut piece, is formed by `lift_kernel` (the lift pass calls it for every loop op;
`reformed` and the sibling-producer fusion call it for pieces). Move the per-kernel Loop transform into it:

1. `lift_kernel` canonicalizes the free coordinates of the loop it lifts (the body → body core already extracted to
   `passes/loop/canonicalize/_free_axes.py`), after the loop op's normalization, before the lift.
2. Delete the `loop/canonicalize` pass: the lift now does it for every kernel, at the one place kernels form. Its
   rule docstring becomes the helper's.
3. `reformed` stops calling the helper itself; `_fuse_sibling_producers` already lifts through `lift_kernel`.
4. Give the formation the buffer shapes. The helper needs them for one fold (a flattened access recomposed through
   the buffer's row-major layout). The lift pass has them from the graph; `reformed` builds `LoopOp(body)` without
   them. Pass the piece's tensors (the parent tile's inputs and outputs, the workspace tensors the cut creates) into
   `reformed`'s loop op, or prove the fold never fires on a piece and drop the argument.

This survives pass changes by construction: a per-kernel transform belongs to formation, and formation is one
function every kernel goes through. A graph-level pass (fusion) stays a pass.

## Guard

Add a test that holds every repository golden kernel to the invariant directly: compiled from its own stored
program through the full pipeline, a kernel forms the identity the golden stores. Lowering is GPU-free, so it runs
in `make test` beside the fresh-lowering nodes, one node per golden file. A future pass that forms pieces
differently from their own programs turns it red, naming the kernels. `emmy run --kernel` relies on exactly this.

## Rollout

1. Implement steps 1–4; run the canonicalize, reform, cut, split and maximal-fusion tests, then the guard.
2. `emmy golden restamp` every repository golden. Re-keyed rows keep their schedule and lose their microseconds.
3. Re-record the re-keyed rows on their cards (the `refresh-golden` skill): the 5090 on the dev box, the 4090 on a
   rented card. Until then those pieces fall to the prior and the affected layers may slow down; the PR body names
   them.
4. Bump the tune DB schema version (identities moved), `make test-corpus-regen` if corpus cases go stale.
5. Rebase PR #1076 on this, re-test `--kernel` on a cut piece: its tuned rows must file under the layer's identity.

## Open questions

- Whether the canonical form is faster on the re-keyed pieces. One measured piece tied (10.8 µs both ways); the
  re-record answers it for all of them.
- Whether `reformed`'s own grid ordering (write order, unit row, peeled sweeps) also breaks the guard for some
  pieces. If it does, that logic moves into formation too, or the guard names it as the next gap.
