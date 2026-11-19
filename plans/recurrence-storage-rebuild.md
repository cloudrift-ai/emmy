# Rebuild the storage of a carried state on the plain staging and atom code

Written 2026-11-18 on `feature/recurrence-storage-rebuild`, which branches from the refactor PR (#1443). The branch
deletes the shared-memory and the register storage of a carried state and is to rebuild them. This note holds what
the deleted code had to do, what was measured, and the design question that is open.

## State of the branch

- Deleted: the shared-memory storage pass (319 lines of code), the register storage emitter (257), the register
  schedule (151), and their hooks in the schedule pass, the kernel schedule, the codec, the site catalog, the graph
  codec, the materializer and the lift. About 1,190 lines under `emmy/`, with their tests.
- Kept: the roll, the lift of a carrying loop, the serial form (the state a buffer in global memory, one launch per
  step), the split across the sequence, and the `STAGE` key of a state, which now offers direct storage alone.
- Golden rows that spell `d1/smem` or `d1/reg` for a state do not realize. None is restamped.
- Corpus PTX is identical for all 258 cases: no corpus case keeps a state in shared memory or registers.

## What the two storages buy, measured on the dev-box RTX 5090

Two kernels of the 256-row Gated DeltaNet layer (Qwen3.8-27B-NVFP4-NVIDIA golden), run alone.

| kernel | recorded storage | with it | state in global memory, 10 rows tuned |
|---|---|---|---|
| the solve, 61 steps over a 64 x 64 state, 48 heads x 4 chunks | shared memory | 36.7 µs | 441 µs |
| the walk, 4 steps over a 128 x 128 state, 48 heads | registers, f16 atom | 24.9 µs | the prior's pick hangs |

So the rebuild has to get back a factor of ten on the solve. A launch per step costs about 7 µs here, not the half
microsecond the refactor plan assumed.

## What each storage had to do

**Shared memory.** One CTA owns each state whole, the batch axes are the grid.

1. Run the steps of the serial kernel inside one launch, a barrier after each.
2. Keep the state in two shared buffers the steps alternate between, so a step reads the state the step before
   left while it writes the next one.
3. Fill the first buffer from the seed, a tensor or a constant.
4. Store what the kernel outputs: a store that reads the step runs every step, a store that does not is the state
   the loop ends with and runs once behind the loop.
5. Drop the state's port from the graph node when nothing outside reads it.
6. For the solve's speed: copy the cells a step does not change with the whole CTA, and only the rows the step
   before changed; move a store into the branch of the select that feeds it, so the copying branch can go.

**Registers.** One warp owns sixteen rows and every column of the state.

1. Evaluate the whole step in C-layout fragments: a load, an elementwise operation and a select are fragment
   operations; a product repacks C fragments into A and B operands and runs the atom.
2. Keep the state in fragments across the steps, with the seed selected in at the first step.
3. Store every per-step output from its fragments, then overwrite the state's fragments last.
4. Offer it where the kernel is one state of static shape with matrix-shaped outputs, under the f16-accumulate
   policy or a named geometry.

## What reading the deleted code showed

- **Shared-memory storage was not a second staging path.** The step was lowered by the plain tiers, as it is for
  direct storage. The pass wrapped that kernel: items 1 to 5 above are about 100 lines, and item 6 is about 180
  lines of two rewrites that exist for the solve. One part was backwards: the serial form writes a select between
  the seed and the stored state, and the pass matched that select to take it out again.
- **Register storage was a second emitter.** It evaluates a term in fragments by itself. The chunk tier of the
  twisted carrier does the same job for its own programs (the evaluation of a straight-line program at the
  residence of each value, 115 lines, and the repack of a C fragment into an operand). Those two are the overlap.

## The design question

A rebuild from scratch of shared-memory storage arrives close to what was deleted, because that code already
reused the plain lowering. What would differ:

- The lift writes the state as a kernel-local buffer directly when the row asks for shared storage: no select to
  write and match back.
- The wrap is one function on the lowered kernel: declare, seed, loop, barrier, store behind the loop.
- The two rewrites of item 6 become ordinary kernel passes, added only when the solve's time asks for them.

For register storage the proper form is one evaluator of a term in fragments, used by the chunk tier and by a
resident state. That touches attention on every card.

Open: rebuild shared-memory storage in that cleaner form (a few hundred lines again, for about the same code), or
restore it and spend the effort on the fragment evaluator alone.
