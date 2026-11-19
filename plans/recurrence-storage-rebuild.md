# Rebuild the storage of a carried state on the plain staging and atom code

Written 2026-11-18 on `feature/recurrence-storage-rebuild`, which branches from the refactor PR (#1443). The branch
deletes the shared-memory and the register storage of a carried state and is to rebuild them. This note holds what
the deleted code had to do, what was measured, and the design question that is open.

## State of the branch

- Shared-memory storage is restored as it was. Reading it showed it is no second staging path.
- Register storage is rebuilt in one module, `lowering/kernel/_resident.py`, 248 lines of code where the old
  emitter and its schedule module had 408. Against the refactor PR the compiler is 189 lines smaller.
- What is shared now: a lift is evaluated by the chunk tier's own `_residence`, and a direct operand load by the
  plain tier's `_direct_operand`. What is not shared yet: the product (pack, load, atom, promote) is written here
  and again in the chunk tier.
- What changed in the design: the state is seeded ahead of the loop, not selected in at the first step. Equal
  products and equal memory reads are one, whatever their axes are called. A value that does not read the column
  axis is one block for every column. There is no second materialization type and no second row codec: the
  geometry is a field of the kernel schedule, and the materializer builds the carrying kernel's own term.
- The offer is on the state's shape alone (`TileOp.resident_state`). A step the fragments cannot evaluate declines
  the row where it is lowered.
- Not held yet: a ragged state (rows no multiple of sixteen, columns no multiple of the fragment's), which the
  old emitter masked; a select in the step. Neither appears in a recorded row.
- No test covers the new module. The old register tests are deleted with the old code.
- Corpus PTX is identical for all 258 cases: no corpus case keeps a state in registers.

## Measured on the dev-box RTX 5090, old and new back to back

| kernel | old | new |
|---|---|---|
| the walk of the 256-row layer, 4 steps, 128 x 128 | 26.8 µs | 27.1 µs |
| delta rule, 4 steps, 128 x 128 | 6.3 | 6.4 |
| delta rule, 8 steps, 64 x 64 | 4.8 | 4.8 |
| delta rule, 2 steps, 256 x 256 | 12.2 | 12.2 |
| delta rule, 16 steps, 128 x 128 | 21.5 | 22.4 |
| dense recurrence, 16 steps, 128 x 128 | 150.4 | 161.1 |

All six rows realize from their golden under strict evidence and pass the strict accuracy check. All thirteen
recorded register rows, on five cards, build; only the six above were run, on one card. The dense recurrence is 7%
slower and is not explained yet.

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

## What is left

- Find the 7% on the dense recurrence.
- Tests for the new module, and a corpus case that keeps a state in registers.
- One product for the chunk tier and this module. It touches attention on every card.
- The rows of the other four cards, on their cards.
