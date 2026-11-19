# One split for every tiled kernel: the block form of a Fold as the schedule's meaning

Written 2026-11-18 against main `45e539c6a`. Every stage is one PR from main, and every stage deletes something.

Line counts and the reading of today's code come from one pass over the tree in this session. They are starting
points. Read the code again before building on a number.

## The idea

A Fold in block form takes four steps. They are the same four for a product, a reduction, a map and a recurrence.

1. **Split.** Cut an axis into a block coordinate and a coordinate inside the block: `k = 16 * kb + ki`.
2. **Lift.** A condition the block coordinates settle becomes whole blocks, taken or dropped. One they do not settle
   stays as a mask inside the block.
3. **Isolate.** For a carried state: a read no earlier step of the block wrote reads what the block started from.
   What is left is a dependence graph. A part outside every cycle runs for the whole block at once.
4. **Break.** A cycle that is linear in what it reads is a small linear system one block wide.

For a Fold with no carried state, step 1 is the whole algorithm:

```
C[i, j] += A[i, k] * B[k, j]          split i, j, k into blocks of 16, 8, 16
-> a Fold over the blocks, and under it a 16 x 8 x 16 product: one tensor-core atom
```

A schedule row already says this. `WORK=w1x4` and `TILE=mma_m16n8k16_bf16_f32/f1x1/k8` are the widths of the split
at every level: CTA, warp, register, atom, and the K chunk. Today nothing writes the split down. The lowering
interprets the numbers while it emits code, and a separate set of readings and refusal rules decides, shape by shape,
which numbers are legal.

The refactor makes the split a term, computed from the kernel and its row. Legality is then one question (does the
split succeed, and is its leaf the atom's product), and the lowering reads levels instead of building axes inside
atom code.

## Three rules for the work

- **On the term.** The split is an operation on a Fold, where the axes are explicit. Loop IR hides a kernel's
  conditions inside selects and index expressions, and a rewrite there first has to dig them out.
- **The plain case first.** A product, a reduction and a map need step 1 alone. A carried state needs all four and
  comes after the split exists.
- **In place.** Each stage puts the split where an existing mechanism is and removes that mechanism. A second path
  beside the first is not a stage.

## What exists today

| Job | Where | Size | After |
|---|---|---|---|
| Read a Fold as a contraction | `readings.rs` (`find_contraction`, `tiles_whole`, `chunked`), `TileOp.contracts`, `_derive_mn` | Rust, plus about 150 lines | stays as the leaf test; the shape cases go |
| Structural refusals | `classic/refusals.py`, parts of `staging.py` | about 470 lines | replaced by "the split succeeds" |
| Hardware and policy rules | `classic/refusals.py`, `staging.py`, `context.py` | about 1,300 lines | stay |
| Agreement between sites | `_relation_refusal`, `derive_workers`, `_kernel_composes` | about 340 lines | stay; may shrink to "one axis, one split" |
| Catalog and codecs | `catalog.py`, `choices.py`, `classic/codec.py` | about 1,650 lines | stay, spellings unchanged |
| M/N tiling | `lowering/kernel/_tiling.py` | about 180 lines | reads levels |
| K loops | `_contract_kloop`, `_staged`, `pipelined_kloop` | about 300 lines | one level |
| Coop reduce | `_tile_reduce_axis`, `_tile_reduce_axis_transposed`, `_strided_fold` | about 200 lines | one level |
| Split across CTAs | `_split.realize_split`, `split_forks`, `_factor_k` and its slicers | about 470 lines | a split whose outer level is another kernel |
| Split across the sequence | `_split.realize_carry_split` | about 320 lines | step 4, if it gives the same kernel sets |
| Flash chunk loop | `_FlashOps`, `_residence`, mask helpers | about 1,260 lines | a split of the key axis; decide after stage 3 |
| Atom emission | `_ScalarOps`, `_MmaOps`, drains, wgmma | about 850 lines | stays |
| Staging and transport | `_stage.py` | about 1,850 lines | stays |
| Packed and coded operands | `_packed_operands`, `_block_scaled_operands`, `packing.py` | about 700 lines | stay |
| Carried-state storage | `_register.py`, `035_shared_carry` | about 670 lines | stays |

Two limits on "replace". The split decides which blocks exist. It does not turn a block into instructions, and it
knows nothing of shared memory, registers or precision. And the test that a leaf is a product under a distributive
sum is the contraction reading itself: it stays, as the one place that asks.

An honest size: stages 1 to 4 remove roughly 1,200 to 1,500 lines and add a few hundred. Stage 5 could
remove about a thousand more and is the riskiest.

## Constraints that shape the design

- **A kernel's identity is its unscheduled body.** It digests the Loop IR the Fold lowers to, without the schedule.
  So the split term is never stored on the kernel. It is computed from the kernel and its row where it is read, which
  is also what "kernel facts are computed, never stored" asks.
- **Spellings do not change.** Golden rows (about 6,200), candidate rows (2,700), the freeze (about 34,000 rows) and
  258 corpus cases spell `TILE`, `WORK`, `REDUCE` and `STAGE`. A restamp can move a route by one segment; nothing
  translates a value. A changed spelling demotes a row to a proposal, and a proposal needs its card to measure it
  again, on nine cards. The row's numbers already are the split widths, so no spelling has to move. New choices get
  new values and touch no stored row.
- **Pieces of a kernel-set decision are kernels with identities.** A split across CTAs mints pieces whose bodies the
  goldens store. Stage 1 has to produce the same Loop IR for them, to the statement, or those rows are re-keyed.
- **Fusion stays maximal, and normalization carries no knob.** The split is a schedule's meaning, never a gate on a
  merge and never a canonical form.

## The gate for every stage

A stage that changes no decision must change no code. Before stage 1, compile every golden row and every corpus case
on main and keep each kernel's PTX. After each stage the same compile gives the same PTX, file for file. This needs no
GPU and it is stronger than any test of numerics.

- `emmy golden check` stays clean, and `emmy golden realize` realizes every row it realized before.
- `make test-corpus-regen` has nothing to write.
- The PTX comparison is empty. `emmy compare` diffs the numbers of two dump directories today, not their PTX, and a
  dump is keyed by kernel, not by row. Holding one PTX per row and comparing two trees is most likely a flag to add
  in stage 0. It is not a script to write.
- `make test` and `make lint`, once, at the end of the stage's PR.

Where a stage means to change code (stages 4 to 6), the gate is a measurement on the card, through
`emmy run --golden ... --bench`, and the PR names every kernel that got slower.

## Stages

Each stage is one PR from main. Each ends with a deletion. Do not start a stage before the one before it has merged,
except stage 4, which needs only stage 1.

### 0. Baseline

- Take the PTX baseline above, adding the flag it needs.

Done when: the baseline exists and a second run on main reproduces it byte for byte.

### 1. The split as an operation on the term

- One function on a Fold: split an axis at a width. It returns the Fold over the block coordinate with the Fold over
  the in-block coordinate under it, the index substitution applied to every slab, and the two axes for the kernel's
  table with their parent recorded.
- A free axis has no Fold of its own, so its split is the substitution and the two axes alone.
- An extent the width does not divide gives a last block with a mask. This is step 2 in its smallest form.
- Its first caller is the split across CTAs, which does this today with its own slicers.

Removes: `_factor_k`, `_sliced_contraction`, `_slice_fold`, `_sliced_edge`.

Done when: the gate holds, and for every corpus kernel a split at each catalog width lowers to Loop IR that computes
what the unsplit kernel computes on the CPU runner.

Risk: the pieces' Loop IR differs by a statement. Then the pieces are re-keyed. Stop, and make the split spell what
the slicers spelled, before touching anything else.

### 2. The scheduled term, and a lowering that reads it

- One function: a kernel and its row give the split term. Each level says where it runs: CTA grid, warp or thread,
  register, atom lanes, a serial loop.
- `_tiling` lowers a level. The K loop and the coop reduce are levels like the others, so their private axis
  building goes.
- Atom emitters receive a leaf and its operand blocks. They stop receiving callbacks that build axes.
- A kernel with a serial launch axis is lifted again when it is materialized, so the Fold that was scheduled is not
  the one that is lowered. The scheduled term is built from the form that is lowered, or that second lift goes.

Removes: the callback structure of `grid_tile`, `_tile_reduce_axis`, `_tile_reduce_axis_transposed`,
`_strided_fold`, and the K-loop duplicates.

Done when: the gate holds.

### 3. Legality from the split

- A site offers a catalog point when the split at that point succeeds, its leaf is the atom's product, and the
  hardware rules pass.
- Run the old and the new offering side by side over every site of every golden and corpus kernel. The new set must
  hold the old one. List what is new. A point the old set has and the new one lacks is a bug.
- Then delete the structural refusals and the shape cases of the contraction reading.

Removes: about 470 lines of refusals, `TileOp.contracts`, the special cases of `_derive_mn`.

Done when: the gate holds, every golden row still realizes, and the PR lists the newly offered points.

### 4. Carried states: isolate and break

Needs stage 1 only. This is where the time is: the Gated DeltaNet recurrence at 256 rows runs at about 170 µs against
FlashInfer's 87. Its parts, in the order to build them:

- **A reference model.** The four steps in numpy, on statements of the form `target += product of reads` under
  conditions, a few hundred lines under `tests/`. Check it against the serial walk on a product, a sum, a map, a
  prefix sum, a triangular solve, the delta rule, a dense recurrence and a max-plus fold, at several block widths. It
  settles what each step must find before any of it is in the compiler, and it stays as the reference for step 4.
- **A dependence test.** "Can this read meet a cell an earlier step wrote" is a few integer comparisons that must all
  hold: the two cells are one, the step is earlier, each variable is inside its loop. Answer it by Fourier-Motzkin
  elimination over the comparisons' affine forms, with `a < b` taken as `a + 1 <= b`. It reads no extent, so it costs
  the same for 64 rows and for a symbolic count, and it is exact when it says no. A test that is no comparison places
  no constraint.
- **Levels of steps.** A split of the step axis gives two nested loops of steps over one state. The IR has to hold
  that: a loop under the carrying loop that does not seed is a further level of the same steps, and in the term the
  inner level is a Fold of its own that carries on and does not seed. A statement between the levels runs once per
  outer step, and a second state carried by the outer level holds what one block computes for the next. The flag for
  a loop that does not seed is already in the statement; who counts as carrying the state, the Loop IR rendering, the
  lift and the lowering have to learn it. Declare the term's new field last with its default left out, so no stored
  form and no identity moves.
- **Running two levels.** One kernel's launches are one level of steps, so the cut pass takes an outer level apart:
  per outer step a kernel that carries the state through the inner steps, and between two of them a kernel that
  carries nothing. Running both levels inside one kernel, or a symbolic number of outer steps, needs the runner to
  loop over a group of kernels. That is out of scope.
- **The split at the carrying site.** Offered as a new value of that site's `REDUCE`. No stored row spells it, so
  none moves.
- **Step 3 on the term.** Which reads see what the block started from. The dependence test answers it.
- **Step 4 on the term.** A linear cycle becomes the product for the reads, the small system, its solve, and the
  product for the writes. The delta rule and the solve are the two cases to pass first.
- If step 4 gives the kernel sets the split across the sequence gives, that split is a case of it.

Three facts to design around, each measured on an RTX 5090:

- A step of a carried walk costs about half a microsecond whatever its state holds. Steps 1 to 3 alone leave the
  number of steps as it was, so they do not pay: the 64-row solve takes 59 µs as one walk, and no cheaper in blocks
  while each block is still walked a step at a time. The split of a carried state is offered together with step 4,
  never without it.
- A rewrite that moves a loop variable adds the same amount to both sides of every test of it. The normal form does
  not cancel that: a walk whose tests read `a + 16 < b + 17` ran 2.5 times slower than one that read `a < b + 1`.
  Spell every comparison again after a substitution, common terms cancelled and the constant on one side.
- A width the walk's length is no multiple of leaves steps in the last block that write nothing. Drop them when the
  block is a number; each is a launch.

Removes: `realize_carry_split` and its probe, prefix and composition kernels, if the last point of the list holds.
Otherwise nothing, and the PR says why.

Done when: the recurrence is measured on the RTX 5090 with step 4 against the recorded row, and the solve against
59 µs. A result that is not faster is reported and not merged as an offered choice.

### 5. Flash as a split

The twisted carrier's chunk loop is a split of the key axis whose leaf is the score product and whose combine is the
recipe's. It is the largest per-shape emitter, and attention is recorded on every card.

Decide after stage 3 whether to do it. The PTX gate is unlikely to hold, so it needs `make bench-kernels` on each
card class and a record run where a row moves.

Removes: up to about 1,000 lines of `_FlashOps` and `_residence`.

### 6. What the old reading never offered

With legality from the split, a Fold the contraction reading refused can be offered an atom. Two known ones are in the
recurrence: the readout's three scalar products (60 µs) and the copy that stacks the chunk start states (33 µs).

- Measure each on the card, record the winners into the hardware golden, refit both priors in the same PR.

## Decisions to take before starting

1. **Order.** Foundation first (stages 1 to 3, then 4), or the recurrence first (1, then 4, then 2 and 3). The second
   reaches the speed sooner and leaves two tiling mechanisms side by side for longer.
2. **How far.** Whether stage 5 is in scope at all.
3. **The gate.** Whether identical PTX is the bar for stages 1 to 3. It makes the stages slow to land and safe. A
   weaker bar, the corpus and `make bench-kernels` on one card, is faster and can hide a regression on another card.
