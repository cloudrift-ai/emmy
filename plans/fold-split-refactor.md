# One split for every tiled kernel: the block form of a Fold as the schedule's meaning

Written 2026-11-18 against main `45e539c6a`. Every stage is one PR from main, and every stage deletes something.

Line counts and the reading of today's code come from one pass over the tree in this session. They are starting
points. Read the code again before building on a number.

## What the first build showed (2026-11-18)

Built on this branch with the recurrence code removed, so that only the plain cases shape the design. Read this
before the rest: it changes stages 2 and 3.

**State of the branch.** A kernel carries no state. The roll, the lift of a carrying loop, the split across the
sequence, the serial launch axis and both storages of a state are deleted: 2,557 lines under `emmy/`, 10 files, and
the tests that covered them. The Rust statement kinds and Fold fields stay, and so does the `g<n>k<p>` spelling, so
stored files load. A program with a recurrence fails at the splice bound. The hardware goldens and the Qwen3.8
goldens still store kernels with a carried state and are not restamped. Restoring is a revert of the removal commit
plus whatever the new design asks of it.

**The gate.** The PTX of every kernel of all 258 corpus cases, taken before any change, with a scratch tool. It is
not a flag yet, and it does not cover the golden rows. Every step below compiles to that PTX byte for byte.

**Stage 1 holds.** `split_axis` and `merged` in `ir/tile/split.py` replace `_factor_k`, `_sliced_contraction`,
`_slice_fold`, `_sliced_edge` and `_state_fold`: 66 lines fewer, identical PTX, and the corpus regen writes nothing,
so no piece is re-keyed. The two slicers differed in the name of the block axis, the order of one sum and whether a
lift reads a clamped position. None of the three reaches the stored Loop IR or the PTX.

**Stage 2 does not hold as written.** The functions it names hold about 510 lines of code (the coop reduce, 145;
the K loops, 363). About a tenth of that decides which blocks exist: a lane axis, a start, a stride, a chunk count.
The rest is how a level runs: register copies with a masked tail, the shuffle and shared-memory combines, the
staging rings, the drains. The split says nothing about those, as "Two limits on replace" already states. Two more
facts:

- The coop reduce is an interleaved split, `k = coop * ks + lane`, with the in-block coordinate outside (case 2).
  The lowering spells it as one strided loop over `k`. The term lowers to a counted loop over `ks` with a multiply
  in every index, which is other PTX. Converting it back costs more lines than the six it removes.
- `_tiling.py` is already the levels of the free axes, held as counts: 75 lines of code.

So a lowering that reads levels from the term adds code and cannot keep the PTX. Not built.

**Stage 3 does not hold as written either.** A rule in `refusals.py` mostly says what the emitter can build or how
memory is laid out: which reduce the binder partitions, a projection that runs per fragment, an operand the
fragment loaders read contiguously, the chunk tier's shape, which free axes are m and n. A split of any axis
succeeds on all of them. Measured by switching one rule off and compiling what is newly offered:

- A census of where the rules fire, over the 296 kernels the corpus compiles to and their 231 contraction sites.
  No atom for the dtype on the card: 71 sites, a hardware rule. Structural rules: 35 sites. Of those, 9 have no
  pair of free axes for a fragment, which is the split's own answer. The other 26 are the emitter's: a projection
  that is no straight-line program per fragment (13), the chunk tier's operands (5), a root an output sweep rides
  (5), an operand the loaders do not read contiguously (2), a nested site on an operand edge (1). The store-address
  rule, the fragment-row rule and the scalar-tile rule fire nowhere.
- Over 67 one-kernel cases (every reduce case, a few fused, scan and pointwise ones) seven of the eight rules
  tried change no offer. The full row product of a contraction kernel (about 50,000 rows) was too slow to
  enumerate eight times, so the 26 sites above are not tested this way.
- One rule is over-cautious: a transposed band (`coop-t`) is refused when the projection has a loop. Without it 20
  cases gain 20 rows each, all 100 sampled rows build, and the four run on the RTX 5090 compute the right answer.
  That is a point the old reading never offered (stage 6). It needs timing before it is offered.

**What may still simplify.** None of it is built or measured.

- The last block is masked in six places, each with its own spelling: the split across CTAs, the tail of the
  register copies, the padded scan, the staged fill, the M and N overhang, the transposed band. One rule for "a
  width the extent is no multiple of" could replace them. It changes PTX, so its gate is a measurement.
- Stage 5 is untested by this build: the chunk loop of the twisted carrier is the one emitter that holds a whole
  second tiling of its own.

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

## The cases, in block form

Six cases, each as the term the compiler stores today and the term after the split. The "today" trees are the
structural dump (`emmy compile --ir tile`, or the lift alone for a carried state). The "after" trees are written by
hand in the same notation: no code produces them yet, and the reference model of stage 4 has to confirm the three
with a carried state before any of them is built.

Notation. `a2c in 0..2 ⊂ a2` is an axis cut from `a2`, as the dump prints a window today. The text after `·` says
where a level runs, which is what stage 2 adds to each level. A free axis has no Fold, so its split shows in the
`axes` lines above the tree and in the index of every load. A `lift` or `combine` the split leaves alone is not
repeated.

The sizes are chosen so that every level has more than one trip. The three carried states were traced at small
sizes and are written here at the real ones, with the step count rounded and the first rolled row moved (the roll
of the solve starts two rows in, and a walk whose last state nothing reads is one step shorter). The reading of each row's numbers (`w1x4` as one
warp along M and four along N, `k8` as eight atom steps a chunk) is the one "The idea" states; check it against the
codec before stage 2 builds on it.

### 1. A map

`y[a0, a1] = f(x[a0, a1])`. Free axes only, so step 1 is a substitution and the tree does not change.

```
today   place  free=(a0, a1)
        Fold  free  ‹pointwise›
        └─ lift: λ(a0, a1) -> (v0)

after   axes   a0 = 128 * a0c + a0t      a0c · CTA grid      a0t · thread
        place  free=(a0c, a0t, a1)  grid=(a0c, a1)
        Fold  free  ‹pointwise›
        └─ lift: λ(a0c, a0t, a1) -> (v0)                 every load reads x[128 * a0c + a0t, a1]
```

### 2. A reduction

`sum(x, dim=-1)` over 64 rows of 4096, under `WORK=t128 REDUCE=coop`.

```
today   place  free=(a0)  grid=(a0)
        Fold[a1 in 0..4096] reduce   ⟨REDUCE=coop⟩
        ├─ operand[in0]: load x[a0, a1]
        ├─ init: (0)
        ├─ lift: λ(a1, in0) -> (in0)
        └─ combine: λ(acc0, acc0__o) -> (acc0)
             acc0 = acc0 + acc0__o

after   axes   a1 = 128 * a1s + a1l
        place  free=(a0)  grid=(a0)
        Fold[a1l in 0..128 ⊂ a1] reduce                  · the CTA's threads, one lane each, merged across lanes
        ├─ operand[acc0]: Fold[a1s in 0..32 ⊂ a1] reduce · a serial loop in each thread
        │  ├─ operand[in0]: load x[a0, 128 * a1s + a1l]
        │  ├─ init: (0)
        │  ├─ lift: λ(a1s, in0) -> (in0)
        │  └─ combine: acc0 = acc0 + acc0__o
        ├─ init: (0)
        ├─ lift: λ(a1l, acc0) -> (acc0)
        └─ combine: acc0 = acc0 + acc0__o
```

Two things this case fixes for stage 1. The in-block coordinate is the outer Fold here, and the block coordinate
the inner one: a sum commutes, so the two levels swap, and the split has to return either order. And the outer
combine is the same ⊕ used as a merge (`Fold.merge`), so a twisted combine splits the same way with its recipe's
merge. `/v<n>` is a third cut of `a1`, the adjacent elements a lane reads in one step.

### 3. A product

`C[a0, a1] += A[a0, a2] * B[a2, a1]` with M = 64, N = 96, K = 256 in bf16, under
`WORK=w1x4 TILE=mma_m16n8k16_bf16_f32/f1x1/k8`.

```
today   place  free=(a0, a1)  grid=(a0, a1)
        Fold[a2 in 0..256] contraction
        ├─ operand[in0]: load x0[a0, a2]
        ├─ operand[in1]: load x1[a2, a1]
        ├─ init: (0)
        ├─ lift: λ(a2, in0, in1) -> (v0)
        │    v0 = in0 * in1
        └─ combine: λ(acc0, acc0__o) -> (acc0)
             acc0 = acc0 + acc0__o

after   axes   a0 = 16 * a0c + a0m                 a0c in 0..4 · CTA grid     a0m in 0..16 · atom rows
               a1 = 32 * a1c + 8 * a1w + a1n       a1c in 0..3 · CTA grid     a1w in 0..4 · warp
                                                   a1n in 0..8 · atom columns
               a2 = 128 * a2c + 16 * a2s + a2k
        place  free=(a0c, a1c, a1w, a0m, a1n)  grid=(a0c, a1c)
        Fold[a2c in 0..2 ⊂ a2] contraction               · a serial loop, one staged chunk a trip (k8)
        └─ operand[acc0]: Fold[a2s in 0..8 ⊂ a2] contraction    · a serial loop of atom steps
           └─ operand[acc0]: Fold[a2k in 0..16 ⊂ a2] contraction   · the leaf
              ├─ operand[in0]: load x0[16 * a0c + a0m, 128 * a2c + 16 * a2s + a2k]
              ├─ operand[in1]: load x1[128 * a2c + 16 * a2s + a2k, 32 * a1c + 8 * a1w + a1n]
              ├─ init: (0)
              ├─ lift: λ(a2k, in0, in1) -> (v0)
              │    v0 = in0 * in1
              └─ combine: acc0 = acc0 + acc0__o
```

The leaf and its two free axes `a0m`, `a1n` are a 16 x 8 x 16 product under a sum: the atom. That one test is
stage 3's legality. `f1x1` is a level one trip wide between the warp and the atom (a warp holds one fragment); a
wider `f` cuts `a0` or `a1` once more. A split across CTAs (`REDUCE=g<n>k`) is one more cut of `a2` above `a2c`,
whose level is another kernel: the pieces and the merge of stage 1.

### 4. A carried state read at its own cell: the delta rule

`S = S * exp(sum g) + Kᵀ V` per chunk of 64 tokens, the inter-chunk half of a delta rule. Each cell reads the
state at its own cell alone.

```
today   Fold  free
        ├─ operand[acc0, v4__obs]: Fold[a0 in 0..4] reduce        (carries)
        │  ├─ operand[acc1]: Fold[a2 in 0..64] reduce
        │  │  └─ operand[in0]: load g[a1, 64 * a0 + a2]
        │  ├─ operand[acc2]: Fold[a5 in 0..64] contraction
        │  │  ├─ operand[in1]: load k[a1, 64 * a0 + a5, a3]
        │  │  └─ operand[in2]: load v[a1, 64 * a0 + a5, a4]
        │  ├─ operand[v2]: Fold  free  ‹pointwise›
        │  │  └─ lift: v2 = pre acc0[a1, a3, a4]
        │  ├─ init: (0)
        │  ├─ cells: (a1, a3, a4)
        │  ├─ lift: λ(a0, acc1, acc2, v2) -> (v4, v4__obs)
        │  │    v0 = exp(acc1)
        │  │    v3 = v0 * v2
        │  │    v4 = acc2 + v3
        │  └─ combine: acc0 = next(acc0, acc0__o)
        └─ lift: λ(v4__obs) -> (v4__obs)
        outputs
        └─ add__steps0[a0, a1, a3, a4] = v4__obs

after   axes   a0 = 2 * a0b + a0i        a0j: a second in-block coordinate, the step a term came from
        Fold  free
        ├─ operand[acc0, v4__obs]: Fold[a0b in 0..2 ⊂ a0] reduce  (carries)   · ordered launches, one a block
        │  ├─ operand[acc1]: Fold[a2 in 0..64] reduce       · isolated: free over a0i, the block's steps at once
        │  ├─ operand[acc2]: Fold[a5 in 0..64] contraction  · isolated: free over a0i, the block's steps at once
        │  ├─ operand[v2]: Fold  free  ‹pointwise›          · the state the block started from
        │  │  └─ lift: v2 = pre acc0[a1, a3, a4]
        │  ├─ operand[w]: Fold[a0j in 0..2 ⊂ a0] reduce     · the cycle, broken: the decay from step a0j to a0i
        │  │  ├─ lift: exp(acc1 at a0j) when a0j <= a0i, 1 otherwise
        │  │  └─ combine: w = w * w__o
        │  ├─ operand[acc3]: Fold[a0j in 0..2 ⊂ a0] contraction   · the product for the writes
        │  │  └─ lift: (decay from a0j + 1 to a0i) * (acc2 at a0j) when a0j <= a0i, 0 otherwise
        │  ├─ init: (0)
        │  ├─ cells: (a1, a3, a4)
        │  ├─ lift: λ(a0b, w, acc3, v2) -> (v4, v4__obs)
        │  │    v3 = w * v2
        │  │    v4 = acc3 + v3                              the state the next block starts from is v4 at a0i = 1
        │  └─ combine: acc0 = next(acc0, acc0__o)
        └─ lift: λ(v4__obs) -> (v4__obs)
        outputs
        └─ add__steps0[2 * a0b + a0i, a1, a3, a4] = v4__obs
```

Step 3 finds that the gate and `Kᵀ V` read no state, so they leave the cycle and run for every step of the block
at once. Step 4 finds a cycle of one scalar per cell, `s ← w s + p`: its system is bidiagonal and its solution is
a weighted prefix sum, with no walk left inside the block. With one block covering every step the outer Fold
carries nothing.

### 5. A carried state read at earlier rows: the solve

Forward substitution, a row a step: `X[r, c] = A[r, c] + Σ_{m < r} A[r, m] * X[m, c]`. It is the intra-chunk
system of the gated delta rule (64 rows, 59 µs as one walk).

```
today   Fold  free
        ├─ operand[acc0, v10__obs]: Fold[a0 in 0..63] reduce      (carries)
        │  ├─ operand[acc1]: Fold[a4 in 0..64] reduce
        │  │  ├─ operand[v1]: pre acc0[a1, a0 + 1, a4]          row r of the seed: no step wrote it yet
        │  │  ├─ operand[v2]: pre acc0[a1, a4, a3]              row m < r: an earlier step wrote it
        │  │  └─ lift: v1 * v2 when a4 < a0 + 1, 0 otherwise
        │  ├─ operand[v5]: pre acc0[a1, a0 + 1, a3]             the cell's own seed
        │  ├─ operand[v8]: pre acc0[a1, a2, a3]                 a row this step does not write
        │  ├─ init: (select_scatter)
        │  ├─ cells: (a1, a2, a3)
        │  ├─ lift: acc1 + v5 when a2 == a0 + 1 and a3 < a0 + 1, v8 otherwise
        │  └─ combine: acc0 = next(acc0, acc0__o)
        └─ lift: λ(v10__obs) -> (v10__obs)

after   axes   a0 = 16 * a0b + a0i       a4 = 16 * a4b + a4i       a4j: the in-block row a term came from
        Fold  free
        ├─ operand[acc0, v10__obs]: Fold[a0b in 0..4 ⊂ a0] reduce (carries)   · ordered launches, one a block
        │  ├─ operand[rhs]: Fold  free                      · isolated: free over a0i, the block's rows at once
        │  │  ├─ operand[acc1]: Fold[a4b in 0..a0b ⊂ a4] contraction   · whole blocks, lifted from a4 < a0 + 1
        │  │  │  └─ operand[acc1]: Fold[a4i in 0..16 ⊂ a4] contraction · the leaf: 16 wide, a product
        │  │  │     ├─ operand[v1]: pre acc0[a1, 16 * a0b + a0i + 1, 16 * a4b + a4i]
        │  │  │     └─ operand[v2]: pre acc0[a1, 16 * a4b + a4i, a3]
        │  │  ├─ operand[v5]: pre acc0[a1, 16 * a0b + a0i + 1, a3]
        │  │  └─ lift: rhs = acc1 + v5
        │  ├─ operand[inv]: Fold[a0i in 0..16 ⊂ a0] reduce  (carries on, does not seed)
        │  │  └─ the small system: the walk above on the 16 x 16 block of A alone, its state in registers
        │  ├─ operand[acc2]: Fold[a4j in 0..16 ⊂ a4] contraction   · the product for the writes
        │  │  └─ lift: inv[a0i, a4j] * rhs[a4j, a3]
        │  ├─ init: (select_scatter)
        │  ├─ cells: (a1, a2, a3)
        │  ├─ lift: acc2 when a2 is a row of this block and a3 is left of it, v8 otherwise
        │  └─ combine: acc0 = next(acc0, acc0__o)
        └─ lift: λ(v10__obs) -> (v10__obs)
```

Step 2 does the work here: `a4 < a0 + 1` is settled by the block coordinates for every earlier block, so the sum
over earlier rows becomes whole blocks with an unmasked 16-wide leaf, and only the rows of the block itself stay
in the cycle. Step 3 sends the reads of row `r` to the seed. Step 4 leaves a unit lower triangular system 16
wide. Its walk is the one walk that stays, and it is where "levels of steps" is needed: a Fold under the carrying
one that carries on.

The dump spells the "today" conditions as nested selects on the index (`a2 < a0 + 3 && a0 + 2 <= a2 ? a3 : 0`).
The tree above writes them plainly. Reading them back to that plain form is part of the dependence test's input,
and the second measured fact of stage 4 applies to every one.

### 6. The Gated DeltaNet recurrence

Qwen3.8's own chunk rule, `S = g S + K̃ᵀ (V − W S)`: per chunk a contraction of the previous state (`W S`), the
correction `U = V − W S`, a second output of the step, and the write `K̃ᵀ U`. Every step writes every cell.

```
today   Fold  free
        ├─ operand[acc0, v6__obs, v9]: Fold[a0 in 0..4] reduce    (carries)
        │  ├─ operand[acc1]: Fold[a4 in 0..64] contraction        K̃ᵀ U
        │  │  ├─ operand[v2]: Fold  free                          U = V − W S
        │  │  │  ├─ operand[acc2]: Fold[a5 in 0..128] contraction W S
        │  │  │  │  ├─ operand[v0]: pre acc0[a1, a5, a3]
        │  │  │  │  └─ operand[in1]: load matmul_3[0, a1, a0, a4, a5]
        │  │  │  ├─ operand[in2]: load matmul_2[0, a1, a0, a4, a3]
        │  │  │  └─ lift: v2 = in2 - acc2
        │  │  ├─ operand[in3]: load mul_14[0, a1, a0, a4, a2]
        │  │  └─ lift: v3 = in3 * v2
        │  ├─ operand[v9]: Fold  free                             U again, for the output
        │  │  └─ operand[acc3]: Fold[a8 in 0..128] contraction
        │  ├─ operand[in0]: load unsqueeze_11[0, a1, a0, 0, 0]    the gate
        │  ├─ operand[v4]: pre acc0[a1, a2, a3]
        │  ├─ init: (0)
        │  ├─ cells: (a1, a2, a3)
        │  ├─ lift: v6 = acc1 + in0 * v4
        │  └─ combine: acc0 = next(acc0, acc0__o)
        └─ lift: λ(v6__obs, v9) -> (v6__obs, v9)
        outputs
        ├─ add_7__steps0[a0, 0, a1, a2, a3] = v6__obs
        └─ sweep(a6.a7) add_7__steps1[a0, 0, a1, a6, a7] = v9

after   axes   a0 = 4 * a0b + a0i        a0j, a4j: the step and the row of the block a term came from
        Fold  free
        ├─ operand[acc0, v6__obs, u]: Fold[a0b in 0..1 ⊂ a0] reduce   (carries)   · one block: nothing to launch in order
        │  ├─ operand[rhs]: Fold  free                      · free over (a0i, a4): 256 rows at once
        │  │  ├─ operand[acc2]: Fold[a5 in 0..128] contraction    · the product for the reads
        │  │  │  ├─ operand[v0]: pre acc0[a1, a5, a3]             the state the block started from
        │  │  │  └─ operand[in1]: load matmul_3[0, a1, 4 * a0b + a0i, a4, a5], times the gates before a0i
        │  │  └─ lift: rhs = in2 - acc2
        │  ├─ operand[sys]: Fold[a5 in 0..128] contraction  · the small system, free over (a0i, a4, a0j, a4j)
        │  │  ├─ operand[in1]: load matmul_3[0, a1, 4 * a0b + a0i, a4, a5]
        │  │  ├─ operand[in3]: load mul_14[0, a1, 4 * a0b + a0j, a4j, a5]
        │  │  └─ lift: in1 * in3 times the gates from a0j to a0i when a0j < a0i, 0 otherwise
        │  ├─ operand[u]: Fold[a0i in 0..4 ⊂ a0] reduce     (carries on, does not seed)   · the solve
        │  │  └─ lift: u = rhs − Σ_{a0j < a0i, a4j} sys * u       a product a step, 64 x (64 a0i) x 128
        │  ├─ operand[acc1]: Fold[a0j in 0..4 ⊂ a0] contraction   · the product for the writes
        │  │  └─ operand[acc1]: Fold[a4j in 0..64] contraction
        │  │     └─ lift: in3 * u times the gates after a0j when a0j <= a0i, 0 otherwise
        │  ├─ init: (0)
        │  ├─ cells: (a1, a2, a3)
        │  ├─ lift: v6 = acc1 + (gates up to a0i) * v4
        │  └─ combine: acc0 = next(acc0, acc0__o)
        └─ lift: λ(v6__obs, u) -> (v6__obs, u)
        outputs
        ├─ add_7__steps0[4 * a0b + a0i, 0, a1, a2, a3] = v6__obs
        └─ sweep(a6.a7) add_7__steps1[4 * a0b + a0i, 0, a1, a6, a7] = u
```

Step 3 isolates nothing but the loads: every cell is written every step, so every read of the state is in the
cycle. Step 4 does all of it. The unknowns are the corrections `U`, 256 rows of them; the state enters once, in
the product for the reads; and the system is strictly lower triangular in blocks of 64, because a chunk's
correction reads only the writes of the chunks before it. This is the chunk rule's own shape one level up: the
token rule split at 64 gives the rule traced here, and the same split at 4 chunks gives one block of 256 rows.

What it buys at 256 rows: four ordered steps of two products each become three products over 256 rows and a
solve of three steps. Every product has an unmasked leaf an atom can take, including the two the old reading
never offered (stage 6). What it costs: the system is a 256 x 256 product per head that the walk never forms, and
the per-step states the first output stores are four partial sums of the last product in place of four states the
walk had anyway. Whether that pays is the measurement stage 4 ends on.

### What the cases ask of the split

- Cases 1 to 3 need step 1 and nothing else, and case 3's leaf test is the contraction reading as it is.
- Either order of the two Folds a split returns (case 2), and a cut of one axis at several widths in one row
  (case 3).
- An extent that depends on a block coordinate (`a4b in 0..a0b`, case 5). The kernel's axis table holds a number
  or a symbolic count today.
- A second in-block coordinate over the same axis (`a0j`, `a4j`): the system and the product for the writes relate
  two steps of one block.
- A Fold that carries on and does not seed (cases 5 and 6): "levels of steps" in stage 4.
- Cases 4 and 6 at one block have an outer Fold that carries nothing. The split should return the term without
  it, so the kernel takes cuts and splits like any other.
- Cases 4 to 6 are each one term. Which operands become kernels of their own is the cut pass's question, not the
  rewrite's (stage 4).

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
- **A cut is not a split.** The split says which blocks exist. A cut says where a value is stored so that its
  readers share it, and no split of an axis answers that. The cut pass stays as it is. What must not grow is cut
  code that knows one kind of kernel.
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
- **The rewrite returns one term.** Steps 3 and 4 take the carrying Fold and give back one Fold tree: the product
  for the reads, the system, the solve and the product for the writes as operands of one root, as cases 4 to 6
  draw them. It mints no piece. The split across the sequence builds its own probe, prefix, fill-in and walk
  kernels; this rewrite must not repeat that.
- **The cut pass does the rest, unchanged.** The rewritten term's seams are offered like any kernel's, with no arm
  that knows a recurrence. Its pieces are ordinary products and each takes the split of stages 1 to 3. The uncut
  arm stays in the fork: for a state whose intermediates are per cell (case 4) it can win, and for one whose
  system many cells read (case 6) it recomputes the system per reader and is expected to lose.
- **It is a kernel-set decision, taken at the cut pass.** The cut pass runs before scheduling, on the unscheduled
  term. A block form that existed only under a schedule row would show the cut pass no seam. So this rewrite is
  unlike the plain split, which is a row's meaning and never stored: it is decided where the split across the
  sequence is decided today, as a new value of the carrying site's `REDUCE`, with the plain walk as the other arm.
  No stored row spells the new value, so none moves. The rewritten term is a kernel of its own with an identity
  of its own, and the kernel the fork is on keeps the one its golden rows name.
- **Running two levels.** One kernel's launches are one level of steps. Where a walk stays after the rewrite (the
  small solve, or more than one block), the work between two of its steps is another kernel, which the cut pass
  gives. Running both levels inside one kernel, or a symbolic number of outer steps, needs the runner to loop over
  a group of kernels. That is out of scope.
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
