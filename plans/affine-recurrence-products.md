# Composing an affine recurrence by matrix products

Status: phase 0 and the CPU half of phase 1 are built (2026-11-16); nothing has run on a card. Replaces step R2 of
`classic-schedule-kernel-gaps.md`, the blocked inverse as a recognizer of one shape.

## Where it stands (2026-11-16)

Built, and equal to the sequential walk on the CPU loop backend at every depth:

- **S0.** The wrapper's substitution is `S <- S + e_i (low_i S)`, with `low` read from its own tensor. Its step reads
  as affine and the split is offered on it. The first respelling tried (same slices, `row` from the input) is affine
  in arithmetic but its carrier reads keep conditional indices, so the reading refuses it.
- **S2 to S4** as one parameter on the existing split, not a second arm: `REDUCE@…/scan=g<n>k<p>` composes the
  parts' maps `p` times before the prefix. An odd map out passes through, so any part count composes down to one.
- **Stores.** A walk whose one store is its state, each step over the last, now ends at the prefix. Before, the
  parts of such a walk would have raced on that store on a card; no kernel reached it, because none was affine.
- **Two limits lifted on the way**, both of which the real rule hits: a size-one free loop outside the carrying one
  (the batch of one), and a probe that would not lift when the step stores a second value beside its state.

Not done:

- **No card has run any of it.** The GPU equality test is written and skipped here. S6 and S7 are open.
- **No realization case.** A case needs a card for its verdict.
- **The goldens.** The respelling re-keys every Gated DeltaNet program: restamp, then a record run per card.
- **The part count of the real rule.** At a chunk of 64 the substitution rolls to 62 steps and the 256-row chunk
  loop to 3, so the pins are `g62k<p>` and `g3k<p>`; the catalog's widths divide neither.

Goal: any carried state whose step is affine in the state can be compiled as products the schedule already tiles,
as one more arm of the fork the carrying kernel already has. Evidence decides whether the arm is picked. Nothing
recognizes a triangular solve, a delta rule or any other named algorithm.

## What exists

- A recurrence is a fold that carries a state. Its step is any computation, so it lowers as a serial walk.
- `Fold.affine` reads whether the step is affine in the state and column-wise: every carrier read keeps the cell's
  own coordinate at every position but one. Then the step is `S <- A·S + b`, with `A` square over the one mixed
  coordinate and shared by every kept coordinate.
- On such a kernel `030_cut` offers the split across the sequence (`REDUCE@…/scan=g<n>k`,
  `_split.realize_carry_split`): a probe walks each part from zero and from the identity and reads the part's map off
  the two walks, a prefix carries the state across the parts, and the walk runs every part from its start.
- All three pieces walk serially. A serial kernel reaches no tensor-core tier, so the split buys parallelism across
  parts and no products.
- The forward substitution offers nothing today. Its step multiplies two reads of the state (`row * sub`, both
  slices of the matrix being inverted in place), so it is not affine as traced.
- Tensor-core atoms take f16, bf16, FP8 and `e2m1` inputs. No atom takes f32 inputs.

## The design

An affine step is a map `(A, b)`. Maps compose: `(A2, b2) ∘ (A1, b1) = (A2·A1, A2·b1 + b2)`. Composition is
associative and has an identity, and it is a matrix product. So a recurrence of `n` affine steps can be halved by one
product kernel, as in cyclic reduction:

```text
A'[t] = A[2t+1] · A[2t]            b'[t] = A[2t+1] · b[2t] + b[2t+1]            t in 0 .. n/2
```

The result is again an affine recurrence, over `n/2` steps, with its maps as operands. The arm is that rewrite
applied a chosen number of times:

1. **Step maps.** The probe at a part length of one: every step applied to zero and to the identity, all steps at
   once. No serial axis, since both seeds are known. Output: `A[t]` and `b[t]` for every step.
2. **Pair products.** `L` kernels, each the product above over half the steps of the one before. Each is a plain
   contraction over a batch axis: it takes whatever `TILE` row evidence has for a product of that shape.
3. **The walk that is left.** `n / 2^L` steps of `S <- A[t]·S + b[t]`, a carrying fold whose step is a contraction
   against the carrier. This form lowers today. At `L = log2(n)` it is one step.
4. **Fill-in, only when a step's state is read.** A recurrence with an observer, or one whose every state is a
   stored output, needs the states the halving skipped: `S[2t+1] = A[2t]·S[2t] + b[2t]`, one more product kernel per
   level, coarsest first.

Where it lives: the same fork, the same site, one new value of `REDUCE` on the scan site that carries `L`. It is a
kernel-set decision, priced as the sum of its pieces' rows like the split beside it. Its spelling follows the
schedule codec's rules and is chosen in step 2. The split `g<n>k` stays: its probe is the part-length-`n/cta` case of
the same step-map kernel, and the two arms should share that kernel and the port plumbing rather than sit side by
side.

Not in this design, on purpose:

- No new Tile IR concept. A fold whose combine is a matrix product, tree-reduced inside one launch, is the form to
  consider only if measurement shows the launch boundaries are what is left.
- Nothing in normalization. The rewrite reassociates and changes what is rounded where, so the knob owns it.
- No fusion gate. The pieces are cut out of a maximal region by a pick, like every other cut.

## What to expect from the dense form

The dense map is honest about its cost, and the cost is high where the state is large. Estimates, not measurements,
for 48 heads at 256 rows (192 systems):

| Recurrence | State | Steps | Serial work | Dense composition |
|---|---|---|---|---|
| Substitution of the 64 x 64 system | 64 x 64 | 64 | ~17 M multiply-adds | ~3.2 G multiply-adds, 200 MB of step maps |
| Chunk loop of the state update | 128 x 128 | 4 | ~400 M | ~300 M, and every state is read |

So the dense arm is expected to be correct everywhere and to win almost nowhere on this model. The substitution's
step changes one row: its map is the identity plus a rank-one term, and the dense form throws that away. The wins
come with the factored maps of phase 2. Phase 1 is the mechanism those maps plug into, and its gate is correctness,
not speed. For that reason phase 1 offers the arm under a pin only: an enumerated arm that never wins costs every
tune a measurement and needs a refit to teach the prior to avoid it.

Baselines to beat, RTX 5090, 256 rows: the substitution 140 µs as traced, 31 µs in blocks of 16 with a tuned row;
FlashInfer's whole kernel 87 µs.

## Steps

### Phase 0: the substitution becomes affine

- **S0. Respell the substitution in the trace wrapper.** Row `i` of the state is written only at step `i`, so the
  `row` the step reads is the seed. Read it from the negated system, an input the loop never writes; only `sub`
  reads the state. The step is then affine and column-wise. Gate: `Fold.affine` returns a reading for the rolled
  substitution; the rule matches Transformers' in PyTorch; the existing `g<n>k` split is now offered on it. Bench
  that split with `--ab` on the dev box as a free data point.
- **S0 re-keys every Gated DeltaNet program.** It lands with the recurrence PR's own respelling in one restamp and
  one record run per card that holds such a golden.

### Phase 1: composition by products, dense maps, pin only

- **S1. Write the design into the Tile IR architecture doc and get it reviewed** before building: the map as the
  closed form of an affine step, the four pieces, what `g<n>k` and the new arm share.
- **S2. Step maps.** Let the carry split's probe run at a part length of one, where the walk piece disappears and
  the prefix is the whole recurrence over stored maps. Choose the knob spelling here. Gate: on the CPU loop backend
  the probe-and-prefix pair matches the sequential walk for the delta rule and the substitution.
- **S3. Pair products and the depth.** The halving kernel, applied `L` times, and the walk over what is left. Gate:
  final state equal to the walk's within the usual tolerance at every `L` from 1 to `log2(n)`; each product kernel
  is an ordinary contraction with a tile site, shown by pinning a `TILE` row on it.
- **S4. States that are read.** The fill-in kernels, so a recurrence with an observer or a stored state takes the
  arm. Gate: the chunk loop of the state update, every start state compared.
- **S5. Tests.** Next to the carry-split tests: the arm is offered under a pin and refused where the split is (a
  coefficient that varies with the kept cell, a walk under an outer loop, fewer kept than mixed cells); CPU equality
  at each depth; one GPU equality test; one realization case per piece shape. A pin on a kernel that offers no arm
  is reported, not ignored.
- **S6. Measure with pins, dev-box RTX 5090.** `emmy run --bench --ab` on three programs: the substitution, the
  chunk loop, and a long sequence with a small state (1,024 rows at a chunk of 16), the one shape where the dense
  form has a chance. Record nothing unless a row wins.
- **S7. Decide the dtype of the maps.** An f32 recurrence composes on thread tiles, since no atom takes f32. Rounding
  the maps to bf16 puts the products on tensor cores and loses bits at every level. Measure the error of the inverse
  against the f32 substitution at each depth. If it holds within the record run's check, the narrowed arm is offered
  under the fast-math knob; if not, say so in the comparison doc and leave tensor cores to phase 2, whose products
  are over the model's own bf16 operands.

### Phase 2: maps in factored form

Outline only; design it after S6 and S7 have numbers.

- A family of maps is usable when it is closed under composition and its composition is products. Dense is one.
  Diagonal is one and exists (`Affine.mixed` is `None`). The two that matter next: a scaled identity plus a low-rank
  term (a block of `p` rank-one steps is rank `p`: thin products only), and block triangular.
- The probe cannot see a family: it returns a dense matrix. Reading the family needs the step's coefficient of the
  carrier read as a term. That reading is the new part, and it belongs beside `Fold.affine`, computed where it is
  read and stored nowhere.
- The composition kernels of phase 1 stay; only the map's representation and its product rule change.
- Gate: the substitution alone at 64 x 64 under 31 µs for 256 rows, and the recurrence within 3x of FlashInfer as
  vLLM calls it, both from the R-steps this replaces.

### Phase 3: let the search have it

- Enumerate the arm in the fork, add its features to the placement featurizer (a version bump: datasets are
  re-exported), record the winning rows in the recipe golden and the hardware golden, refit both priors.
- Update the Tile IR and classic schedule architecture docs, the glossary entry for the carried state, and the
  verdict of the kernel comparison. Delete this plan.

## Faults that stand in the way

From the recurrence PR, each a compiler bug to raise on its own:

- A cut pinned on a serial kernel does nothing and is not reported. S5 covers the report for this arm.
- An inverse written as repeated squaring is rolled into a serial kernel whose compile takes 25 GB. Its step squares
  the state, so it is not affine and this plan does not reach it.
- A bare `PLACE=cut` on the respelled program gives wrong results. Record runs here pin each cut alone until it is
  fixed.
