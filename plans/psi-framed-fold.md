# One vocabulary for a twist: the Fold stores what a recipe declares

Working note, drafted 2026-09-06 from a design discussion; nothing implemented yet. Branch from `main` as
`feature/psi-framed-fold`. Independent of `plans/symbolic-reading-and-twist-recipes.md`; the relation is stated
below. Neither file may be referenced from durable docs or code.

## Goal

Make attention's expectation channel read as a contraction off the STORED term, with no blocking rewrite, no
carved axes and no width. The bilinear product already exists in `SOFTMAX.lift` — `exp(s) * v`, the base monoid's
per-element contribution. The compiler throws it away: `Fold.twist` conjugates the pair into the stable program
and stores THAT, after which the product lives in the combine where no reading looks. Blocking is the round trip
back — it reads the product out of the combine and rebuilds a term to put it in.

Store the base instead, and keep ψ. Then the channel reads as a contraction directly, a human reads attention off
the dump, and the schedule decides everything blocking decided by form.

## Why now

The retired `schedule-space-inflation` note measured what blocking costs. On f16 SDPA `(1, 8, 512, 128)`, sm_120,
domains from the classic projection alone:

| | schedule space | node sites | contraction sites | `--ir tile` |
| --- | --- | --- | --- | --- |
| unblocked carrier | 6.5 × 10⁹ | 4 | 1 | seconds |
| blocked carrier | 2.3 × 10¹⁷ | 9 | 3 | 55 s |

Two of the three blocked contraction sites are the same value — α-equal copies of the block's `Q·K`, differing
only in which binder they read. Collapsing them took the space to 5.0 × 10¹⁴ and made a 4.4× better arm reachable
to the greedy at all: cold picks went 176 µs → 40 µs against a hand-swept best of 42 µs. That collapse was
reverted because a pinned row stopped realizing under pytest for a reason nobody understood.

Under this design the duplicate never exists: there is one score node because there is no second binder. The same
is true of the width. `block_width` is the largest power-of-two fraction of the extent within 64 — a form rule,
never measured — and it refuses a symbolic extent outright, so dynamic-length attention gets no tensor core today.

## Relation to the symbolic-reading plan

**No dependency. This does not need sympy.** An earlier draft sequenced behind that plan's five phases and was
wrong to. What step 4 needs is a scope change, not an algebra: channel 2's cone is `multiply(e, in7)`, a bare
product of two operand edges, which is exactly what `as_contraction` reads today. It declines only because it
requires the whole lift body to be nothing but products and the body also carries channel 0's score. Read the
channel's cone — `Lambda.cone`, which blocking already uses — and the existing syntactic test passes unchanged.

The rest is splicing. Building the base form binds `Recipe.lift` to the term's operands by renaming. The weight
cone is the semiring hoist formation already performs. The epilogue's `1/l` is `_hoist_invariant`, unchanged. ψ at
lowering is not an algebra question at all.

What the two notes share is a direction, and this one supplies an argument for the other. Today a recipe's `lift`,
`psi` and `psi_inv` are read by no code outside the recipe module — they exist for the certification tests, which
is that plan's own complaint. This work makes the definition half load-bearing.

Sequencing between them is therefore free, with one asymmetry: that plan's phases prove emission neutrality by
digest, and this one changes kernel identity for every attention kernel by design. Whichever lands second restamps.
Landing this one first is the cheaper order, because every digest A/B on an attention case currently carries the
nine-node tree and the 55 s compile, and the revert below removes both.

## The representation

### Ordinary Fold

Its algebra is

`F = (v, λ, e, ⊕, ω)`,

where `v` is the ordered operand list, `λ` lifts one set of element values into the state, `(B, ⊕, e)` is the
state monoid, and `ω` is an optional observer. The axis, its domain and the schedule sit outside this tuple because
none of them changes the algebra.

### Twisted monoids

A twisted monoid changes the state representation. Write the underlying algebra as `(B, ⊕_B, e_B)`, its
per-element lift as `λ_B`, and a bijection from the base state to the stable carrier state as `ψ : B → S`. The
inverse is `ψ⁻¹ : S → B`. These objects define the operations seen by an ordinary `Fold`:

```text
lift_ψ(x)       = ψ(λ_B(x))
e_ψ             = ψ(e_B)
combine_ψ(s, t) = ψ(ψ⁻¹(s) ⊕_B ψ⁻¹(t))
```

The last equation defines the combine mathematically. It does not produce a stable program mechanically. The
recipe supplies a numerically stable `combine_ψ` and certifies that it equals the conjugate.

A twist recipe packages the extension, and the resulting twisted `Fold` denotes an ordinary `Fold`:

```text
T   = (⊕_B, e_B, λ_B, ψ, ψ⁻¹, combine_ψ)
F_T = (v, T, ω)  ≡  (v, lift_ψ, e_ψ, combine_ψ, ω)
```

The matching section below uses `T` and these same definitions.

### Stored representation

Store the base and derive the stable operations. A twisted `Fold` and a `Recipe` use the vocabulary above. The
recipe is the schema; the fold is its instance bound to operands and an axis. Today `Recipe.lift` is `λ_B`, while
`Fold.lift` is `lift_ψ`. This change makes `Fold.lift` mean `λ_B` too.

| field | meaning | today |
| --- | --- | --- |
| `lift` | `λ_B`, the per-element contribution to the base state | `lift_ψ`, the stable-carrier lift |
| `init` | `e_B`; the stable-carrier identity `e_ψ` is derived | `e_ψ`, stored directly |
| `base` | `⊕_B`, one componentwise operation per state | absent; read back from `combine` |
| `twist` | the `Recipe` providing `ψ`, `ψ⁻¹` and stable `combine_ψ`, or `None` | absent; conjugation is baked in |
| `combine` | derived as `componentwise(base)` or the recipe's stable `combine_ψ` | stored, twelve statements for softmax |

`advance` and `rescale` stay on the recipe, authored and certified. Stability is not preserved by conjugation, so
the stable combine cannot be derived from ψ alone. The fold reaches it by naming the recipe instead of restating
the program. A planar fold is the identity case: `twist=None`, `lift_ψ=λ_B`, and `combine_ψ=⊕_B`.

### Matching

Matching reuses the same objects. A recipe states `λ_B`, `⊕_B`, `e_B`, `ψ`, `ψ⁻¹` and the certified stable
`combine_ψ`. `Fold.twist` binds candidate operands to the roles in `λ_B` and checks the dependent reductions
against the recipe's channels. On success it stores the instantiated `λ_B`, `⊕_B` and the recipe on one twisted
fold. On refusal it leaves the ordinary fold tree unchanged.

The input need not contain ψ or the stable combine. They state how the matched base reduction is represented and
how lowering must execute it. Matching therefore introduces no second notation for lift or combine after this
section.

## Target tree — SDPA

```
Fold  free                                                  -- epilogue: o = O * 1/D
└─ operand[m, D, O]: Fold[a2 in 0..128] reduce  ⟨twist=softmax⟩
   ├─ operand[acc0]: Fold[a3 in 0..32] contraction          -- Q·K                    <- mma site 1
   │  ├─ operand[in1]: load x0[0, a0, a1, a3]
   │  └─ operand[in2]: load x1[0, a0, a2, a3]
   ├─ operand[e]: Fold  free  ‹computed›                    -- the weight cone
   │     lift: λ(acc0, 0.176777) -> (e)
   │        v1 = multiply(acc0, 0.176777)
   │        e  = exp(v1)
   ├─ operand[in7]: load x2[0, a0, a2, a5]                  -- V
   ├─ init: (-1e+30, 0, 0)
   ├─ base: (maximum, add, add)
   └─ lift: λ(a2, acc0, 0.176777, e, in7) -> (v1, e, p)
        v1 = multiply(acc0, 0.176777)
        p  = multiply(e, in7)                               -- one monomial, two distinct edges
```

Channel readings, derived and available before any schedule exists:

```
channel 0   m   planar        maximum
channel 1   D   planar        add                    Sum e
channel 2   O   CONTRACTION   add / multiply         k=a2  left={a1}  right={a5}      <- mma site 2
```

Five computed nodes, one score node, one axis, no width. The pivot appears nowhere in the term: it lives in ψ,
which is where it always belonged.

## Target tree — Welford

```
Fold[a2 in 0..N] reduce  ⟨twist=welford⟩
├─ operand[in0]: load x[0, a1, a2]
├─ init:  (0, 0, 0, 0)
├─ base:  (add, add, add, add)                              -- carrier (S, n, T, W)
└─ lift:  λ(a2, in0) -> (in0, one, in0, sq)
     one = 1
     sq  = multiply(in0, in0)
```

Channel 3 folds `x·x` — one monomial, but its two factors are the same edge — so it reads as a planar add and
declines a contraction, the same verdict it reaches today for a reason now visible in the lift instead of buried
in a merge. ψ reads states 1 and 2 where softmax's reads state 0; neither takes a parameter the other lacks, so
the field serves both recipes rather than carrying a passenger for one.

## Steps

0. **Revert `0b781fabd` (#726).** Verified: it reverse-applies to
   `main` cleanly. It restores the four-node carrier, the 6.5 × 10⁹ space and a seconds-long `--ir tile`, and
   takes `035_block`, its two test files and the corpus stamps with it. It also gives back the FA-2 numbers and
   returns the SDPA mma tests to strict xfail — a stated regression that stands until step 5.
   *Verify:* `make test`, `make lint`, `make test-goldens`. Read any golden row recorded since #726 as suspect
   rather than as a row to re-record. The revert LEFT the blocked emitter itself standing, unreachable; step 4
   finishes the job.
1. **`Fold` speaks the recipe's vocabulary.** Add `base` and `twist`; derive `combine`. Planar folds take
   `twist=None` and `componentwise(base)`, which is what they already have. *Verify:* the suite unchanged, digest
   A/B byte-identical — this step alone still is neutral.
2. **The twisted rewrite emits base + ψ.** `020_twisted` stops baking the conjugate and builds the fold above: the
   recipe's own `lift`, instantiated on the term's operands. Emission changes here. *Verify:* the SDPA dump is the
   target tree; corpus restamped deliberately, with the identity change named.
3. **Per-channel contraction reading.** DONE. `as_contraction` reads one channel's cone rather than the whole lift,
   against the BASE monoid's per-state ⊕ — asking the stable `combine` would refuse every twisted carrier before a
   channel was looked at. `Fold.bilinear_channels` is the algebraic half; the geometry takes the first entry. The
   site stays keyed on the node: the channel is derived, not a second key. *Measured:* the three channel readings
   above; Welford's channel 3 declines, its product squaring one edge.
4. **One reading for "a tile folds this whole", and the old emitter is gone.** DONE. `Fold.tiles_whole` — every
   carried state is a bilinear channel, so the tile's accumulators ARE the carrier — is what decides a TILE site
   (`TileOp.contracts`), the node's schedule domain, its transport catalog, and whether a root has a chain. A
   twisted carrier answers no, because both tiers fold one accumulator per product channel and have no residence
   for a state that is not one; nor is the stored lift what a step may fold there, since for a twist it is the base
   contribution and denotes `Sum exp(s)`. Nothing may see through ψ.

   Refusing at the ENUMERATION rather than at the binder is the load-bearing half. Offered-but-unrealizable rows
   cost the greedy one blocklist retry per rank and there are more ranked rows than the retry budget, so a pinned
   P·V shape wedged with an unlowered `TileOp` instead of falling back. The old blocked emitter went with this
   step — see **What may not come back**.
5. **The carrier's tensor-core form.** NOT BUILT, and until it is, the site is NOT OFFERED. The design is stated
   below; it is a fresh emitter over the psi-framed term, not a revival. *Verify when built:* accuracy on a stream
   long enough that the base form would overflow; `mma.sync` count >= 4; #726's 3.3 µs at `(1,4,128,32)` and 150 µs
   at `(1,8,512,128)` met or beaten; the `attention/sdpa-*-mma`, `qwen3emb/sdpa-*` and `matmul/f16-symbolic-*`
   corpus cases close.
6. **Symbolic key extent.** Nothing sizes itself against the extent any more, so a dynamic stream should reach the
   same sites once step 5 lands. *Verify:* a symbolic-length trace compiles and still emits mma; the two symbolic
   corpus cases close.

## What may not come back

The tree carried a second emitter for attention until this branch removed it. Its shape: a carrier term holding one
operand per carried component, every one folding the same EXPLICIT block, with the block loop bound to the staged K
loop, plus the residence evaluator that interpreted a `Lambda` at fragment / row / uniform residence and the kernel
IR nodes only it produced.

Removed with it, and not to be restored: `scheduled_fold_contraction`, `_fold_staged`, `_carrier_values`,
`_projection_finalize`, `fold_store_tail` / `fold_store_sink`, the `_child_*` producer-seam builders, the
`kernel/_eval.py` residence evaluator, `FragmentRowReduce`, `FragmentSelect`, `FragmentLoad`, `frag_layout`, and
`staged_kloop`'s lead segment.

**The rule.** An emitter for this carrier takes its block from the SCHEDULE — the staged K chunk — and never from a
second reduce axis carved into the term. A design that reintroduces per-component operands, a block axis, a `Window`
carrying a block, or any width derived from an extent is reintroducing the thing that took the attention schedule
space from 6.5 × 10⁹ to 2.3 × 10¹⁷, made `--ir tile` take 55 s, and put three α-equal copies of one score in the
tree. Kernel identity may not turn on a form rule nothing measured.

## The carrier's tensor-core form — the shape to build

The term is one axis, one lift, one carrier. What an mma tier needs, it derives:

- the CHUNK is `STAGE`'s `bk_elems`, a schedule choice — there is no other block;
- the SCORE for the chunk is the cone of the pivot channel's result, which is the nested contraction the tree
  already carries as a site of its own;
- the chunk's PIVOT is that score's ⊕ over the chunk — a row reduce of the score fragments, one scalar per
  physical row;
- the WEIGHT is the recipe's own `Channel.pattern` instantiated on that pivot (`exp(s − g)`), which is why the
  pattern is load-bearing at lowering and not, as the open question below guessed, dead once matching moves;
- every non-bilinear channel folds its own cone over that weight, again as a row reduce;
- the bilinear channel stores the weight as the A slab and mma's it against its streamed operand;
- the chunk's partial merges into the carrier through the stored `combine` — the recipe's stable ⊕, applied once
  per chunk rather than once per element.

Two things this must NOT do. It must not read `lift` as the thing to fold — that is the base contribution, and the
weight above is what ψ makes of it. And it must not need the term to name the chunk: the same term must lower on
every `STAGE`, including none.

## What breaks

`Fold.lift` changes meaning, so every reader of it needs an audit — the one genuinely wide edit here.

Kernel identity changes for every attention kernel. Checked-in goldens go stale and need a tuning round on the
card; the corpus needs `make test-corpus-regen`. Neither is a reason to re-record a red row into green.

There is no FA-2 on this branch. Attention lowers at the scalar reduce tier, which is correct and slow.

The base form overflows. The refusal in step 4 is not a nicety: without it a wrong lowering is silent and only the
numerics move — and one already was. `Fold.roles` read an extra by its operand EDGE, which mistook the weight for
the streamed value whenever the score arrived on a slab of its own, and the carrier folded `exp(s)` where it meant
to fold `v`. Two disqualifiers are needed: a value this term also carries is the recipe's own derived channel, and
a value whose subtree computes the score is that definition too. Neither alone covers both a cut (where the weight
is a workspace slab) and the fixpoint's intermediate carrier (where it is no result yet).

Still red, and not from this design: `attention/rmsnorm-gqa-sdpa-stat-fill` and `attention/rmsnorm-qk-sdpa-stat-cut`
lose the SCORE contraction's own register tile — the accepted rows carry `TILE=''` where they carried `f1` — and
`reduce/cross-cta-flash-kernel` fails at `realized` under strict evidence. All three date from the base + ψ rewrite,
which put the weight cone on its own operand edge and changed these kernels' site set.

## Open questions

- **Idempotence of the rewrite.** `rewrite_twisted` runs to a fixpoint. Confirm it cannot revisit a fold this
  note produced and try to twist it again.
- **Which channel is the pivot.** Softmax's is state 0 by author ordering. Nothing here should re-derive that;
  confirm the fold can name it the way the recipe does.
- **Whether `020_twisted` still needs `Channel.pattern` to match.** ANSWERED, and the other way round: the pattern
  is not dead, it is what step 5's emitter builds the weight from. Matching and lowering read the same declaration.

## Acceptance

- The SDPA dump is the tree above: five computed nodes, one score node, two contraction sites.
- Schedule space on `(1, 8, 512, 128)` near 10¹⁰, not 10¹⁴ or 10¹⁷; `--ir tile` in seconds.
- Cold greedy on that shape at or under the 42 µs a hand sweep of 36 pinned rows found.
- Dynamic-length attention emits mma, which it cannot today.
- `git diff --stat main -- emmy/` net negative against `main` after the revert.
- No symbol from **What may not come back** exists in the tree.
