# A value-numbered normal form: recognize shared computation, derive once per value

Status: design, 2026-10-08. Branch `feature/compact-lowering` holds a superseded prototype (alpha classes in the term
lowering); the recognizer this plan builds on is measured but not yet in the tree.

## The problem

Fusion inlines a producer under every consumer, specialized to that consumer's coordinates and to the components it
reads. The fused Qwen3.8 GDN decode kernel is 1504 Loop IR statements, but only 266 distinct values: the attention
output cone appears three times, the post-attention norm statistic is recomputed inside the MLP's 17408-wide loop,
and so on. Every expensive step after fusion — the normalization passes, the canonical order, the lift, identity,
features — is a whole-body function, so a kernel costs its inlined size, and the cut pass pays that cost once per
arm it ranks (537 arm builds in one compile) because an arm's pieces are lowered, normalized and lifted from scratch.

Reuse by recognizing copies in the normal form fails today because the form specializes before it canonicalizes: names
are numbered through the whole body, sibling order comes from a whole-body labeling, loop-invariant hoisting and load
dedup cross a copy's boundary, and a narrowed copy has fewer statements. Three copies of one function become three
texts no hash matches.

## The recognizer

Value numbering with coordinates abstracted (prototype: 12 ms for 1504 statements):

- A statement's number is its kind and payload (op, dtype, buffer) over its operands' numbers.
- Every maximal coordinate-only index expression is a parameter, numbered by first appearance across the statement
  and its operands; a bare loop variable is the simplest case. A statement is therefore a function of coordinate
  EXPRESSIONS, so `W[k, (h / 384) * 128 + d + 2048]` in the parent and `W[k, g]` in a piece whose grid axis `g`
  took that index's place number alike.
- A reduce binds the parameters that mention its axis; the accumulator's number records which positions it bound.
- Operands compose by parameter identity, so `A[i, j] * B[j, i]` and `A[i, j] * B[i, j]` differ by their maps.

Measured on the fused kernel: 266 distinct values; of 139 repeated values every non-leaf value's occurrences are
alpha-identical, and the 13 that differ are loads whose index expressions differ — one function applied at different
coordinates, which is the intended merge. Across the first cut arms the pieces of output-side seams keep 88–97% of
their statements' numbers from the parent; an input-side seam renumbers about half, as every dependent must.

## Design

The term IR stays as it is: an inlined DAG the cut, the schedule and the lowering already understand. The compact form
lives in the Loop IR normal form only, and leaves it at one boundary.

1. **`ir/stmt/values.py`** — the recognizer as a pure function over a `Body`: statement → (number, parameters), plus
   the application table (number → its occurrences and their argument expressions). Computed where read, never
   stored (the kernel-facts invariant); no knob.
2. **Outlining in normalization** — `normalize_body` numbers the raw lowered body FIRST, before any context-dependent
   pass. Every non-leaf value with two or more applications becomes a `Subroutine` (parameters = the lambda's
   coordinates, body = its cone) and each application a `Call` with the coordinate expressions as arguments; the
   fusion splicer's own `Subroutine`/`Call` pairs are the same thing and stay. Normalization then runs on the compact
   body: definitions once each, the root over calls. A call is a statement with its arguments' coordinates as deps,
   so hoisting lifts a call out of a loop it does not read, and dedup merges two calls with equal target and arguments
   (CSE of applications). Canonical order and identity run on the compact body, a call colored by its target's
   canonical digest. This is a new canonical form: every golden kernel re-keys.
3. **Expansion at the kernel-lowering boundary** — `inline_calls` (exact: the definition's own names renamed first,
   then the arguments substituted) followed by today's normalization gives the executable form. Measured on the
   fused kernel: expand-then-normalize of the compact form equals today's normal form bit for bit; the gate below
   extends that to the corpus.
4. **The lift expands first** — `lift_kernel` lifts the expanded body, so the term stays today's DAG and no tile pass
   meets a call. Lift cost stays proportional to the inlined size until a later step memoizes it per definition.
5. **Per-compile value table** — definitions are hash-consed by number in a table on the `Context`, like the session
   kernel cache: the normalized definition body, its canonical digest, its stamps. The 46 arms of a fork share most
   definitions, so their normalization, identity and features hit the table; an arm re-derives only what its seam
   reaches. A cache of computed values, so it carries a version.
6. **Features** keep reading the expanded raw body (`stamps` do today), so both priors stay valid; features on the
   compact form and a refit are a later experiment.
7. **Goldens and the corpus cure themselves.** A golden stores no identity: the import lifts each stored body to a
   term and normalizes again, so a stored old-form body and the fresh kernel compute to the same new identity and
   every row keeps its microseconds. Only the fresh-lowering test goes red, and `emmy golden restamp` rewrites the
   bodies into the compact form without re-keying a row; `make test-corpus-regen` does the same for the corpus. Seams
   and schedule spellings address sites in the term, which the design leaves untouched, so pins and routing rows
   stay valid. Two things must exist first: the Loop IR wire format carries definitions and calls (today a call never
   survives to a stored body), and the tune DB's kernel-key version bumps so the old table is rebuilt from the goldens.

## Is this the principled form? Two stages

The design above is a cache bolted onto today's normal form: it outlines what the recognizer finds and keeps the pass
pipeline, the whole-body canonical labeling and the global sequential names. Its virtue is that the executable form
does not move, so it is measurable and reversible. It is not the principled form.

The principled form takes the recognizer's output as the definition: a body IS its hash-consed DAG of values (lambdas
over coordinates) plus the stores that apply them, and the Loop IR text is one deterministic rendering of that DAG.
Rendering is three rules the compiler already owns in other places:

- **Placement**: every instance of a value is evaluated at the shallowest scope ON ITS CONSUMER'S PATH that binds its
  coordinates, the rule `Fold.lower` applies to a term, applied here to every value. A value two consumers reach on
  different paths is instantiated on each (the three attention copies stay three in the executable form; storing
  one is the cut's decision, not normalization's). Loop-invariant hoisting, common-branch hoisting, copy-alias
  elimination and load dedup fall out of it: an instance exists once per path and sits where its coordinates are
  bound. Sibling reduces over one extent at one scope share a loop when independent, today's merge rule.
- **Order**: dependency order within a scope, ties broken by the instance's number. Two independent siblings with
  equal numbers are the same instance and merge, so no tie survives for the exact canonical labeling to break; it is
  retired from the normal form rather than confined.
- **Names**: positional within a scope (or the value's hash), so a sub-body's text does not depend on what precedes
  it in the body, which is what lets a definition's text be cached and a golden be diffed.

Identity is the DAG's digest with leaf loads colored by type, linear in the number of values. That retires
`hoist_loop_invariants`, `hoist_common_branches`, `dedup_loads`, `eliminate_copy_aliases`, `_unify_siblings`, the
fixpoint loop around them, the whole-body labeling as the primary order, and the splicer's demand-driven
`expand_calls` in favour of the one plain inliner. It also removes two fragilities found on the way: idempotence of
the order today rests on the labeling's ranks being consistent with the input order for symmetric vertices, and the
sequential renaming puts every later name in motion when one statement changes.

The cost is a new executable form — sibling order and names change in every kernel — so the gate is no longer byte
identity but equivalence: the same values, the same placement, measured on the corpus by `make bench-kernels`.
Stage 1 below is the cache; stage 2 is the form. Stage 1 is worth doing first because its recognizer, its table and
its expansion boundary are exactly stage 2's components, and its byte-identity gate proves the recognizer sound
before the form is allowed to move.

## Experiments run (2026-10-08, scratchpad `vn.py`, `vn_experiments.py`; 309 corpus bodies plus the fused GDN body)

- **Recognition**: the fused GDN width-1 body has 1504 statements and 266 distinct values (278 with every bound
  composite index keeping its free coordinates); numbering takes 12 ms.
- **Identity**: the digest of the stores' numbers, with buffers ranked by the sorted numbers of the statements
  touching them (two passes), reproduces today's partition of the 309 corpus bodies exactly: 208 classes, no split,
  no merge, invariant under a dependency-valid random reorder plus a random renaming of every name and axis. It
  costs 0.13 s where today's normalization costs 1.5 s over the same bodies.
- **Placement**: in today's normal form no statement of the 5985 can move one scope up on its path. The raw term
  lowering of the GDN body leaves 27 such statements (hoisting does real work there), and normalization changes no
  value: the normal form is a rendering of the same DAG, which is the claim.
- **Dedup**: today's form holds six duplicate instances across the corpus and the GDN body that the DAG merges.
- **Across arms**: output-side seams leave 88–97% of a piece's values numbered as in the parent; an input-side seam
  renumbers about half, every dependent, as it must.

## Milestones and gates

Going straight to the form, as agreed:

1. **Recognizer in the tree**: `ir/stmt/values.py` with tests — lambda equivalence under loop renaming and composite
   indices, reduce binding keeping a composite's free coordinates, parameter maps telling transposes apart, the
   corpus soundness check (every repeated non-leaf value's occurrences alpha-equal) and the identity partition check
   above as a test over the corpus. GPU-free.
2. **Identity from the digest**: `canonicalize_identity` keys on the DAG digest; gate is the partition check on every
   corpus body and golden kernel (no split, no merge against today's keys), then the tune DB key version bump.
3. **Rendering**: `normalize_body` becomes number → place → order → name; the retired passes go; gate is per-scope
   multiset equality of instances with today's form over the corpus (the same work in the same scopes), then the
   corpus regen, `emmy golden restamp` with microseconds kept, and `make bench-kernels` for the executable form.
4. **Compact storage and the value table**: definitions and calls in the wire format, one plain inliner at the
   kernel-lowering boundary, the per-compile table; gate is the GDN decode compile with identical picks (115 s wall /
   220 s single-worker today).
5. **Later**: the lift per definition; features on the compact form with a refit by the nightly refresh.

## Risks

- Hoisting and dedup no longer cross a call boundary, so an executable form may change where the inline form let a
  copy's load merge with its consumer's; the byte-identity gate finds every such case, and each is a decision: accept
  the form or add the matching rule at call granularity.
- Confluence (expand-then-normalize equals normalize) was measured on one body; the corpus gate is the proof.
- A leaf load is one lambda for every index expression; outlining excludes leaves, so this merge only feeds the
  numbers of their consumers, where it is wanted.
- Every golden re-keys; the restamp rule above keeps the evidence, and the nightly refresh owns the refit.
