# The pure terms

`emmy-pure` implements the terms `emmy/compiler/ir/pure` exports: the `Lambda` every IR level binds with, the Tile IR
`Fold` term and the `Twist` recipe that conjugates its monoid. A term denotes a value: it binds names, carries an
algebra and compares up to α-renaming, and it has no position in a statement stream (`ir/ARCHITECTURE.md`, "Pure terms
vs statements", owns that invariant and what the terms ARE). This crate sits beside `emmy-ir`, as `ir/pure` sits beside
`ir/stmt`: it builds on the statement vocabulary, and `emmy-ir` knows nothing of it. Pure Rust; `cargo test -p
emmy-pure` runs its tests, and the Python suite runs it through the `emmy.emmy_ir` extension (`emmy-ir-py`).

## The boundary

Python holds a handle on a term and asks whole questions — a reading, the canonical form, a rewrite, the recipe
matching, the tree normalization, the lowering — and never walks a term node by node across the boundary. What a pass
DECIDES stays a pass in Python (the cut, the split, the layout, which recipe to try, the schedule), calling the crate
for the mechanics. The recipe definitions (SOFTMAX, WELFORD) are Python data built through this crate's constructors.
The one Python member of a term is the `rewrite` glue, which turns a rename function and an axis function into the
maps the crate takes.

## A term is reached by object

Two terms with the same stored parts are EQUAL, and they are still two terms. The placement reads one value where it
sees one term, and a pass tells two occurrences of equal terms apart by which one it holds, so the crate keeps
identity (`ptr_eq`) beside equality and never merges equal terms on its own. Sharing is made in one place, the tree
normalization, which collapses same-value cones onto one term.

The extension gives a term one Python object while anything holds it (a table keyed by the term's address keeps a
weak reference to it), so `is` in Python is `ptr_eq` in the crate. A term Python builds takes its operands' objects
as it is made, so it reads back the objects it was given, and a wrapper keeps every object it hands out, so every
term reached by walking from a held one keeps its object, and its `id`, for the whole walk. A `Lambda`, a `Recipe`
and a view are values: they compare by what they hold.

## Modules

- `lambda.rs` — `Lambda`: construction normalizes the body (`emmy-ir`'s lambda normalization) and refuses an impure
  statement, an undefined result or an unbound read; `closing` forms a closed lambda; `rename` in lockstep (a map
  that keeps every name apart keeps the checks true, so its result is formed without them); `canonical`, `cone`,
  `componentwise` / `components`.
- `twist.rs` — `Recipe`, `Channel`, `Twist`: the fused ⊕ over a carrier's state names (`program`) and the injected
  singleton (`inject`).
- `fold.rs` — `Fold`: its stored parts and formation in two halves that `Fold::new` runs in order:
  the canonical rewrites (a zero-axis term loses its shadowed operands, a bilinear term takes its A operand first)
  and the checks, which refuse and never repair (the positional binding, the monoid arity, the carrier reads, the
  observer, a lift that applies). Then the reads every pass asks: `exposes`, `bindings`, `applied`, `axis`,
  `free_axes`, `carries`, `combine`, `canonical`. A reducing term may state the `range` its axis walks; the
  coordinates the range reads join the lift's trailing params, so they are part of what the term is evaluated over.
  A range may also read the value of an operand that is one value for the whole walk, spelled as the lift's param
  for it (`applied_range` spells it as the operand result, as `applied` does the lift): the tree's pruning and the
  lowering's demand walk count that read, so the operand stays bound and is placed ahead of the loop.
- `values.rs` — `Fold::inlined`: a tree with every read by index replaced by the value it reads. A term that states
  `cells` over one operand and holds one `Pre` per value read (`Fold::read`, `Fold::as_read`) reads that operand, a
  value defined once over its own coordinates, at an index. A kernel keeps such a value: `Fold::lower` writes the
  operand once, in a nest of its own that stores each cell with a `Define`, ahead of the scope that reads it, and
  each read as the `Load` of one cell. A coordinate every read takes at the coordinate itself is not walked where
  that scope binds it. Where a kernel is fused into another, the read gives way instead (`Fold::inlined`) to the
  operand with each coordinate the read binds replaced by the index read. One copy is taken of a term per
  substitution of the coordinates that term is evaluated over: two reads that differ only in a coordinate a term does
  not read share that term, as a softmax's row maximum is one term under the sum over a row and under each cell of
  it. The first copy of a term keeps its names and every later one spells apart what the term defines and the axis
  it walks. A Loop IR body reads a defined value with a `Load`; the `Pre` of a read term is how a lift names an
  operand it reads by index, as a carrier read does.
- `readings.rs` — the derived readings: `exposing`, `based`, `roles`, the bilinear channels and the contraction they
  read as, `tiles_whole`, `chunked`, `per_state`, the reduction, slab and affine readings, `merge`, `step`.
- `rewrite.rs` — `Fold::rewrite`: a term renamed, substituted and axis-mapped the way a statement is, the
  substitution hygienic about the axis each term binds and folded under a range context when it carries one. The
  rewrite of one term over operands already rewritten (`TermRewrite::node`) is what the copies of a read value are
  built with.
- `fuse.rs` — `Fold::fuse`: the one generic algorithm over the recipes. It finds the pivot among the term's operands,
  matches the score cone and a channel's pattern by canonical form, and instantiates the recipe by renaming. Two
  folds fuse where they walk the same range of the axis, and the fused fold keeps it.
- `tree.rs` — the tree-wide canonical forms (`Fold::normalized`): the pass-through dissolve, the per-state split, the
  unread-component prune and the cone sharing, to a fixpoint remembered on the result per stored set.
- `lower.rs` — the one lowering to Loop IR (`Fold::lower`), whose result kernel identity digests, and the demand walk
  under it (`read_components`). The free loops open outermost the coordinate the most terms are evaluated over, and a
  term read along a block of a partition counts for the block's coordinate (`Fold::block_start`): it is evaluated at
  `start + trip`, so the nest is ordered as it would be were the start written into every index.
- `range.rs` — `Fold::scheduled`: the tree as the schedule addresses it and a kernel tier lowers it. A tier reads
  a term's range itself, so every term keeps its range. A range starts at the greatest of its starts and ends at
  the least of its ends. A block of a split (an axis whose window is a partition) leaves from the start some end
  lies a fixed way past: the axis takes that distance as its trips, every read of it is `start + axis`, the term
  keeps what its own bounds leave of the block as its range, and a read below the term clamps onto the last
  position of a symbolic stream, whose last block may run past it. Any other term is the term it was.
  `Fold::block_start` names the start a block leaves from, the one place that says which coordinate moves a block.

## Memos

A term is immutable, so every derived read is computed once and kept on the term (a `OnceLock`, or a small table for
a read that takes an argument): a structural copy that keeps a subtree keeps its answers, and a walk over a large fused
tree stays linear. A lowering is one of them, keyed by everything it reads: the bound coordinates, the stores, the
axis table with each axis's window and hint, and the symbolic extents. The memos are caches on the term, never fields
of a golden or an op's knobs, and transport pickles the stored parts only. The extension keeps no second memo: its
wrapper holds only the Python objects it has already handed out.

## Invariants

- **Normal form unchanged by the port.** The crate computes every canonical form the Python terms computed, byte for
  byte; a change to one is a restamp decision, not a port detail. That includes the tie-break of the A-first
  orientation between two slabs that both read the reduce axis last, which follows the name of each one's own row: a
  term carries no axis order to read instead, so any other rule re-keys the kernels it touches on every card.
- **Normalization carries no knob.** The tree normalization is a pure function of the tree and the stored names.
