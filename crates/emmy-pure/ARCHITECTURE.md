# The pure terms

`emmy-pure` implements the terms `emmy/compiler/ir/pure` exports: the `Lambda` every IR level binds with, the Tile IR
`Fold` term and the `Twist` recipe that conjugates its monoid. A term denotes a value: it binds names, carries an
algebra and compares up to α-renaming, and it has no position in a statement stream (`ir/ARCHITECTURE.md`, "Pure terms
vs statements", owns that invariant and what the terms ARE). This crate sits beside `emmy-ir`, as `ir/pure` sits beside
`ir/stmt`: it builds on the statement vocabulary, and `emmy-ir` knows nothing of it. Pure Rust; `cargo test -p
emmy-pure` runs its tests, and the Python suite runs it through the `emmy.emmy_ir` extension (`emmy-ir-py`).

## The boundary

Python holds a handle on a term and asks whole questions — a reading, the canonical form, the tree normalization, the
lowering — and never walks a term node by node across the boundary. What a pass DECIDES stays a pass in Python (the
cut, the split, the layout, which recipe fuses, the schedule), calling the crate for the mechanics. The recipe
definitions (SOFTMAX, WELFORD) are Python data built through this crate's constructors; `Fold.fuse` (the recipe
matching) and `Fold.rewrite` (whose `Sigma` may carry a Python `after` hook) are the term's last Python members.

A term is reached by OBJECT: the placement reads one value where it sees one term, so the extension gives a term one
Python object while anything holds it (a table keyed by the term's address keeps a weak reference to it), and a term's
operands read back as the objects they were built from. A wrapper holds its operands' objects, so every term reached
by walking from a held one keeps its object, and its `id`, for the whole walk.

## Modules

- `lambda.rs` — `Lambda`: construction normalizes the body (`emmy-ir`'s lambda normalization) and refuses an impure
  statement, an undefined result or an unbound read; `closing` forms a closed lambda; `rename` in lockstep (a map
  that keeps every name apart keeps the checks true, so its result is formed without them); `canonical`, `cone`,
  `componentwise` / `components`.
- `twist.rs` — `Recipe`, `Channel`, `Twist`: the fused ⊕ over a carrier's state names (`program`) and the injected
  singleton (`inject`).
- `fold.rs` — `Fold`: its stored parts and formation (`Fold::new`: the shadowed operand of a zero-axis term, the
  positional binding, the carrier reads, A-first orientation, the observer), and the reads every pass asks:
  `exposes`, `bindings`, `applied`, `axis`, `free_axes`, `carries`, `combine`, `canonical`.
- `readings.rs` — the derived readings: `exposing`, `based`, `roles`, the bilinear channels and the contraction they
  read as, `tiles_whole`, `chunked`, `per_state`, the reduction, slab and affine readings, `merge`, `step`.
- `tree.rs` — the tree-wide canonical forms (`Fold::normalized`): the pass-through dissolve, the per-state split, the
  unread-component prune and the cone sharing, to a fixpoint remembered on the result per stored set.
- `lower.rs` — the one lowering to Loop IR (`Fold::lower`), whose result kernel identity digests, and the demand walk
  under it (`read_components`).

## Memos

A term is immutable, so every derived read is computed once and kept on the term (a `OnceLock`, or a small table for
a read that takes an argument): a structural copy that keeps a subtree keeps its answers, and a walk over a large fused
tree stays linear. The memos are caches on the term, never fields of a golden or an op's knobs, and transport pickles
the stored parts only.

## Invariants

- **Normal form unchanged by the port.** The crate computes every canonical form the Python terms computed, byte for
  byte; a change to one is a restamp decision, not a port detail.
- **Normalization carries no knob.** The tree normalization is a pure function of the tree and the stored names.
