# The statement and expression vocabulary

`emmy-ir` implements what `emmy/compiler/ir/stmt` and `emmy/compiler/ir/expression.py` export: the expression kinds
and their text, the forty-five statement kinds, `Body`, the body text, the statement protocol, value numbering,
identity and body normalization. Pure Rust, no Python in it, so `cargo test -p emmy-ir` runs it alone, and the tests
of the normalization steps sit beside the steps there. The extension beside it, `emmy-ir-py`, is the `emmy.emmy_ir`
module those two Python packages re-export from; `make setup` builds it into the package like the runtime extension.
It exports the driver (`normalize_body`, `normalize_uncached`, `prepare_body`) and the steps a pass calls on its own
(`fold_affine`, `simplify_body`, `rename_free`), no other step. Design: `ir/ARCHITECTURE.md` owns what a statement, a
body and the normal form ARE; this file says how the crate holds them.

## Layout

The module tree follows the design, one module per concept: `expr/` (the tree and its text in `mod.rs`, the
simplifier and affine form in `simplify.rs`, the reader in `parse.rs`), `kinds/` (the kind list and its macros in
`mod.rs`, the `Field` trait in `field.rs`, axes, dtypes and the elementwise tables beside it), `body.rs`, `text/` (the
printer in `print.rs`, the reader in `parse.rs`, what both spell in `mod.rs`), `protocol.rs`, `numbering/` (the `Key`
tree and its hash in `key.rs`, the value numbering in `values.rs`, the scope-tree digest in `digest.rs`, the identity
in `identity.rs`), `normalize/` (the driver and the helpers the steps share in `mod.rs`, the steps by what they do:
`axes`, `fold`, `siblings`, `simplify`, `place`, `canonical`, the dependency order in `order.rs` and the tie-breaks in
`ties.rs`) and `memo.rs`, the one bounded memo both the normal form and the identity are kept in.

## Data

- A name is an `Arc<str>`; a tuple of names, expressions or axes an `Arc<[…]>`; an expression an `Arc<Expr>`; a body an
  `Arc<BodyData>`. A rebuild that changes nothing returns the object it was given, so every walk tells a change by
  pointer and a Python wrapper can hand back the same object.
- **Expressions** (`expr/`): eight kinds in one `enum Expr`, the printer and the reader (exact inverses, the text being
  the one wire and structural form), substitution, renaming, the simplifier under a `SimplifyCtx` of ranges and
  symbolic bounds, the range analysis, the affine form. A literal keeps Python's number model (`Num`: int, float or
  bool, equal by value) because a stored kernel's text spells it that way.
- **Kinds** (`kinds/mod.rs`): every kind is a struct declared once in the `with_kinds!` macro — its fields, each with a type,
  the roles it plays (`DEFINES`, `READS`, `EXPRESSION`, `AXIS`, `BODY`, `STMTS`, `TERM`, `RECORD`, …) and a default — and
  the Python extension generates its classes from the same list. `define_kinds!` turns the list into the structs, the
  `enum Statement` and, per kind, the role-driven walks (`visit`, `rewrite`, the header fields the text prints).
- **Fields** (`kinds/field.rs`): one trait, `Field`, every field type implements: how it is visited and rewritten by role,
  spelled on a declared kind's header and read back by its `Ty`. A number where a name or an expression is expected
  (`NumOr<T>`) is plain data. A record (`SelectBranch`, `FragmentArg`) is a struct whose own fields play roles.
- **Axes** (`kinds/axis.rs`): `Axis` (name, `Dim`, window), equal by name and extent as the Python class is; the window is
  provenance. The Python `Axis`, `Dim`, `DataType` and `ElementwiseImpl` objects stay Python classes: a statement holds
  a name (or an axis's parts) and the extension makes the object on first read.

## Protocol and body

`protocol.rs` writes the statement protocol once over the roles: `deps`, `defines`, the buffers, `expressions`,
`bound_axes` / `binds_axes`, `nested` / `with_bodies`, `map_expressions`, `rename_buffers` (one `Renaming` per body, so
a subroutine two calls share is renamed once), and `rewrite` behind
`rename` (α, through binders) and `substitute` (β, hygienic: a mapping a statement's own binder shadows is dropped for
its subtree). The four members no role gives are stated by kind there: an `Accum`'s reduced axes follow a splitting
substitution, a `Let` and a `Call` read buffers inside an expression or through a subroutine, a `StridedLoop` with an
`end` binds `<axis>_end`. The constructors' checks (`checked`, `finished`) are the ones Python's `__post_init__` made.

`body.rs` is `Body`: the statements, a content hash, and every def-use reading computed once on first use and kept
with the body (`definitions`, `carried_names`, `local_defs`, `free_ssa`, `axis_names`, `ssa_uses`, the kind filters),
plus the walks (`map`, `map_stmts`, `walk`), `backward_cone`, the topological order and `coalesce`. `StmtRef` names a
statement by the body that holds it and its position, which is what a Python-side cache keys on. `Subroutine` is the
shared reduction body a `Call` names, compared by object.

## Text

`text.rs` prints and reads the body text `ir/ARCHITECTURE.md` specifies: a clause per shared kind, the generic
`Kind(field=value, …):` form over the declaration for every other kind, subroutines ahead of their callers. The reader
is a `Parser` over the expression tokenizer with the statement punctuation added. An error names its line.

## Numbering, identity, normalization

`numbering/values.rs` is the value numbering (`value_numbers`); `digest.rs` the scope-tree hash, the use-keyed buffer
roles and the digest. Statements are keyed by the address of their shared object (`distinct` makes a body's statements
unique first), digest material is a `Key` tree hashed with BLAKE3 (`Number`, 128 bits), and every order among ties is
by `Key`'s total order. `identity.rs` memoizes the identity per body and per coloring. `normalize/` holds the steps
and the driver `normalize_body`; `canonical_order` and `place_values` are the two steps that read the numbering. Both
memos are `memo::Memo`, keyed by the body's text under its content hash and bounded by entry count: body equality
leaves the axis windows out, the text spells them, and a window is identity material.

The canonical order decides what the digest leaves open, and the lift of a body into terms reads that order: sibling
regions are addressed by position in every recorded cut decision, and a product's operand order is its orientation
where neither operand owns an output axis. So the ties are broken the way the stored goldens were spelled, and
`normalize/ties.rs` states each rule with its reason: siblings a scope may place either way sort by kind name, then by
their fields in declaration order with names abstracted and a one-element tuple after longer ones, so every store
follows the values it reads and an `Assign`'s operation decides before its dtype; sibling loops sort by a digest of
their extent, the one the Python order used; a commutative operation lists accumulators, then loads and the copies that
widen them, then computed values with the deeper cone first, then by value. Changing a rule is a restamp decision.

What the port keeps of the Python implementation: stored kernels keep their rows, two raw bodies share an identity in
Rust exactly when they did in Python, the normal form is final. The text of a normal form, the order among ties and
the digest value were allowed to move, and did: a tie the Python order broke by the hash of a structural form, the
Rust order breaks by the text.

## Errors

A function that can refuse returns `Result<T, Error>`; the error kinds map one to one onto the Python exceptions the
extension raises (`ValueError`, `TypeError`, `KeyError`, `ZeroDivisionError`, `NotImplementedError`). Library code holds
no `unwrap` or `expect` (STYLE.md, Rust): a lock that may be poisoned is reopened, an invariant that holds by
construction is a total function (`rebuild_bodies`, `map_children`), and digest material falls back to a debug form
where a name has no text.
