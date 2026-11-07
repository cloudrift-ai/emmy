# The statement and expression vocabulary

`emmy-ir` implements what `emmy/compiler/ir/stmt` and `emmy/compiler/ir/expression.py` export: the expression types
and their text, the forty-five statement types, `Body`, the body text, the statement protocol, value numbering,
identity and body normalization. Pure Rust, no Python in it, so `cargo test -p emmy-ir` runs it alone, and the tests
of the normalization steps sit beside the steps there. The extension beside it, `emmy-ir-py`, is the `emmy.emmy_ir`
module those two Python packages re-export from; `make setup` builds it into the package like the runtime extension.
It exports the driver (`normalize_body`, `normalize_uncached`, `prepare_body`) and the steps a pass calls on its own
(`fold_affine`, `simplify_body`, `rename_free`), no other step. Design: `ir/ARCHITECTURE.md` owns what a statement, a
body and the normal form ARE; this file says how the crate holds them.

## Layout

The module tree follows the design, one module per concept, and the modules stand in a tower: each imports only the ones
below it, `field`, `statement` and `body` aside, which define each other's types. From the bottom: `expr/` (the tree,
its text, the simplifier's walk and the range analysis in `mod.rs`, the simplifier's rewrite rules and the affine form
in `simplify.rs`, the reader in `parse.rs`), `memo.rs` (the memo store every memo is declared in), `elementwise.rs` (the
operation table), `dtype.rs` (the dtype set and promotion, which reads the table), `axis.rs` (axes with their speller,
and the keywords a statement's text reserves), `numbering/key.rs` (the `Key` tree and its hash, over expressions alone),
`field.rs` (the `Field` trait and the field value types), `statement.rs` (the declaration list of the statement types
and its macros), `body.rs`, `protocol/` (the protocol in `mod.rs`, what the shared statement types answer beyond their
fields in `shared.rs`, the construction checks in `checks.rs`), `text/` (the printer in `print.rs`, the reader in
`parse.rs`, what both spell in `mod.rs`), `order.rs` (the dependency constraints; the sort they feed is a `Body`
method), the rest of `numbering/` (the value numbering in `values.rs`, the scope-tree digest in `digest.rs`),
`normalize/` (the driver and the helpers the steps share in `mod.rs`, the steps by what they do: `axes`, `fold`,
`siblings`, `simplify`, `place`, `canonical`, and every tie-break in `ties.rs`) and `identity.rs` (the identity of a
body's normal form). Every file imports what it uses by name; no child module takes its parent's imports.

## Data

- A name is an interned symbol (`name.rs`); a tuple of names, expressions or axes an `Arc<[…]>`; an expression an `Arc<Node>` (its tree
  with its height); a body an `Arc<BodyData>`. A rebuild that changes nothing returns the object it was given, so
  every walk tells a change by pointer and a Python wrapper can hand back the same object.
- **Names** (`name.rs`): a name is a symbol into one process-wide table, interned on construction (`Name::new`,
  `From<&str>`), so equality and hashing are integer compares and a copy is a word; `Deref<Target = str>` and `Display`
  resolve its text, and a name orders by its text, as the canonical order sorts names and as every digest spells them
  (`Key::Str` holds a name and feeds its text bytes), so an interned name moves no identity. The table never frees; a
  lowering mints names a body at a time, and the distinct names of the A100 golden's fresh lowering and the corpus stay
  in the tens of thousands. A map keyed by names is looked up by a name, so a text looks itself up by interning first
  (`map.get(&Name::new(text))`).
- **Expressions** (`expr/`): eight expression types declared once in the `with_exprs!` macro (the variant, its Python
  class and its fields), which generates the `enum Expr`, the walks every expression type answers through its fields
  (children, height, the rebuilds) and, in the extension, the classes with a getter per field; the printer and the
  reader (exact inverses, the text being the one wire and structural form), substitution, renaming, the simplifier under
  a `SimplifyCtx` of ranges and symbolic bounds, the range analysis, the affine form. A literal keeps Python's number
  model (`Num`: int, float or bool, equal by value) because a stored kernel's text spells it that way, and is typed by a
  `LitType`. Integer rewrites are checked: a product that would overflow declines the rewrite rather than fold wrong.
- **Statement types** (`statement.rs`): every statement type is a struct declared once in the `with_statement_types!`
  macro — its fields, each with a type, the roles it plays (`DEFINES`, `READS`, `EXPRESSION`, `AXIS`, `BODY`, `STMTS`,
  `TERM`, `RECORD`, …) and a default — and the Python extension generates its classes from the same list.
  `define_statement_types!` turns the list into the structs, the `enum Statement`, the `enum StatementType` (one variant
  per statement type, with its name, layer and fields; what `Statement::ty()` returns and what a body's type queries and
  the reader take) and, per statement type, the role-driven walks (`visit`, `rewrite`, the header fields the text
  prints). Roles are a `Roles` set and a statement type's flags a `StatementFlags` set, two types that cannot be
  mistaken for one another. `KW_ONLY`, a constructor property, is declared beside a field's roles and lands in the
  field's `kw_only`, not in the set. A statement type is a block when a field of it plays `BODY`. The accessors every
  statement type has (`op`, `dtype`, `name`) are the `Field` trait's: a field type that is one answers, every other
  field's default is none. A test pins the digest of the declaration list, the names and field orders every stored text
  spells.
- **Fields** (`field.rs`): one trait, `Field`, every field type implements: how it is visited and rewritten by
  role, spelled on a declared statement type's header and read back by its `FieldType`. A number where a name is
  expected (`NumOr<T>`, a carry's literal seed) is plain data. A record (`SelectBranch`, `FragmentArg`) is declared in
  the same list with a layer of its own, a struct whose own fields play roles and a `Field` of the statement type that
  holds it. Every closed set of spellings the crate branches on is an enum declared with `closed_set!` (one text per
  variant, `Display` and `FromStr` over them, a test pinning every text, since each is a stored spelling or identity
  material): the dtypes (`DType`, `dtype.rs`, with the promotion rule; the extension exports the set as `DTYPES` and a
  Python test holds it equal to the `DataType` registry), the binary operators and literal types (`BinOp`, `LitType`),
  the GPU built-ins (`expr::Builtin`), the expression types (`ExprType`, generated from the one expression list; the
  digest spells a node by it) and the digest's other tags (`numbering::Tag`). A cast's target is `CastTarget`: a literal
  type the simplifier folds through, or any other C type the renderer spells. A value the crate only carries stays a
  name: a slab's swizzle, a tile's raster orientation, a fragment's layout and role, a slab's or an MMA wrapper's C
  type. An elementwise operation is `ElementwiseOp::Known(Operation)` or `ElementwiseOp::Other(name)`, a spelling only a
  renderer knows. The known operations are one declaration (`operations!` in `elementwise.rs`): each variant with its
  spelling and its properties, which generates the enum, its `Display`/`FromStr` pair and the table `OPS` the enum
  indexes, one `OperationInfo` row per operation (commutative, associative, the neutral element, the cluster
  representative, integer arithmetic, whether it copies its operand, its affine role, its host evaluator). Every reader
  of an operation's nature (the affine fold, the alias reads of placement and the ties, the dtype promotion, the
  clustered identity, the Python tables) reads a property; none matches a name. The neutral element is the real one
  (`-inf` for a maximum) and `seed` is what a kernel initializes with, the bounded stand-in `±1e30` the CUDA source
  spells (and two recipe goldens' bodies, as `-1e+30`; the hardware goldens and the corpus seed no maximum). A string of
  any of these sets exists where the text is read or written, at the Python boundary, and nowhere else.
- **Axes** (`axis.rs`): `Axis` (name, `Dim`, window), equal by name and extent as the Python class is; the window
  is provenance. The Python `Axis`, `Dim`, `DataType` and `ElementwiseImpl` objects stay Python classes: a statement
  holds a name (or an axis's parts) and the extension makes the object on first read.

## Protocol and body

`protocol/` writes the statement protocol once over the roles: `deps`, `defines`, the buffers, `expressions`,
`bound_axes` / `binds_axes`, `nested` / `with_bodies`, `try_map_expressions` / `with_expressions`, `rename_buffers`
(one `Renaming` per body, so a subroutine two calls share is renamed once; the body's renaming and `coalesce` live
here too), and `rewrite` behind
`rename` (α, through binders) and `substitute` (β, hygienic: a mapping a statement's own binder shadows is dropped for
its subtree). The four members no role gives are stated by statement type there: an `Accum`'s reduced axes follow a
splitting substitution, a `Let` and a `Call` read buffers inside an expression or through a subroutine, a `StridedLoop`
with an `end` binds `<axis>_end`. A statement type that checks itself is flagged `CHECKED` and does so over its own
fields (`validate`); `Statement::check` dispatches on the flag, and `finished` runs it for every constructor: the checks
Python's `__post_init__` made.

`body.rs` is `Body`: the statements, a content hash, and every def-use reading computed once on first use and kept
with the body (`definitions`, `carried_names`, `local_defs`, `free_ssa`, `axis_names`, `ssa_uses`, the filters by
statement type), plus the walks (`try_map`, `try_map_stmts`, `walk`; `map` and `map_stmts` are the same walks with an
error type that has no values, and the extension calls the fallible ones with a Python function and `PyErr`),
`backward_cone` and the topological order, which ranks by any ordered priority. `rebuilt` returns the body itself when a
step hands back its own statements, so a step that changed nothing keeps the readings. `StmtRef` names a statement by
the body that holds it and its position, which is what the numbering and a Python-side cache key on. `Subroutine` is the
shared reduction body a `Call` names, compared by object.

## Text

`text/` prints and reads the body text `ir/ARCHITECTURE.md` specifies: a clause per shared statement type, the generic
`StatementType(field=value, …):` form over the declaration for every other statement type, subroutines ahead of their
callers. The reader is a `Parser` over the expression tokenizer with the statement punctuation added. An error names its
line.

## Numbering, identity, normalization

`numbering/values.rs` is the value numbering (`value_numbers`); `digest.rs` the scope-tree hash, the use-keyed buffer
roles and the digest. Statements are keyed by their position in the body they sit in (a `StmtRef`), so a body that
holds one statement object twice numbers it twice and nothing is copied first; digest material is a `Key` tree hashed
with BLAKE3 (`Number`, 128 bits), and every order among ties is by `Key`'s total order. The operands of a commutative
operation that share a number take the order whose layout key is least under a bounded search: every order is tried
while the groups allow at most 7! of them, generated in place, and a larger operation (an unrolled window sum over
eight or more shifted reads, which the full search never finished) takes the order built position by position:
canonical and independent of the input order like the full search's, but not always its minimum. The identity baseline
over the corpus and the golden check are what shows no stored kernel's order moved. `identity.rs` memoizes the
identity per body and per coloring. `normalize/` holds the steps and the driver `normalize_body`; `canonical_order`
and `place_values` are the two steps that read the numbering, and the order ranks a product's operands through the
statement objects it permuted.

## Memos

What a process remembers across bodies is a `memo::Memo`, declared with the `memo!` macro that names its capacity in
entries (`normal_forms` in the normalization driver, `identities` in `identity.rs`, both `BODY_MEMO_CAPACITY`) and
lists it in one registry through `memo::declared`, so every store is inspected (entries, hits, misses) and cleared
from one place (`memo::stats` and `memo::clear_all`; `emmy_ir.memo_stats()` and `emmy_ir.clear_memos()` in Python,
for a test worker or a process pool that starts clean). The caches are
`quick_cache`'s: bounded by entry count, evicting by its one policy (S3-FIFO), sharded so no lock is held across a
computation, and one asker computes a missing value while a second asker of the same key waits for it, so a
computation that asked its own memo for the key it is computing would wait on itself; `get_or_try_insert` refuses
that with an error, and the key leaves the thread's list however the computation ends (no normalization step recurses
into `normalize_body` today).
A memo's key is the body's exact form, `body::MemoKey`: equal when the statements are and so is everything body
equality leaves out, every axis's hint and window, every expression's literals and every number a field holds by
their bits (a signed zero keys apart from zero, every NaN alike, as the digest keys them), with a hash computed once
and kept with the body by one walk of the subtree; no key is a printed text. A body also keeps the one normal form it
was given (`normal_form`, `remember_normal_form`, `is_normal`, `mark_normal` in `body.rs`), so a body normalized
twice costs one lookup and a protocol rewrite that keeps a normal form says so through that API. The extension's
wrapper tables (a body's statement wrappers, a subroutine's one Python object) are listed in the same registry: not
bounded, since a live object keeps its one wrapper, but purged of dead entries when their size reaches a multiple of
1,024 and on `clear_all`.

The canonical order decides what the digest leaves open, and the lift of a body into terms reads that order: sibling
regions are addressed by position in every recorded cut decision, and a product's operand order is its orientation
where neither operand owns an output axis. So the ties are broken the way the stored goldens were spelled, and
`normalize/ties.rs` states each rule with its reason: siblings a scope may place either way sort by statement type name,
then by their fields in declaration order with names abstracted and a one-element tuple after longer ones, so every
store follows the values it reads and an `Assign`'s operation decides before its dtype; sibling loops sort by a digest
of their extent, the one the Python order used; a commutative operation lists accumulators, then loads and the copies
that widen them, then computed values with the deeper cone first, then by value. Changing a rule is a restamp decision.

What the port keeps of the Python implementation: stored kernels keep their rows, two raw bodies share an identity in
Rust exactly when they did in Python, the normal form is final. The text of a normal form, the order among ties and
the digest value were allowed to move, and did: a tie the Python order broke by the hash of a structural form, the
Rust order breaks by the text.

## Errors

A function that can refuse returns `Result<T, Error>`; the `Error` variants map one to one onto the Python exceptions
the extension raises (`ValueError`, `TypeError`, `KeyError`, `ZeroDivisionError`, `NotImplementedError`,
`RecursionError`). Library code holds no `unwrap` or `expect` (STYLE.md, Rust): a lock that may be poisoned is reopened,
an invariant that holds by construction is a total function (`rebuild_bodies`, `map_children`), and digest material
falls back to a debug form where a name has no text, and the canonical order's keys carry the error up. An expression
carries the height of its tree; the reader and the one base of every Python expression object refuse one higher than 256
levels (`RecursionError`), whatever built it, rather than let a later walk overflow the stack, and a chain of
expressions drops level by level. An error met while a Python value converts keeps its own type across the boundary,
with the statement type and field ahead of its message.