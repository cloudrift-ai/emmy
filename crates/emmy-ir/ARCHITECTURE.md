# The statement and expression vocabulary

`emmy-ir` implements what `emmy/compiler/ir/stmt` and `emmy/compiler/ir/expression.py` export: the expression types
and their text, the forty-five statement types, `Body`, the body text, the statement protocol, value numbering,
identity and body normalization. Pure Rust, no Python in it, so `cargo test -p emmy-ir` runs it alone, and the tests
of the normalization steps sit beside the steps there. The extension beside it, `emmy-ir-py`, is the `emmy.emmy_ir`
module those two Python packages re-export from; `make setup` builds it into the package like the runtime extension.
It exports the driver, `normalize_body`, the steps a pass calls on its own (`one_operation_per_definition`,
`simplify_body`, `prepare_body`) and the lambda normalization (`normalize_lambda_body`: `fragment_order`, then
`canonical_values`), no other step. Design: `ir/ARCHITECTURE.md` owns what a
statement, a body and the normal form ARE; this file says how the crate holds them.

## Layout

The modules stand in a tower: each imports only the ones below it (`field`, `statement` and `body` aside, which define
each other's types), from the interned names and the expression tree at the bottom, through the operation table, the
dtypes, the axes, the field types, the statement declarations and `Body`, the protocol, the text, the ordering
constraints and the value numbering, to `normalize/` (the driver `normalize_body` and the steps, one module per step,
every tie-break in `ties.rs`), `identity.rs` and `strength.rs` (an emitter's spelling of an index division). Every
file imports what it uses by name; no child module takes its parent's imports.

## Data

- A name is an interned symbol (`name.rs`); a tuple of names, expressions or axes an `Arc<[…]>`; an expression an
  `Arc<Node>` (its tree with its height); a body an `Arc<BodyData>`. A rebuild that changes nothing returns the object
  it was given, so every walk tells a change by pointer and a Python wrapper can hand back the same object.
- **Names** (`name.rs`): a name is its text, kept once in one process-wide table that never frees (`Name::new`,
  `From<&str>`), so equality and hashing compare the pointer, a copy is a word and `Deref<Target = str>` costs nothing;
  a name orders by its text, as the canonical order sorts names and as every digest spells them (`Key::Str` holds a name
  and feeds its text bytes), so an interned name moves no identity. Only names are interned: a text that is digest
  material and not a name (a body's text, a statement's spelling, an axis header) is a `Key::Text`, which hashes the
  same bytes and is kept by no table.
- **Expressions** (`expr/`): seven expression types declared once in the `with_exprs!` macro (the variant, its Python
  class and its fields), which generates the `enum Expr`, the walks every expression type answers through its fields
  (children, height, the rebuilds) and, in the extension, the classes with a getter per field. An operation over its
  operands is one node, `Apply(ElementwiseOp, args)`, whether the table knows the operation or only a renderer does,
  and whether the text spells it as an operator or a call is the printer's concern; Python sees an operator over two
  operands as `BinaryExpression` (`op`, `left`, `right`) and every other application as `FunctionCallExpression`
  (`name`, `args`), two views over the one node. The printer and the
  reader (exact inverses, the text being the one wire and structural form), substitution, renaming, the simplifier under
  a `SimplifyCtx` of ranges and symbolic bounds, the range analysis, the affine form. Where the range analysis answers
  nothing — a guarded or padded coordinate, a select whose condition correlates with a branch — `exact_range`
  evaluates the expression at every point of a small static domain; the analysis stays conservative-or-nothing, since
  the simplifier and the cut pass both read it and a loose bound (a select's branch hull) would lay a workspace out
  over addresses no channel writes. A literal keeps Python's number
  model (`Num`: int, float or bool, equal by value) because a stored kernel's text spells it that way, and is typed by a
  `LitType`. Integer rewrites are checked: a product that would overflow declines the rewrite rather than fold wrong.
  The two kinds of scalar arithmetic the IR has are the two expression classes (`ExprClass`), each a predicate over the
  operation table that every node of an expression must satisfy, one inside the other: `Value` admits every operation
  (the table's, and a spelling only a renderer knows) over any operand (SSA names, axes, GPU built-ins, literals of any
  type, select, cast, a buffer's flat index), and `Index` is its restriction to integer arithmetic over integer operands
  (the affine operations, division and remainder, minimum and maximum, the comparisons and logical operators, the
  bitwise operators and shifts). Every expression field declares the class it admits beside its
  roles (`index: Exprs [EXPRESSION | READS] : Index`, a record's fields their own), a field has a class exactly when it
  plays `EXPRESSION`, and construction refuses a value outside it, naming the field. The text spells an operator by
  its infix, a negation as the prefix `-x` (over a literal it stays the call `negative(3.0)`, which the reader would
  otherwise take as a signed number), every other operation as a call; a name that is a word of the reader's grammar
  (`load`, `for`, `step`, …) is written between backticks wherever it stands, on a statement line or inside an
  expression, so the reserved words are the expression layer's.
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
  Python test holds it equal to the `DataType` registry), the literal types (`LitType`),
  the GPU built-ins (`expr::Builtin`), the expression types (`ExprType`, generated from the one expression list; the
  digest spells a node by it), the digest's other tags (`numbering::Tag`) and a fragment's layout and role
  (`fragment::Layout`, `fragment::Role`, which the lane-to-element table is keyed on). A cast's target is `CastTarget`:
  a literal type the simplifier folds through, or any other C type the renderer spells. A value the crate only carries
  stays a name: a slab's swizzle, a tile's raster orientation, a slab's or an MMA wrapper's C type. An elementwise operation is `ElementwiseOp::Known(Operation)` or `ElementwiseOp::Other(name)`, a spelling only a
  renderer knows. The known operations are one declaration (`operations!` in `elementwise.rs`): each variant with its
  spelling and its properties, which generates the enum, its `Display`/`FromStr` pair and the table `OPS` the enum
  indexes, one `OperationInfo` row per operation (commutative, associative, the neutral element, the cluster
  representative, integer arithmetic, whether it copies or selects its operand, its affine role, its host evaluator,
  its infix spelling and precedence, and for a composite function its definition in primitive operations, which
  `Operation::definition` gives over an operand and kernel lowering writes into a body). An approximate instruction
  is an operation of its own, named after the exact one (`exp_fast`, `divide_fast`, `sqrt_fast`, …): the same
  function and the same host value, a faster instruction with a documented error on a device. Every reader
  of an operation's nature (the affine fold, the alias reads of placement and the ties, the dtype promotion, the
  clustered identity, the Python tables) reads a property; none matches a name. The neutral element is the real one
  (`-inf` for a maximum) and `seed` is what a kernel initializes with, the bounded stand-in `±1e30` the CUDA source
  spells. A string of any of these sets exists where the text is read or written, at the Python boundary, and nowhere
  else.
- **Axes** (`axis.rs`): `Axis` (name, `Dim`, window), equal by name and extent as the Python class is; the window
  is provenance. The Python `Axis`, `Dim`, `DataType` and `ElementwiseImpl` objects stay Python classes: a statement
  holds a name (or an axis's parts) and the extension makes the object on first read.

## Protocol and body

`protocol/` writes the statement protocol once over the roles: `deps`, `defines`, the buffers, `expressions`,
`bound_axes` / `binds_axes`, `nested` / `with_bodies`, `try_map_expressions` / `with_expressions`, `rename_buffers`
(one `Renaming` per body, so a subroutine two calls share is renamed once; the body's renaming and `coalesce` live
here too), and `rewrite` behind
`rename` (α, through binders) and `substitute` (β, hygienic: a mapping a statement's own binder shadows is dropped for
its subtree). The three members no role gives are stated by statement type there: an `Accum`'s reduced axes follow a
splitting substitution, and a `Let` and a `Call` read buffers inside an expression or through a subroutine. A loop
that stops early stops at a value bound ahead of it (a `Let`), which its `end` and the statements of its body read
like any name; the loop implies no name of its own. A statement type that checks itself is flagged `CHECKED` and does
so over its own
fields (`validate`); `Statement::check` dispatches on the flag, and `finished` runs it for every constructor: the checks
Python's `__post_init__` made.

`body.rs` is `Body`: the statements, a content hash, and every def-use reading computed once on first use and kept
with the body (`definitions`, `defined`, `free`, `buffers`, `defined_states`, `escaping_states`, `local_defs`,
`free_ssa`, `axis_names`, `ssa_uses`, the filters by statement type), plus the walks (`try_map`, `try_map_stmts`,
`walk`; `map` and `map_stmts` are the same walks with an error type that has no values, and the extension calls the
fallible ones with a Python function and `PyErr`), `backward_cone`, `pruned` and the topological order, which ranks by
any ordered priority. `rebuilt` returns the body itself when a step hands back its own statements, so a step that
changed nothing keeps the readings. `StmtRef` names a statement by the body that holds it and its position, which is
what the numbering and a Python-side cache key on. `Subroutine` is the shared reduction body a `Call` names, compared
by object.

## Text

`text/` prints and reads the body text `ir/ARCHITECTURE.md` specifies: a clause per shared statement type, the generic
`StatementType(field=value, …):` form over the declaration for every other statement type, subroutines ahead of their
callers. The reader is a `Parser` over the expression tokenizer with the statement punctuation added. An error names its
line.

## Numbering, identity, normalization

`numbering/values.rs` is the value numbering (`value_numbers`); `digest.rs` the scope-tree hash, the use-keyed buffer
roles and the digest; `congruence.rs` the congruent carried states (`congruent_states`), numbered with every grouped
state keyed by its group until the groups hold. Statements are keyed by their position in the body they sit in (a
`StmtRef`), so a body that holds one statement object twice numbers it twice and nothing is copied first; digest
material is a `Key` tree hashed with BLAKE3 (`Number`, 128 bits), and every order among ties is by `Key`'s total
order. A value carries its parameters to every statement that reads it, so a `Param` is shared, not copied, and keeps a
fingerprint; inside a key it hashes, compares and orders as the key it holds. The operands of a commutative operation that share a number take the order whose layout key is least under a
bounded search: every order is tried while the groups allow at most 7! of them, generated in place, and a larger
operation (an unrolled window sum over eight or more shifted reads, which the full search never finished) takes the
order built position by position: canonical and independent of the input order like the full search's, but not always
its minimum. `identity.rs` memoizes the identity per body and per coloring. `normalize/` holds the steps and the
driver `normalize_body`; `canonical_order`, `place_values` and `merge_congruent_states` are the steps that read the
numbering, and the order ranks a product's operands through the statement objects it permuted. A commutative node's
two sides enter the digest in content order (an operand's whole value, its number with its parameters), keyed twice:
once to learn the order on state rolled back, once in it, so a parameter or buffer first met inside the node takes the
index the content gives it. Two loads of one buffer that tie on rank order the normal form by their names, and the
identity stays the same whatever the buffers are called.

## Kernel IR bodies

The normal form is two kinds of step. The optimizing steps remove work: the exact simplifications, every value
computed once and placed at the shallowest scope that holds its operands, congruent states merged, unread values
dropped. The canonical steps make equal bodies one body: sibling reduce loops over one axis merged into one loop, the
sibling order, the operand order its ranks spell, and the sequential names. An identity reads a Loop IR body, so
`normalize_body` runs both until they settle. No identity reads a kernel body, so a `KernelOp` takes the optimizing
steps alone, through `optimize_body`, at construction: its statements stand in the order and under the names the pass
that built them wrote, and its loops stay the loops its schedule wrote, which is what a listing shows and what the
CUDA compiler schedules. The merge is the step that shows the difference: four rows' reduce loops are one body in Loop
IR, and as one kernel loop four times the size they ran at half the speed on an A100. Motion can put two bindings of
one name in one scope, so the optimizing steps start from `distinct_names`: the first binder of a spelling keeps it, a
later one takes it with a number, sibling loops over one axis keep the one name, and storage a statement updates in
place keeps its name whether or not a statement in scope declares it.

The crate names no kernel statement type for any of it. The numbering, the ordering constraints and placement read a
kernel statement through the protocol like any other, so a name a statement defines at each of several positions is a
value apart, a name it reads and defines again (storage it updates in place) is read as the value it held where the
read stands, a state a loop updates is read inside the loop as the step before, and a block holding an ordered
protocol (a barrier) stays where it is. A read of storage the body writes carries the version it reads and the barrier
it stands behind, since another thread's write of the storage lands at a barrier; a read of storage nothing in the
body writes holds one value throughout, so it is shared and placed across a barrier.

A built-in has a range: what the hardware allows (`Builtin::hardware_range`), narrowed by the launch a statement
states. A `Grid` states its own (`launch_ranges` in the protocol: `thread_idx.x` below the CTA's threads, `block_idx.x`
below its CTAs), and the simplifier reads it for the `Grid`'s decode and its body, so a decode of the flat thread id
splits into the CTA's and the thread's digits, and a built-in the launch pins to one value is that value. Loop IR
holds no built-in, so no identity moves.

`fragment.rs` holds the lane-to-element table of the tensor-core fragments, the one statement of which elements of a
tile each lane of a warp holds in which register. `fragment_table(layout, role, dtype)` gives one row: the lane
coordinates the row reads (a name and an expression over the lane, `fragment::LANE`, and the coordinates before it),
per register and element the row and column offset in the tile as expressions over those coordinates, the XOR masks
of a per-row reduce, and for a register of block scales the lanes that supply it under the instruction's two
selectors (`SCALE_SELECTOR_A`, `SCALE_SELECTOR_B`). `paired_accumulator_table` is the accumulator of Volta's paired
tile. A `RegFragment` holds as many registers as its row (`RegFragment::registers`); no statement carries a count.
The CUDA renderer reads the rows of an accumulator of floats through the extension (`FragmentApply`,
`FragmentRowReduce`, `RegStore`); the operand rows, the packed accumulators and the block scales are what its C++
loaders compute, written here for the PTX compiler and checked by a test that every element of a tile is held once
(twice by Volta's operands), not against a card.

`input.rs` is the check a code generator's input passes (`check_input`): typed in full (`types.rs`), one flat index
per access outside the accesses their dimensions define, no loop asking for unrolling, and every fragment statement
naming a layout and a role the table has.

`strength.rs` is not a step of the normal form. `reduce_strength` spells every index division and remainder by a power
of two over a dividend the ranges prove non-negative as a shift and a mask, the target's spelling. The simplifier reads
`/` and `%` and knows no range of a shift, so an emitter runs it on the body it is about to spell and no body keeps it.

## Memos

What a process remembers across bodies is three `memo::Memo`s, the normal forms and the optimized forms
(`normal_forms`, `optimized_forms` in the normalization driver) and the identities (`identities` in `identity.rs`),
each bounded by the weight its entries hold between them: an entry weighs the statements of the bodies it holds,
`BODY_MEMO_WEIGHT` between them, since a whole-layer body is thousands of statements and a bound by entry count let a
process that lowers many goldens hold gigabytes of normal forms. The caches are `quick_cache`'s: evicting by its one
policy (S3-FIFO), sharded so no lock is held across a computation, and one asker computes a missing value while a
second asker of the same key waits for it (no normalization step recurses into `normalize_body`). A memo's key is the
body's exact form, `body::MemoKey`: equal when the statements are and so is everything body equality leaves out, every
axis's hint and window, every expression's literals and every number a field holds by their bits (a signed zero keys
apart from zero, every NaN alike, as the digest keys them), with a hash computed once and kept with the body by one
walk of the subtree; no key is a printed text. A body also keeps the one normal form it was given (`normal_form`,
`remember_normal_form`, `is_normal`, `mark_normal` in `body.rs`), so a body normalized twice costs one lookup and a
protocol rewrite that keeps a normal form says so through that API. A normal form also enters `normal_forms` under its
own form, since it is its own normal form: a body built again equal to one, which passes do often, reads it back
instead of taking a round to learn it is final. It keeps, the same way, whether every scope of it is in dependency
order (`is_sorted`, `mark_sorted`): the dependency sort's result and the canonical order's are, so the sort leaves
such a body alone without working out its constraints. The extension's wrapper tables (a body's statement wrappers, a
subroutine's one Python object) are not bounded, since a live object keeps its one wrapper, and drop their dead
entries when their size reaches a multiple of 1,024.

The canonical order decides what the digest leaves open, and the lift of a body into terms reads that order: sibling
regions are addressed by position in every recorded cut decision, and a product's operand order is its orientation
where neither operand owns an output axis. `normalize/ties.rs` states each rule with its reason: siblings a scope may
place either way sort by statement type name, then by their fields in declaration order with names abstracted and a
one-element tuple after longer ones, so every store follows the values it reads and an `Assign`'s operation decides
before its dtype; sibling loops sort by a digest of their extent; a commutative operation lists accumulators, then
loads and the copies that widen them, then computed values with the deeper cone first, then by value. Changing a rule
moves identities: a restamp decision.
The order permutes and names, and the operand order inside a value reads the ranks the order gives, which the order in
turn spells by; so `normalize_uncached` runs its rounds until one whole round leaves the body alone, and refuses a body
that does not settle in eight. For that to settle, a number must not read what the order decides: the numbering keys a
chain of one associative commutative operation as one node over every term (`Chains::Flat`), so the two associations
the term order can give a sum number alike; the exact identity is keyed the same way. The clustered identity keeps the
chain (`Chains::Kept`): it spells every FMA-class tree as a chain of one representative, and the shape of that chain is
what tells `a * b - c` from `a * c - b` when the buffers are typed.

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