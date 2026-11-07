# The statement crate's next step: layout, names, storage, one scalar definition

Written 2026-11-07, after the port (#1365) reached finalization. It is the first refactoring step after that PR lands
and runs as one follow-up PR on `main`, in five phases, each its own commit series. Nothing here touches the port
branch while its finalization is in flight.

Goal: a reader of `crates/emmy-ir` finds an axis, a dtype or an elementwise operation where its name says it is; the
concepts carry their descriptive names (a **statement type**, an **operation**, an **elementwise op**) in the crate,
the extension, the Python glue, the tests and the documentation; a statement's leaf fields are stored by value
where a shared pointer buys nothing, with names interned; and scalar arithmetic has one home: an expression, over
the one operation table, embedded in the statement that uses it. A statement exists for an effect or for a shared
name; a pure scalar value used once is an expression. `Assign` is the one scalar definition and `Affine` goes.

Purity of the design is the priority of this plan. A restamp, a refit and a re-key of every golden are acceptable
costs; a second spelling of one concept is not.

Three phases change where things sit and what they are called, and nothing else; the fourth changes how leaves are
stored and is measured; the fifth changes the IR. Phases 1 to 4 move no stored text, identity, golden or corpus
case, and that is the check on each of their commits. Phase 5 changes the normal form, so every body re-keys; it
pays for that with one restamp, one record run per card and one refit, taken once, at the end.

## Why

**Layout.** The `kinds/` module holds the statement declarations and their macros, and beside them the types a
statement's fields are made of: `axis.rs` (`Dim`, `Window`, `Axis` and their spellers), `dtype.rs` (the `DType`
closed set and promotion), `elementwise.rs` (the operation table, the stored op, the aliases) and `field.rs` (the
`Field` trait, `Visitor`, `Rewriter`, `NumOr`, `Float`, the field value types). They are there because each
implements `Field`. From outside, a module named after statement types is the last place to look for an axis, and
the file list makes the module look like it holds more than it declares. The crate's layering note already describes
them as the layer below the declarations; the layout should say the same.

**Names.** "Kind" names the type a statement belongs to (Loop, Load, Accum) as opposed to a statement, one instance
in a body. Both need a name and the code uses both (`Statement` holds instances, `Kind` names the types). The Python
module used the word before the port and the port kept it. The owner wants the descriptive name: **statement type**,
and "expression type" for the eight expression variants. The method that answers it is `ty()`, the Rust compiler's
abbreviation, on statements and expressions alike; the three enums read as a set: `StatementType`, `ExprType`,
`FieldType`. In the operation table, `Elementwise` is an adjective standing for the enum of known operations and `Op`
is what a statement stores; the crate has no graph ops, so the broad word is free: `Operation` and `ElementwiseOp`.

**Storage.** The data is immutable and a rewrite builds new values; the `Arc`s are the structural sharing that keeps
that cheap. A rewrite copies the path, not the tree; a walk returns the object it was given when nothing changed, so
twenty-nine sites answer "did anything change" with a pointer compare; Python wrappers point at shared nodes; the
memo stores one normal form for many bodies. That argument holds for trees (`Body`, `Statement`, an expression
`Node`) and not for leaves. A name is an `Arc<str>` (74 fields hold one, 17 hold a tuple of them, 10 an optional
one): every comparison is a string compare, every hash a string hash, and the value numbering, the canonical order
and the ties compare names constantly. Small fixed leaves (`Ints`, `Floats`, an `AxisRef`, the two-element `Names`)
are heap allocations whose sharing saves a few words. These are measured changes, not convention changes, so the
phase starts with a profile and keeps what pays.

**One scalar definition.** The IR spells pure scalar arithmetic three ways. An index (a buffer index, a loop's
start and step, a predicate, a slot, a guard) is an expression tree embedded in its statement. A value computed from
SSA values is a statement, and there are two of them: `Assign` (`name = op(args)`, one operation) and `Affine`
(`name = offset + Σ coefficient·arg`, a flat n-ary sum with its own fold, because a binary tree gave sums no canonical
form). The expression module already has the canonical sum (`affine_form`), its simplifier already knows which
operations are affine (the table's affine role), and the expression operators (`BinOp`) duplicate the operation table
under other spellings. The rule that makes it one concept: **a statement exists for an effect or for a shared name;
a pure scalar value with one reader is an expression.** `Assign` becomes `name = expression` over the table, the one
scalar definition; `Affine` goes; the normal form folds a single-use `Assign` into its reader's expression and keeps a
multi-use one as the statement that names the shared value; the affine form is the canonical form of every sum,
wherever it sits. Value numbering keeps merging equal definitions, placement keeps hoisting them, and the lowering's
pattern matchers read expression shapes instead of statement chains. Index arithmetic is already in that shape; this
phase brings value arithmetic to it.

## Phase 1: layout

Crate root, in tower order (each module imports only the ones before it; `field`, `statement` and `body` define each
other's types, as `kinds` and `body` do today):

| Module | Holds | Today |
| --- | --- | --- |
| `expr/` | the expression tree, its text, simplifier and reader | unchanged |
| `axis.rs` | `Dim`, `Window`, `Axis`, `spell_axis`, `spell_window`, `DEFAULT_SEQ_HINT` | `kinds/axis.rs` |
| `dtype.rs` | the `DType` closed set, `promote` | `kinds/dtype.rs` |
| `elementwise.rs` | the operation table and its macro, the stored op, the aliases, `AffineRole`, `Eval` | `kinds/elementwise.rs` |
| `field.rs` | the `Field` trait, the field value types and aliases | `kinds/field.rs` |
| `statement.rs` | the declaration list, its macros, the structs, `Statement`, the type-tag enum, `Roles`, the flags, `Layer`, `FieldInfo` | `kinds/mod.rs` |
| `body.rs` … `memo.rs` | unchanged | |

`protocol/kinds.rs` (what the shared statement types answer beyond their fields) becomes `protocol/shared.rs`.
`closed_set!` stays in `lib.rs`; `operations!` stays beside its table. The extension crate (`crates/emmy-ir-py`)
keeps its three files; its imports follow the moves. `git mv` the files, declare the new modules in `lib.rs`, and let
every `crate::kinds::` and `emmy_ir::kinds::` path follow (over a hundred). No identifier changes in this phase.

## Phase 2: names

Every identifier below changes. The spelling a type-tag's `name()` returns (`"Loop"`, `"Load"`) is the Python class
name and stays. So do the operation spellings (`"add"`, `"maximum"`), the tables the extension exports
(`OP_IDENTITY`, `OP_CLUSTERS`, `OP_ALIASES`), the frozen spellings in `normalize/ties.rs` and every string the text
form or a digest reads.

| Today | After |
| --- | --- |
| `Kind` (the type-tag enum), `Statement::kind()` | `StatementType`, `Statement::ty()` |
| `Ty` (`field.rs`, what a field's value reads back as) | `FieldType`, which frees `ty` for the statement's type |
| `KindFlags`, `Statement::flags()` | `StatementFlags`, unchanged method |
| `with_kinds!`, `define_kinds!` | `with_statement_types!`, `define_statement_types!` |
| `define_py_kinds!`, `register_kinds`, `kind_of`, `wrap_kind` (extension) | `define_py_statement_types!`, `register_statement_types`, `statement_type_of`, `wrap_statement` |
| `Body::of_kinds(&[Kind])` | `Body::of_types(&[StatementType])` |
| `carried_kind` (`protocol/checks.rs`) | `carried_type` |
| `ExprKind`, `Expr::kind()` | `ExprType`, `Expr::ty()` |
| `kind_error` (`emmy-ir-py/src/expression.rs`) | `expr_type_error` |
| `Elementwise` (the enum of known operations) | `Operation` |
| `Op` (what a statement stores: a known operation or another spelling), `Ops` | `ElementwiseOp`, `ElementwiseOps` |
| `OpInfo` (one row of the property table) | `OperationInfo` |
| crate test names holding `kind` (`every_kind_has_a_text`, …) | the same names with `statement_type` / `expression_type` |

Not renamed, because they are not statement types: `TokenKind` (the expression lexer), `ErrorKind`, `SignalKind` and
`AtomKind` (Python, the tile IR's atoms). If the owner wants the word gone from the crate altogether, `TokenKind`
becomes `TokenType` in this phase; the other three are outside the statement module. The Python-visible surface does
not change: no class, method or constant the extension exports carries the word today. `cargo clippy -D warnings` is
the guide to what a rename missed.

## Phase 3: words

Comments and doc comments in the crate and the extension; `fields_of`'s docstring and the `stmt` package docstring;
`tests/compiler/ir/stmt/test_protocol.py` and `test_text.py` (the `_kinds()` helper, `_count_kind`, and the test
names `test_every_kind_reads_back_from_its_text…`, `test_every_kernel_ir_kind_has_a_text`,
`test_every_exported_kind_has_an_example`, `test_a_kind_holds_its_elementwise_operation_as_op`,
`test_a_kind_check_is_native`, `test_lambda_rejects_impure_stmt_kinds`). The other Python files that say "kind"
(`ir/base.py`'s "op kind", `ir/atom.py`, the tile IR's "node kind") speak of graph ops, atoms and tile nodes, not
statements, and stay.

Documentation: `crates/emmy-ir/ARCHITECTURE.md` (27 lines say kind; its Layout and Data sections are rewritten to
the tables here), `emmy/compiler/ir/ARCHITECTURE.md` (the statement-module interface, the protocol, "a declared
kind", the text section, the value-numbering lines and the Kernel IR section: 30 of its 37 lines; the tile IR's "node
kind" lines stay), the two lines in `emmy/compiler/ir/tile/ARCHITECTURE.md` and
`emmy/compiler/ir/schedule/ARCHITECTURE.md` that mean a statement type, and a **Statement type** entry in
`GLOSSARY.md`, which defines neither word today:

> **Statement type** — The type a Loop IR statement belongs to: `Loop`, `Load`, `Accum`, … Forty-five are declared in
> one list in the `emmy-ir` crate, fifteen shared and thirty of the Kernel IR; each is a Rust struct and a Python class
> of the same name. A statement is one instance of its type in a body.

After this phase `grep -rniw kind crates/emmy-ir crates/emmy-ir-py emmy/compiler/ir tests/compiler/ir tests/architecture`
returns the lexer (`TokenKind`), the `FragmentArg` field named `kind` (stored text and a Python field, so it keeps its
name), the tile IR's node kinds and the PTX `.kind::` qualifiers, and nothing else.

## Phase 4: storage

Measured, on the A100 golden's fresh lowering (the port's own benchmark: parse 0.07 s, normalize 1.82 s) and on the
identity of every corpus case: wall time and allocation count (a counting allocator behind a feature flag or a test
binary), before and after each change. A change stays if it is faster or equal with less code; otherwise it is
reverted in the same commit series and the measurement goes in the PR body.

1. **Interned names.** `Name` becomes a `u32` symbol into one process-wide interner (`lasso`'s threaded table, or a
   `RwLock<HashSet>` of our own if the crate pulls more than it gives). Equality and hashing are integer; `as_str()`
   and `Display` resolve; the extension interns on the way in and resolves on the way out, so Python still sees
   strings. Two rules keep identity: a digest spells a name by its text, never its symbol (symbols are
   process-local), and the canonical order sorts names by text where it sorts them today. The interner never frees,
   and a fresh lowering mints names a body at a time; measure the distinct-name count over the A100 lowering and a
   long `emmy compile` before committing to it. If the table's growth is a problem, names stay `Arc<str>` and the
   phase records why.
2. **Small leaves by value.** `Ints` and `Floats` (three and one fields, a handful of values) become inline arrays or
   a small-vector; `Shape3` and `Int2` already are. `AxisRef` (two fields) and `Axes` (one) are candidates once
   `Axis` holds a symbol and a `Dim`; so is a two-element `Names`. The rule: a clone of a rebuilt parent's unchanged
   field must stay a few words with no allocation, which `Arc` guarantees and a `Box` does not, so a by-value leaf
   is inline or it stays shared.
3. **Trees stay shared.** `Body`, `Statement`, `Expression` (`Arc<Node>`), `Stmts`, `Exprs` and the memo keep their
   `Arc`s; the pointer-compare change detection and the Python wrappers depend on them.

## Phase 5: one scalar definition

Measured first: over every stored body (goldens, corpus, freeze), the operation and node types that appear in each
expression field and in `Assign` and `Affine` values, the share of `Assign`s read once, and the body size before and
after the fold. The class each field admits is a design decision, stated below; a stored body outside it is a
lowering that emits the wrong thing, a finding to fix in the lowering, never a reason to widen the class.

1. **The operation table absorbs the expression operators.** `BinOp` (`+`, `-`, `*`, `/`, `//`, `%`, the
   comparisons, the logical and bitwise operators) goes; a binary expression node holds an `Operation`, and the table
   gains an infix spelling column for the operators that have one, so expression text is unchanged. The affine role
   is already a column (`Sum`, `Difference`, `Negation`, `Scale`, `Quotient`), and so are the properties the classes
   below read, so no second operation enum exists in the crate. The simplifier's rules match on properties, as the
   normalization steps already do, never on operator variants.
2. **Two classes, one expression type.** A `closed_set!` of classes, each a predicate over the table that every
   operation node in an expression must satisfy:
   - `Index`: integer arithmetic over axes, GPU built-ins, literals and names: the affine operations, floor division
     and remainder by a literal, minimum, maximum, the comparisons, select and cast.
   - `Value`: any operation in the table, over SSA names and literals, with a dtype.
   An expression field declares its class in the one declaration list beside its roles
   (`index: Exprs [EXPRESSION | READS] : Index`); the construction check refuses a value outside it. The affine form
   is not a class: it is the canonical form of a sum in either.
3. **`Assign` is the one scalar definition.** Its fields become `name: Name [DEFINES]`, `value: Expression
   [EXPRESSION | READS] : Value`, `dtype: Option<DType>`; `op` and `args` go. `Affine` is deleted: an affine sum is an
   `Assign` whose value is in affine form. `Accum` keeps its reduction operation and takes its operand as a `Value`
   expression (`acc = op(acc, value)`), so `acc += a * b` is one statement. `Load`, `Select`, `Pre` and `Let` stay
   statements: a buffer read, a branch, a binder are not scalar arithmetic.
4. **The normal form.** Two rules replace `fold_affine`: a pure `Assign` with exactly one reader in its scope folds
   into that reader's expression (the reader may be an `Assign`, an `Accum`, an index or a predicate), and every
   expression is simplified, with sums in affine form, as the simplifier does today for indices. A definition with
   two or more readers stays a statement: sharing is what a statement expresses. Value numbering merges equal
   definitions as before and the canonical order sorts definitions ahead of readers as before; the depth bound on
   expressions holds, so a chain longer than it stays a chain of statements, which is a canonical form too.
5. **The lowering.** 111 `isinstance` checks on `Assign`, `Affine` and `Accum` across the compiler (21 files in the
   pipeline) match statement chains; each becomes a match on the expression shape (`Accum` over a product of two
   loads, a sum of scaled terms). Seventeen compiler files read `Affine`'s `coefficients`; each reads the affine form.
   The renderer emits an `Assign`'s expression as one C expression, which is what nvcc saw after its own folding.
6. **Identity.** Every body re-keys: the fold changes the statement count of almost every kernel. `emmy golden
   restamp` over every repository golden re-keys each kernel from the fresh lowering; rows keep their schedules and
   lose their microseconds; one record run per card measures them again (the `refresh-golden` skill), then both
   priors are refit (AGENTS.md step 22) and the tune DB version is bumped. The corpus is restamped by `make
   test-corpus-regen`; a case whose verdict changes is a finding. The CUDA hash baseline is re-cut, and the perf lane
   compares every corpus case against its stored number, since the fold can change what nvcc is given.

## Verification, after every phase

- `cargo test -p emmy-ir` and `cargo clippy --all-targets -- -D warnings` in both crates; `make lint`.
- The identity baseline over the 258 realization corpus cases is byte-identical to the one at the PR's base, and
  `make test-corpus-regen` writes nothing.
- The text round trip over every stored body (the goldens, the corpus, the freeze) is exact; `emmy golden check` over
  every repository golden says current for every file.
- `./venv/bin/pytest tests/compiler/ir/stmt tests/compiler/ir/test_loop_op.py tests/architecture` green; the
  layering test still finds the single normalization entry point and the `ir/` → pipeline boundary under the new
  module names.
- Phase 4 adds the timing and allocation table, before and after, to the PR body; the timing includes a Python-side
  walk over a lowered body (the renderer's), since an interned name is resolved at every Python read.
- Phase 5 is the exception to the identity, golden and CUDA checks: its checks are the text round trip over every
  converted body, every corpus case keeping its verdict after regen, every golden current after its restamp, the
  correctness lane green over the corpus, and the perf lane's comparison of every corpus case against its stored
  number in the PR body. The checks above still hold for phases 1 to 4 individually, so phase 5 is the last commit
  series, and its restamp is one commit by itself, apart from the code, so a bisect lands on the code change.

The full suite runs once, at finalization, on a rented card or on the dev box with three capped workers
(`AGENTS.md`, "Running Tests"); no phase touches a kernel, so the GPU half is a formality.

## What this step is not

- Phases 1 to 4: not a change to any declaration, role, flag, walk, text clause or normalization rule; a body's text
  and identity before and after are the same bytes.
- Not the moment to split `statement.rs` further (its 1,336 lines are the list and the two macros; a split by layer
  or by role is a separate question) or to touch `body.rs`, `protocol/` beyond the file rename, or the extension's
  layout.
- Phases 1 to 4: not a restamp, a refit or a tune DB version bump; nothing computed from a kernel changes.
- Phase 5: not a new operation enum, not a new statement type, and not a move of index expressions out of the
  statements that use them. The fold is a canonical form (it always applies and decides nothing), so it is a
  normalization step and not a pass.

## Line balance

Phases 1 to 3 are a move and a rename: the core under `emmy/` changes only docstrings and the diff under `crates/` is
paths and names; net lines zero or negative (the glossary entry adds three). Phase 4 may add one dependency and the
interner's resolve calls at the boundary; it must take out at least as much in string handling, or it does not stay.
Phase 5 removes `BinOp`, the `Affine` statement with its text clause and its fold, and `Assign`'s `op`/`args`
encoding, and adds one class set, one check and the single-use fold; the crate comes out smaller. Under `emmy/` the
lowering's matchers change shape without growing: a match on an expression is one match, where a match on a chain
walked the chain.
