# The statement module in Rust: how it maps

Written 2026-11-06. It replaces `plans/stmt-module-interface.md` and the two uncommitted notes that followed it.

Goal: `emmy/compiler/ir/stmt` implemented in Rust behind the interface it has today, with the Python implementation
deleted in the same change that switches to it. The reason is normalization's constant factor: a body is normalized at
every construction and inside every identity, and the trial below measured the hottest step 19 times faster as a
line-by-line translation.

## Decided by the owner

1. **The whole module moves**, the statement kinds and `Body` included, not only the normalization function.
2. **Bit-exactness is relaxed.** The text of a normal form, the order among ties, the digest value and the tune DB
   version may all move. What has to hold is the section "What replaces bit-exactness".
3. **Rust owns the data**, and Python reads it through accessors.
4. **Only basic types cross the boundary**: strings, numbers, booleans, tuples of those, and the module's own Rust
   classes. No Python object is passed in for Rust to hold or call.
5. **No statement kind is ever a Python class on a Rust base**, not even as a step on the way.
6. **Text everywhere.** A body's wire form is its text; tests state bodies as text and assert on printed text; an
   expression field crosses the boundary as its text. No reader of the old wire form exists.
7. **Every kind is a struct in the crate** (option A below), and **the port lands as one PR** that replaces the Python
   implementation: no PR adds Rust beside Python, and nothing is declared in both.
8. **A second extension**, the compiler's own (a name like `emmy-ir`), beside the optional runtime extension.
9. **Rust folds constants.** The kernels whose folded constant moves are counted first and re-recorded.
10. **Kernel IR kinds print and read through their declarations**; every Kernel IR listing may change shape.
11. **An expression's identity is its text**, not its class name, so the identity moves once with the port and never
    again on a rename; the four concrete expression classes take full names then.

## The module when the port starts

The preparation is Python only and changes no kernel. Most of it is merged or in review (#1334 to #1348).

- One package, 37 exported names, 20 of them classes. 71 compiler files and 117 test files import it, and nothing
  imports a submodule. The compiler constructs a statement at 566 sites and tests a statement's kind at 401.
- 5,039 lines: normalization 1,427, value numbering 536, the order and identity 222, `Body` 679, the text 483, the
  kinds 888, the protocol 641, the package file and two helpers 163.
- Every kind is declared: a class under the `@statement` decorator whose fields carry a type and a role
  (`role("defines")`, `role("expression", "reads")`, ...). The decorator makes it a frozen dataclass and writes every
  protocol member from the roles, for all 45 kinds: the 15 shared ones and Kernel IR's 30. Four members no role gives
  are stated on their kinds and pinned by a test.
- A body has one text, read back by `parse_body` to an equal body. It is the wire form in every golden, corpus case
  and freeze, and the language of the statement module's tests.
- Normalization runs each step once and refuses a body that holds a subroutine call; fusion expands its own calls.

## The map

| Python today | Rust | Notes |
| --- | --- | --- |
| A kind: frozen dataclass, fields with `role(...)` | A struct per kind, all in one `enum Statement` | The role table carries over one to one. See "Kinds". |
| The `@statement` decorator: members generated from the roles when the class is defined | A derive on the struct: the same members generated when the crate is built | The decorator is the derive's Python form. |
| `Statement.rewrite`, `rename`, `substitute`, `rename_buffers`, `map_expressions`, `with_bodies` | One generated "rebuild with these parts changed" per kind, and thin functions over it | Today these are six entry points over one field walk. |
| `Body`: a sequence with content equality | A shared, immutable slice of statements | A member read twice from Python is one object. |
| About twenty cached readings on `Body` (`definitions`, `ssa_uses`, `loads`, ...) | Fields of the body's shared data, each computed once on first use | Rust can keep a derived value inside the object; Python cannot see the difference. |
| `Expression` and its subclasses (`ir/expression.py`, outside the module) | The crate's own expression type: parse, print, substitute, simplify, the affine form | Python keeps its class. A field crosses as text. See "Expressions". |
| `Axis`, `Window`, `Dim`, a dtype, an elementwise op | Plain values of the crate | They cross as their text or their name, like an expression. |
| `text.py`: printer and reader | The crate's printer and reader | The stored files do not change. |
| Value numbering, the order, identity (`values.py`, `order.py`, `identity.py`) | Functions of the crate over `&Body` | Statements are numbered by position, so no statement is keyed by object identity. |
| Normalization (`normalize.py`) | Functions of the crate, `Body` in and `Body` out | Same steps, same driver. |
| The normal-form and identity memos, keyed by a body's text | A table inside the crate keyed by the body's hash | Hashing a body of 20 statements again is 15 ns against 2,765 ns today. |
| The structural form (`structural.py`) of a statement | Generated with the kind, feeding the identity digest | The digest value may move (decision 2). |
| `Call`, `Subroutine` | Two more kinds; a subroutine is shared data | Needed only by the one normalization step fusion runs on a body with calls. |
| `Fold`, a Tile IR term that holds `Lambda` bodies | Stays in Python; its `rewrite` is its own entry and it answers no statement member | A Python term holding Rust bodies is the allowed direction of the boundary. |

### Kinds

Every kind is a struct in the crate (decision 7). The 45 declarations are transcribed once as Rust structs, Kernel
IR's included: they are data, a few hundred lines, and the Python declarations are deleted in the same PR. The
extension exports a class per kind with the same name, the same keyword constructor and the same field reads, so the
71 importing files do not change. A layer adds a kind by adding a struct. This is the fastest form and the one the
trial measured. The role of a field can be an attribute on the field or the field's type (a name that is defined and a
name that is read being two types); that choice is made when the crate's first kinds are written.

The alternative, kinds declared in Python and registered with the crate by the decorator, was rejected: it keeps a
Python class per kind, stores every statement as a generic record read by position, and creates classes at run time
under the stable ABI.

### The boundary

- **A field Python reads is held as the Python object too.** A string converted on every read costs 17 ns against 2.4
  today, a tuple of names 38 ns. Held beside the Rust value, a read costs what it costs now. So a statement built from
  Python keeps the objects it was built from, and one built in Rust makes them on first read.
- **A kind check is a native class check**, as today: each kind is its own class.
- **Construction, a copy with changes and equality** are 2 to 6 times faster than the dataclass, and hashing a body
  far more.
- **Pickling works**: test workers and the process pools pass bodies.
- **Nothing calls back.** A walk that calls into Python costs 100 ns a call against 40 ns for a whole statement read
  in Rust, and decision 4 rules it out.

### Expressions

A fifth of normalization runs in expression code the module does not own, so expressions cannot stay behind a call
into Python. The crate has its own expression type and the two meet through text:

- Python builds a statement with its `Expression` objects; the binding takes each one's text and looks the Rust
  expression up in a table. A read hands back the Python object it was built with, or parses the text once.
- 88 files build and match the Python `Expression` and keep doing so.
- Until the Python simplifier calls the Rust one there are two simplifiers. A test that both give the same text over
  every stored expression holds them together.

## What replaces bit-exactness

1. **Stored kernels keep their rows.** `emmy golden check` reports every repository golden as current, and the
   realization corpus needs at most a regeneration with no verdict change. This holds when Rust normalizes a stored
   kernel to what it normalizes that kernel's raw body to.
2. **The same grouping.** Two raw bodies share an exact identity in Rust exactly when they do in Python. Each
   exception is listed with its reason.
3. **The normal form is final.** Normalizing a normal form changes nothing. The test over every stored kernel
   already states this.
4. **Kernels stay correct.** The corpus's `correct` stage on a card.

Rule 1 has one consequence: a folded constant is stored in the kernel as Python folded it, so a fold that differs in
the last bit re-keys that kernel. See the open questions.

## Order of work

Both implementations exist until the switch and never share an object: a body goes from one to the other as text.

### Preparation still open, in Python

| Step | State |
| --- | --- |
| Plain constructors: `Load` and `Write` take their fields as declared; no string, tuple or number is converted after construction | open; 62 compiler sites, and about 1,080 test sites once the tests are text |
| The tests of the passes state bodies as text | open; about 100 files, 3,244 hand-built statements |
| Kernel IR kinds print and read through their declarations | done (decision 10); the thirty hand-written printers are gone |
| Loopify compares statements without listing their fields | done; it reads `fields_of` and rebuilds through `replace` |
| The debug dump stores a body as its text, not as `repr` read back by `eval` | done; the Tile layer's term and output specs still dump as `repr` |
| A fixture that yields the raw bodies of a corpus lowering | open; rules 1 to 3 run over it |
| `Fold` stops answering the statement protocol; the walks that reach one become functions over the interface | done (`Fold` keeps a `rewrite` of its own; nothing else was read) |
| An expression's structural form is its text; the four concrete expression classes take full names | open; moves every identity once (decision 11): the tune DB version and its pinned identities move with it |
| No Tile layer table is keyed on a statement or a body | open; check, convert the caller if one is |
| The kernels whose folded constant would move under a Rust fold are counted | open; decision 9 re-records them |

### 1. Expressions and their text

The crate, pure Rust with no Python in it, so `cargo test` runs it. Its expression type with the printer, the reader,
substitution, simplification and the affine form.

Verify: every expression in every stored file reads and prints back to the same text, and simplifies to the text the
Python simplifier gives.

### 2. Kinds, bodies and the text

The kinds as structs with their roles, `Body`, the printer and the reader. No algorithm yet.

Verify: all 4,158 distinct stored bodies read and print back to the same text; reading and printing time against
Python's.

### 3. The protocol

The generated members: reads, definitions, buffers, expressions, child bodies, and the rebuild behind rename,
substitution and the buffer rename.

Verify: over the stored bodies, each member's answer equals Python's, compared as text; a rename and a substitution
print the same text on both sides.

### 4. Identity

Value numbering, the order's ranks, the structural form and the digest. The numbering that is keyed again for each
coloring of the buffers was prototyped in Python and was correct but slower there, because of its bookkeeping; it
belongs here.

Verify: rule 2 over the raw bodies fixture.

### 5. Normalization

The steps and the driver, then the memo.

Verify: rules 1 and 3 over the fixture and over every repository golden; the time against Python on the A100 hardware
golden's fresh lowering (8.2 s when the one-pass normalization landed).

### 6. The switch

Steps 1 to 5 are built on one branch and land together (decision 7). The extension exports the classes and functions
under the module's 37 names, `emmy/compiler/ir/stmt/__init__.py` re-exports them, and the Python implementation is
deleted in the same PR. The tune DB version moves.

Verify: the full suite on a card; `emmy golden check` on every repository golden; the corpus regenerates with no
verdict change; `git diff --stat main -- emmy/` is negative by about 5,000 lines.

## What the trial measured, 2026-11-05

Branch `trial/stmt-rust-object-model`, never to merge. Six kinds and `Body` as Rust classes with the runtime
extension's settings (PyO3 0.29, stable ABI for Python 3.12).

| Value numbering of 17,941 statements | Time | Against today |
| --- | --- | --- |
| Python today | 236 ms | |
| Rust, first numbering of a body | 29.5 ms | 8.0x |
| Rust, every field already rendered | 12.6 ms | 18.7x |

The grouping was the same on both sides: 6,190 distinct values, no statement disagreeing. The first numbering was
slower only because the trial kept expressions, dtypes and axes as Python objects; with those in Rust that cost does
not exist.

| Operation from Python | Today | Rust class |
| --- | --- | --- |
| Read a string field, held | 2.2 ns | 2.4 ns |
| Read a string field, converted on every read | 2.4 ns | 17.3 ns |
| Kind check, hit / miss | 5.6 / 15.1 ns | 5.4 / 15.4 ns |
| Construct an assign / a load / a loop | 263 / 385 / 258 ns | 127 / 168 / 116 ns |
| Copy with a change | 830 ns | 394 ns |
| Equality of two equal statements | 88 ns | 15 ns |
| Hash a body of 20 again | 2,765 ns | 15 ns |
| Pickle round trip of a real body | 57 us | 66 us |

Measured before the trial, on 2,340 raw bodies: normalization spends 12% of its time in expression code and 7% in the
structural form; a round trip through the wire form changes no normal form.

One hazard met: overwriting the loaded extension file while tests ran crashed two workers. Install a new build by
rename, never by copy over the old file.

## Questions still open

None. The five that were open (where the kinds are declared, one extension or two, constant folding, the Kernel IR
text, the expression class names) are decisions 7 to 11; what each asks for is in the preparation table.

## Not in this plan

- The Python expression class moving to Rust. It stays, and meets the crate through text.
- Any speedup in Python.
