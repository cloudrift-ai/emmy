# IR Dialects

Per-dialect op definitions. A `Graph` (`compiler/graph.py`) hosts nodes
from every dialect; the population shifts as passes run. For the
top-level layer/pass picture see `compiler/ARCHITECTURE.md`.

## Dialects at a glance

| Dialect           | When populated                  | Ops                                                                                                   |
|-------------------|---------------------------------|-------------------------------------------------------------------------------------------------------|
| `base`            | always                          | `Op` (base), `InputOp`, `ConstantOp`                                                                  |
| `frontend/ir`     | after tracing / loader spelling | `LinearOp`, `MatmulOp`, `SdpaOp`, `MeanOp`, layout ops                             |
| `tensor/ir`       | after decomposition             | `ElementwiseOp`, `ReduceOp`, `ScanOp`, `GatherOp`, `ScatterOp`, `IndexMapOp`                          |
| `loop/ir`         | after fusion                    | `LoopOp` + body types (`Load`, `Assign`, `Accum`, `Write`, `Select`, `Loop`, `Axis`)                  |
| `tile/ir`         | after `tile/schedule`           | `TileOp` holding the structural root `op`, output specifications, placement, workers, knobs, a typed classic schedule, and its materialization |
| `kernel/ir`       | after `lowering/kernel`         | `KernelOp` + hardware stmts (`Tile`, `Smem`, `Sync`, `TreeHalve`)                                     |
| `cuda/ir`         | after `lowering/cuda`           | `CudaOp` (rendered `__global__` source)                                                               |

## Pure terms vs statements

Two vocabularies, and exactly one direction between them.

A **statement** (`ir/stmt/`) occupies a position in an instruction stream: it has an order, a
scope, and — for a carrier — a seed the enclosing scope has to declare. A **pure term**
(`ir/pure/`) denotes a value: it binds names, carries an algebra, substitutes and compares up to
α-renaming, and has no position at all. `Lambda`, the `Fold` term and the twist recipes (`ir/pure/twist.py` — a
twisted monoid as data, which `Fold.fuse` fuses a reduce and the reduce it reads into) all live on the term side.

**A pure class is never a `Statement` subclass and never occupies a statement position.** When a term
has to reach the instruction stream it is RENDERED into statements at the point of use — never
spliced in as one. `Fold.merge(other)` is the shape of that: the cross-partition state⊕state
combine IS the fold's stored `combine` applied with its second operand naming the partial being
merged, and it becomes `Assign` rescale temps plus one `Accum` per state component wherever the
lowering needs statements (the REG-tree merge, the cooperative tail, the cross-CTA finalize loop);
the serial step is the same derivation at the injected singleton. There is no `StateMerge` type: the term is the
`Lambda` the fold already stores,
and a rendering function is not a statement type.

The invariant is what stops facts from acquiring a second home. A term that renders itself needs
no private spelling of anything the statements already carry:

- the neutral elements come from `Accum.op.identity` through the ONE identity placement
  (`Loop`'s render / `StridedLoop`'s render), so no `identities` field rides on the term and no
  `Init` seed is emitted beside it;
- the rescale temps arrive as ordinary `Assign`s, so the generic SSA rename, liveness and read
  counters see them with no special `deps()` channel to keep complete.

While that combine travelled as one opaque renderable stmt it needed both, and both were subtly
wrong: its temps were invisible to `rename_ssa_sequential` (patched by a uniquifying overlay in the
rewrite handler) and its seed was a second placement path that `_lift` stripped. The flash
cross-CTA finalize was numerically wrong as a result, and became correct when the combine started
arriving as ordinary statements.

**Derived reads memoize on the immutable term; pickling carries only the stored params.** A term's
expensive derived reads — the structural key, `Fold.deps`, `Lambda.free_names`, the synthesized
`loop`, the normalize fixpoint stamp, the codec's spelling tables — ride the instance, which is what
keeps a walk over a large fused tree linear instead of re-deriving every subtree per ancestor. They
are declared members: a `cached_property` on the type that owns the value, or a field computed in
`__post_init__`, per STYLE.md. `__getstate__` strips them: every memo recomputes after transport,
and an id-keyed cache carried across processes could collide with a fresh object's id. A memo holds
only values derivable from the term — never decisions, never mutable policy.

The `structural.instance_memo` tables are a holdout predating that rule, not a second sanctioned
form. STYLE.md forbids the undeclared memo slot they stash into; new derived reads take a declared
member, and the existing tables are to be converted to one.

**Tile IR stores terms, not statements.** `TileOp` holds the `Fold` term; the typed classic
schedule, materialization, output specifications, and knobs belong to `TileOp`, not the term. So `Fold` lives in
`ir/pure/fold.py` and is not a `Statement`.

**Lambda construction binds every read and orders definitions before uses.** Both checks use the scope-aware
`free_names` analysis, including coordinates in predicates and indices. `Lambda.closing` appends unbound reads as
parameters; direct construction rejects them. Canonical ordering uses the same dependencies, so a predicate cannot
move before its definition. The narrower `Body.ssa_uses` query follows statement `deps()` and does not include every
expression read.

Coordinate substitution through a `Fold` preserves its iteration and operand-binding parameter prefix. An environment
parameter replaced by an expression leaves that tail only if the replacements no longer read it. Existing coordinates
keep their order; `Lambda.closing` adds newly introduced free coordinates only for expression substitutions. Pure
renaming preserves the established tail. A nested reduction keeps its own binder; substitution cannot capture it.

## Classic schedule model

The [schedule package](schedule/ARCHITECTURE.md) separates schedule-wide interfaces and reusable choices from concrete
implementations. `schedule/classic/` owns the semantic model for the ordinary grid/CTA/warp/thread/register schedule.
The problem compatibility composes against is the unscheduled `TileOp` itself, paired with a target.
The `TileOp` assigns one stable integer node id to each Fold identity and one `(consumer, operand)` edge site to every
consumer operand position, so a shared producer is scheduled once while each use receives an independent transport
choice. It also derives each node site's projection or reduction view, and each contraction's schedule-independent
`ContractionFacts`, from the Fold alone; target facts cannot affect whether a site is a projection, reduction, or
contraction-capable reduction, nor what its K axis, cone seam, producer, or fragment need are. `ClassicScheduleContext` composes compatibility
over those sites and the separately projected node and edge domains.

`Schedule` is the immutable, generic binding of kernel, node, and edge choices. Direct work, flat raster, untiled
nodes, serial reductions, and direct edges are explicit values rather than missing fields. Choice values never carry
site identities, paths, target facts, encodings, or materialization results: `Tile` is axis-free and `Stage` contains
no slab names or resolved K chunk. The wire codec is injective: empty `TILE` means only per-cell, while a parallel
unit-register thread tile spells `f1`; `WORK` never changes the meaning of an empty node choice.
`ClassicMaterialization` separately maps accepted sites to `PlacedTile`
geometry and `ResolvedStage` transport facts. Construction enforces the value types; completeness, site scope,
node-sum agreement, worker inventory and thread limits, producer-band/TMA agreement, raster eligibility, target
choice availability, and current per-contraction transport agreement need the problem and are enforced by
`ClassicScheduleContext`. Every enumeration and decode leaf crosses that complete-schedule boundary exactly
once before search or lowering can observe it. A validated leaf retains its canonical codec row; inspection and
materialization reuse that row and typed schedule instead of repeating the compatibility walk. Encoding an arbitrary
schedule remains a validating public boundary.

Graph reconstruction is the staged exception forced by the wire dependency: a private codec step parses typed schedule
values and checks their canonical spelling, those values identify the separately encoded materialization sites, and
constructing the complete `TileOp` then performs the one context validation. Parsing alone is never an acceptance
boundary.

A schedule problem is factored into sites, one per node and the kernel site last; each site projects its factor from
static offers and the knob row, never from another selected choice. Their Cartesian product is the definition of the
candidate space. For unscheduled Fold program `p`, target `t` and row, Algorithm 1 is exactly:

    D(p, t, row) = K × ∏ N(node) × ∏ E(edge)
    Algorithm 1(p, t, row) = {a ∈ D(p, t, row) | extend(c + p + t, a) succeeds}

The one `ClassicScheduleContext` is the immutable `c + p + t` prefix. It owns compatibility state only and composes
each node and its incident edges, followed by the kernel factor, through `extend`; the row it never sees, because a
site the row names offers the row's value alone. The generic enumerator never imports classic scheduling. The context
retains derived physical-axis and fragment-seam facts outside the choice values. Domain membership uses the sites'
immutable indexes; local support is derived only after the context has selected one node and its incident edge
values, so a row that names a site does not construct the rest of the relation. Production may prune only prefixes
whose `c + p + t` state proves they have no completion.
The literal reference enumerator remains the oracle. Bounded-product checks and traversal-order tests require every
traversal order to produce the same complete set; the lowering implementation must satisfy that product contract.

A composed step — flash's `Σ Q·K` ahead of its `Σ_j P·V`, split-K's sliced contraction — used to be
the argument for `Statement`-hood: it has to appear at a POSITION in the emitted step stream. It does not
need to be a statement to get there. The tree already carries it: a composed node is an entry in
`operands`, and its position is produced by the derivation — `Fold.lower` places every term of the
tree at the shallowest scope binding its free coordinates, operands ahead of the term that reads
them. A loop-invariant scale `Load` ahead of attention's score contraction is that rule at work, not
a special case: it reads no coordinate the reduce loop binds, so it lands ahead of the loop even
though the cone that reads it rides inside. The only place terms become statements is `Fold.lower()`.

`Fold` answers no part of the statement protocol. Its `rewrite()` is the term's own entry, spelled like the
statement's so one caller renames or substitutes either vocabulary, but a body refuses a term and no statement walk
reaches one; no render function is registered for `Fold`. Impure computation stays in Loop IR until total lift; there
is no impure `Lambda` construction path.

**`nested()` is the STATEMENT protocol, and it deliberately does not reach a Fold's operand edges**
— it yields the lift body, and nothing at all for a contraction, whose algebra is meant to read as
edges rather than body deps. Every walk built on it (`Body.iter`, and so `Body.loads` /
`Body.writes`) therefore answers for a fully flattened stream only. The STORED tree is not one:
its operand edges are terms, and a statement walk over a lift body alone silently under-reports
everything beneath them.
Ask `loaded_buffers` instead whenever the answer must cover what a consumer of the STORED tree will
reach — the kernel materializer walks that tree, so anything deciding a node's graph inputs has to
see what it sees. Asking the lowered view there is what let a cut declare fewer inputs than the
kernel it produced went on to read.

## Invariants by stage

- **Frontend → tensor** (after `decomposition`): `LinearOp`, `MatmulOp`,
  `SdpaOp`, `MeanOp`, and the layout ops are gone. Only
  `ElementwiseOp`, `ReduceOp`, `IndexMapOp`, scan/gather/scatter, plus
  boundaries survive. (The broadcast-explicit invariant for
  `ElementwiseOp` inputs lives in `compiler/ARCHITECTURE.md`.)
- **Tensor → loop** (after `fusion`): only `LoopOp` + boundaries.
  Tensor-IR ops survive only *inside* `LoopOp.body` as the operation nodes of
  `Assign` values and as `Accum.op` (`ElementwiseOp` only — `ReduceOp` is not a valid body
  op; reductions are `Accum` statements inside a reduce `Loop`). `LoopOp` construction orders a free-loop chain by
  the row-major coordinate depth in its boundary writes; axis spelling is only the fallback when output storage does
  not totally order the chain. The resulting geometry, rather than source names, reaches Tile IR placement.
- **Loop → tile** (after the `tile/` passes): `LoopOp` nodes are replaced by
  `TileOp` holding the structural-IR root `op` directly (`tile/ir` — one `Fold` kind), structural
  placement, one accepted site-indexed `Schedule`, and separate `ClassicMaterialization`
  facts. A kernel's structure is read from each node's derived classification, not a Python kernel
  type. `010_lift` lifts the `Fold` tree (the loop nest reconstructed on demand; a `Loop` carries no
  annotation, it folds iff its body carries an `Accum`; the algebra is the term's own
  `lift` / `(init, combine)`) with an UNMAPPED `Placement`;
  The tile schedule maps the free axes onto the grid and decides the reduce `Reduce` via the single
  `REDUCE` codec knob (`g<n>` cta / `coop` (its width in `WORK`) / `r<n>` reg; the
  decision hierarchy = env pin > the deploy evidence hierarchy, and nothing
  else — there is no default partition). The knob is ephemeral — resolved here
  into the typed schedule's `Reduce`; the combine stays the `Fold` node's stored program. Any static
  `PLANAR` / `TWISTED` reduce is cooperation-eligible (degenerate
  `sum`/`max`/`mean` AND twisted online-softmax, scalar AND
  full-row outputs), and the schedule enumerates the serial fold beside every
  band the reduce extent can feed, whatever the grid measures. The one exception is a split's finalize, whose merge axis
  windows its whole parent: one partial per split per cell, serial only — its parallelism is the cells.
- **Tile → kernel** (after `lowering/kernel`): `TileOp` materialized to
  `KernelOp` whose body is a `Tile` (the thread-grid decode) over the
  lowered op tree. A cooperative `Reduce` lowers the reduce as a
  `StridedLoop` (lane-strided fold) + the derived algebra-generic
  cross-thread combine (`_factor.emit_combine`, reading the fold node's
  stored combine → `WarpShuffle` /
  `Smem`+`Sync`+`TreeHalve`, multi-component for a twisted fold) +
  the projection (a full-row output sweep distributed across the coop
  lanes, a scalar output guarded to lane 0); the `Tile` gains the coop
  lane axis and `block_threads = coop`. A **symbolic reduce axis**
  (dynamic `seq_len`) is supported — the `StridedLoop`'s `< seq_len`
  bound is the runtime-extent mask (idle lanes fold the identity; no
  ceil-div / clamp) and the `Dim` name is threaded as a runtime `int`
  arg. Cross-CTA reduction splits are structural choices in `030_cut`. A symbolic FREE axis
  (dynamic grid), strided rows, and the tensor-core `warp_tile` are reserved future tiers.
- **Kernel → CUDA** (after `lowering/cuda`): `KernelOp` replaced by
  `CudaOp` carrying rendered source.
- **Every buffer index spells every dim of its buffer**, `0` at a size-one dim. `render_index` flattens it
  row-major for both the CUDA render and the Loop IR C++ runner, and refuses any other count: without strides the
  coordinates cannot be summed into an address. A single coordinate is an already-flat address.

Multi-output ABI order always comes from `Node.buffer_names()`: matcher population reorders a body-carrying op's
input/output maps to the graph ports after body normalization, Loop execution returns outputs in that order,
`010_lift` copies every port onto Tile IR, and Kernel/CUDA lowering renders every `OutputSpec` and output pointer.
Independent body placement may reorder sibling writes without changing the ABI.

`Op.source` is the rewrite-chain predecessor — the engine's
`_apply_one` stamps it on every 1:1 in-place rebind, so a fully
lowered `CudaOp` carries the full chain back to its originating
`LoopOp` (`cuda.source.source.source`) without any rule needing to
pass it explicitly. The base-class field is keyword-only and
`compare=False`, so subclass positional construction and equality
keep working unchanged. `source` is excluded from
`Graph.structural_key` and from the variant key (`identity_key(with_io=True, with_knobs=True)`) — kernels rendered
along different lowering paths still dedup in the tuning cache.

**A statement is immutable data** — every concrete Loop-IR / Tile-IR / Kernel-IR statement (`Loop`, `Cond`, leaves,
`Tile`, `Smem`, `Sync`, `CpAsyncCopy`, `TmaDescriptor`, …) has named read-only fields, compares by content and is
hashable. To "edit" one, take a copy with `stmt.replace(field=value)`; nothing outside `ir/stmt/` calls
`dataclasses.replace` on a statement or lists its fields. A statement
type states its fields to the text reader and the structural form only. A pass that generalizes whole statements (the
kernel loopify pass, off unless pinned, rolls repeated ones into a loop) reads each field by the name `fields_of` gives
and rebuilds through `replace`, so it needs no dataclass behind the statement type. Ops, by contrast, are frozen and
unhashable — rewrites replace the op and rebind its graph node. Op fields stored inside statements (`Accum.op`) must be
lightweight value objects (`ElementwiseImpl`, not `ElementwiseOp`) so the surrounding statement stays hashable.

## The statement module's interface

`ir/stmt/` is the vocabulary all three IRs share, so the rest of the compiler uses it through one package and nothing
else: no submodule path, no private name. The package exports the statement types, `Body` and `OutputSpec`,
`normalize_body` and `canonicalize_identity`, a few functions over statements (`free_names`, `exposed_defines`,
`rename_free`, `mask_select_predicate`, `ordering_constraints`, `dtype_promote`), the text form (`pretty_body`,
`parse_body`), four normalization steps a caller may run on its own (`one_operation_per_definition`,
`canonical_values`, `simplify_body`, `prepare_body`), the two statement types fusion's body assembly in `ir/loop/`
builds a region with (`Call`, `Subroutine`), and the names of a statement type's fields (`fields_of`). The statement
types, `Body`, the protocol, the text, the value numbering and the normalization are implemented in Rust — the
`emmy-ir` crate, reached through the `emmy.emmy_ir` extension (`crates/emmy-ir/ARCHITECTURE.md`) — and the package is
the glue over them: `__init__.py` re-exports the names, and `glue.py` writes the few members that take a Python
function and keep its results (`rename` by a function, `rewrite` under a `Sigma`, `Body.fold`, the identity with its
typed roles); `Body.map` and `map_expressions` call their function from Rust. The package imports expressions, axes,
the sigma and the wire form: no op, tensor, graph, backend or pipeline. `tests/architecture/test_layering.py` pins the
export list and both directions.

**The statement protocol** is what a statement type owes, all of it methods on the statement type. No statement type
writes them: the shared statement types are declared (below), and a layer adds a statement type of its own by declaring
it, and nothing else:

| Group | Members |
| --- | --- |
| Traits | `pure`, `deps_deep`, `has_side_effects` |
| Reads and writes | `deps`, `defines`, `external_reads`, `external_writes`, `local_decls`, `bound_axes`, `binds_axes` |
| Structure | `nested` with `with_bodies`; `expressions` with `map_expressions` |
| Rebuilding | `_rewrite` (behind `rewrite`, `rename` and `substitute`), `rename_buffers`, `replace` |
| Display and storage | the text, a statement type of another layer through its declaration; a body's text is its wire form |

Each reading member has a rebuilding half: `with_bodies` takes what `nested` returned, and `map_expressions` applies a
function to each expression `expressions` lists, in that order. `bound_axes` are the axes a statement binds and
`binds_axes` the names it brings into scope, which are those axes' names and any other name it hoists. `rewrite` is the
one entry for renaming and substitution; it drops the substitutions a statement's own binder shadows and hands the rest
to the statement type's `_rewrite`. `tests/compiler/ir/stmt/test_protocol.py` checks that every statement type in every
layer states the protocol, and the rebuilding halves on an example of each shared statement type and each Kernel IR
statement type.

**A declared statement type** is a statement type stated as data: a struct in the crate's one declaration list
(`with_statement_types!`) whose fields carry a type and the roles that say what the field is to the rest of the
compiler. The roles are names the statement defines, reads, updates in place or binds for its own use; buffers it reads,
writes or declares kernel-local; expressions; axes it binds and names that follow them; a child body; statements it
carries; the body of a pure term; and records whose own fields play roles. A field with no role is plain data. The crate
writes every protocol member from the roles once, and the extension generates the Python class of every statement type
from the same list, so a declared statement type writes none and no statement type is ever written as a Python class:
what it reads is its `reads` and `updates` fields, a rename maps every name field and every expression, a buffer rename
maps the buffer fields, and so on. Two roles hold statements that are not a child body. Carried statements (`stmts`: the
cells a `RegStore` stores with it, the combine of a `TreeHalve`) are part of the statement: they are rewritten with it
and their buffers, expressions and free reads are its own, a name one of them defines or the statement binds being
hidden. A term (`term`: a `RegStore`'s epilogue) keeps its statements' names private: a rename reaches only the
coordinates its expressions read. An expression field that is also a `reads` field is read through: the names its
expressions mention are names the statement reads (a `Load`'s index, a `Select`'s predicates, a `Cond`'s condition). A
number where a name, a buffer or an expression is expected is plain data (a constant stride, a literal seed).

The shared statement types and the Kernel IR statement types are all declared, in one list. A statement type whose
member the roles cannot give is a missing role, not a hand-written member; the shared statement types state four, which
`test_protocol.py` pins: an `Accum`'s reduced axes follow a substitution that splits an axis, a `Let` and a `Call` read
buffers their fields do not name (inside an expression, through the subroutine), and a `StridedLoop` with an `end` binds
one more name than its axis.

Behaviour that is not on this list lives outside the module as a function over statements. CUDA rendering is the
main case: `ir/render.py` holds the render context, the index flattening and one render function per statement type,
dispatched on the statement's type; Kernel IR registers its own statement types beside their classes.

**`Body`** is an immutable sequence of statements, built from any iterable. A slice or a sum of bodies is a `Body`;
two bodies with equal statements are equal and hash alike; a statement placed in a body is the same object every
time it is read, which passes rely on when they key side tables by statement; it survives the wire form and
pickling. It is not a `tuple` and equals none: of a sequence it has a length, iteration, indexing and slicing, and a
field that holds one defaults to a `Body`. Its identity reads an axis's window, which equality leaves out: two bodies
may be equal and key apart. Its executable normal form is cached on that `Body`; the memos shared across bodies (the
normal forms, the identities) are keyed by the body's exact form, which reads the windows, hints and the bits of every
number, never by equality or by text.
Callers use its walks (`iter`, `map`, `fold`, `iter_of_type`,
`loads`, `writes`, `carries`, `accums`, `loops`), its def-use readings (`definitions`, `ssa_uses`, `axis_names`,
`axis_dependencies`, `backward_cone`), its orders (`topological_order`, `coalesce`), `rename_buffers`, and its
identity (`identity`, `structural_key`). The names a body defines are the keys of `definitions`; the names one
statement makes visible are `exposed_defines`.

**A body has one text**, which is also its wire form. `str(body)` is it and `parse_body` reads it back to an equal
body, so the text says
everything a statement holds and nothing derived from it: a field at its default is left out, an axis carries its
window and a symbolic extent its hint. One statement is one line, a block's statements sit indented under its
header, and expressions are in their own text:

```
let f32 v0 = 0.125
for a0 in 0..4:
    for a1 in 0..seq~128:
        in0 = load x[a0, a1]
        v1 = 2*in0 - v0 + 1
        acc0 <- add(acc0, v1) over a1
    y[a0] = acc0
```

A subroutine a call names is defined once, ahead of the statements, and the call spells the buffers it binds. Since
subroutines compare by object, what a text read back keeps is which calls share one.

The printer and the reader sit side by side in one file of the crate, one clause per shared statement type, and its
documentation is the grammar. A statement type of another layer (the Kernel IR's thirty) prints and reads through its
declaration: its class name, the fields not at their default as `field=value` in parentheses, and one indented block per
field that holds statements — `Smem(name=s_x, extents=(64, 64), dtype=f16)`, `Tile(axes=(a0 in 0..64),
block_threads=128):` over its body. Each value is spelled by its type and read by the field's declared type, so a layer
that adds a statement type adds no printer and no reader. The crate's text tests hold both directions over one example
of every statement type, and `tests/compiler/ir/stmt/test_text.py` over every corpus kernel and the Kernel IR of six
corpus lowerings.

The tests under `tests/compiler/ir/stmt/` state what goes into the module and what comes out, and hold for any
implementation of it. The ones under `tests/compiler/ir/stmt/implementation/` read this implementation's own
summaries and memo and leave with it. The inner steps of normalization are tested beside their Rust, in
`crates/emmy-ir` (`cargo test -p emmy-ir`); the extension exports the driver and the steps a pass calls, no other.

## `base.py`

Cross-cutting root. Imported by every dialect, imports nothing from
them.

| Symbol          | Role                                                                           |
|-----------------|--------------------------------------------------------------------------------|
| `Op`            | Base class. Subclasses implement `infer_output_shape` and `forward` (numpy).   |
| `InputOp`       | Sentinel: graph input tensor. Value supplied by the executor.                  |
| `ConstantOp`    | Sentinel: weights / scalar constants. Scalars carry `value`; tensors carry `source_path` / `source_shape` / `source_dtype` (the safetensors / `nn.Module` address) plus `load_ops` — a chain of frontend ops applied at bind time by the loader. `source_parts` is the multi-source alternative: `(path, shape)` pairs the loader reads and concatenates along axis 0 before running the chain. `source_graph` is the N-source bind record (`032_fold_constant_subgraphs`' collapsed static cone): a mini-graph whose external leaves name source paths and whose scalar leaves carry values — the loader binds/evaluates it through the NumPy backend, then runs the chain. Exactly one of `source_path` / `source_parts` / `source_graph` is set on a loadable constant. |
| `_keepdim_axis` | Shape helper shared by `ReduceOp` (tensor) and `MeanOp` (frontend).            |

## `expression.py`

Shared expression sublanguage used by every IR layer: `Load.index`,
`Write.index`, `SelectBranch.select`, `IndexMapOp.coord_map`,
`StridedLoop.start`/`step`, `Cond.cond`, etc. Imports nothing from
other IR files.

An expression has one text, `str(expression)`, and `parse` reads it back to an equal expression: C-like, parenthesized
only where precedence needs it (`a5 / 128 * 128 + a6`). The same text is the expression on the wire, in every listing
and in every error message, so there is no second printer to keep in agreement. A name that is no identifier, or that
the reader would take for something else (`true`, a GPU built-in), is written between backticks. Text that holds
expressions among other things is read by a subclass of `Parser`, which takes each expression where it stands.

| Symbol                                                                       | Role                                                                     |
|------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `Var`, `Literal`, `BinaryExpression`, `Builtin`, `FunctionCallExpression`, `TernaryExpression`, `CastExpression`, `FlatIndex` | Expression nodes, implemented in the `emmy-ir` crate and exported here. An operation over its operands is one node in the crate; `BinaryExpression` (an operator over two operands: `op`, `left`, `right`) and `FunctionCallExpression` (`name`, `args`) are the two views Python takes of it. Each has `eval(env) → value/ndarray`, `substitute(mapping)`, `free_vars()`, plus the generic tree walks `subterms()` / `children()` / `rebuild(fn)`, and the Python operators build them. `FlatIndex` is a buffer coordinate's row-major offset, flattened against the buffer's shape at render time; it only appears in a kernel body. |
| `Expression`                                                                       | The base class. Its text is its wire form and its structural form; `repr` is the constructor form the graph's dict form reads back. |
| `PLACEHOLDER_PREFIX`, `placeholder`                                          | Convention for output-coord placeholders in coord maps.                  |
| `affine_terms`, `affine_sum`, `affine_node`                                  | The crate's one affine form: a value's `(coefficient, operand)` terms, offset and whether a float literal fed it, the sum they spell, and whether a node is one of the form (a sum, a difference, a negation, a literal scale or a quotient by a literal). Every reader that takes a sum apart uses these; none carries a list of operation names. |
| `promoted_dtype`                                                             | The dtype a value computes at: the table's promotion of its named operands, node by node; a literal contributes none. |

## `frontend/ir.py`

Ops captured directly from PyTorch. Every one has a decomposition rule
under `pipeline/passes/frontend/decomposition/`; after that pass none of these
remain.

| Group         | Ops                                                                                                      |
|---------------|----------------------------------------------------------------------------------------------------------|
| Layout-only   | `TransposeOp`, `ReshapeOp`, `SliceOp`, `CatOp`, `UnsqueezeOp` — rewrite to `IndexMapOp`.                 |
| Compound math | `LinearOp`, `MatmulOp`, `SdpaOp`, normalization/reduction ops — rewrite to elementwise + reduce chains. |
| Convolutions  | `Conv1dOp`, `Conv2dOp`, `ConvTranspose1dOp` — im2col maps contracted by one `MatmulOp`, or shifted taps.   |

`CatOp` takes any number of tensors: each becomes one `IndexMapOp` source, selected below its end on the cat axis.
Dense `conv1d` and `conv2d` build one im2col map and one GEMM; depthwise `conv1d` sums shifted taps instead.
`conv_transpose1d` lowers in polyphase form: output `q * stride + r` takes phase `r` of a stride-1 convolution over
`K / stride` taps, so one GEMM with `C_out * stride` rows does exactly its multiply-adds and an index map interleaves
the phases. Zero insertion would multiply by zero `stride - 1` times out of `stride`. The tracer captures only these
forms and rejects groups and dilation it cannot lower. A conv2d or conv3d whose kernel covers its whole input — a
vision patch embedding — is captured as the linear layer over flattened patches it computes.

## `tensor/ir.py`

Minimal IR fusion consumes. `IndexMapOp` is the unified layout-only op;
it replaces the frontend layout ops via `coord_map` expressions.

| Symbol                               | Role                                                           |
|--------------------------------------|----------------------------------------------------------------|
| `ElementwiseOp`                      | Per-element scalar function (`add`/`mul`/`where`/`exp`/`sin`/`cos`/…); unary `pad` is the frontend's exact zero-width identity. |
| `CastOp`, `BitcastOp`                | Numeric conversion and same-width bit reinterpretation.       |
| `RangeOp`                            | One-dimensional integer sequence, static or of a symbolic length. |
| `ReduceOp`                           | Collapse one axis via associative binary op.                   |
| `ScanOp`                             | Cumulative variant of reduce.                                  |
| `GatherOp`, `ScatterOp`              | Data-dependent reads / writes.                                 |
| `IndexMapOp` + `IndexSource`         | Unified layout-only op over `Expression`.                            |

`RangeOp`, `CastOp`, and `BitcastOp` are generic value operations, not checkpoint-format operations. Their current
consumer is static reconstruction algebra, so `032_fold_constant_subgraphs` removes them before Loop lifting. A
future runtime consumer may add ordinary lifting for the same semantics without changing checkpoint ingestion.

Op metadata (arity / `commutative` / `associative` / `identity` /
`has_identity` / `selecting` / `semiring_product`) lives on `ElementwiseImpl` in
`ir/elementwise.py` — the single source of truth shared across elementwise,
reduce, scan, and accumulator use sites. The algebraic traits are what
reassociation gates (split-K, cooperative tree-combine) query instead of matching
op names. The per-op trait *properties* (`op.semiring_product` — is this op a `⊗`
in some semiring) and the binary method `⊗.distributes_over(⊕)` (does this product
distribute over that reduce — the `_SEMIRING` table, only `(+, ×)` today) live on
`ElementwiseImpl`; the module's op-name-free **role queries** the planner /
atom-cell matchers ask round them out: `reduce_canon` (alias →
base combine, `sum` → `add` …) and the `_REDUCE_SPELLING` registry
(`reduce_spelling`) — the single op-keyed table
behind the four sites that used to switch on the reduce op name (`Accum`'s render's
`+=` / `*=` / `fmax` / `fmin`, `kernel/ir._binary_combine_expression`, and
`ReduceOp.forward` / `ScanOp.forward`'s numpy reductions). `op.selecting` (the
max/min family) drives the init-placement dtype choice. `op.decodes` names the storage
dtype an op is the decode cast for (the f8 family today) — the trait the tile binding
arm's factor hoist queries instead of matching op names.
Non-ufunc scalar functions whose arity cannot be read from NumPy declare it in the same module; ternary `where` is
the current example. Its condition and both value operands are explicitly broadcast before the elementwise node.

## `loop/`

One `LoopOp` = one GPU kernel described as an SSA program over named
iteration axes. Free vs reduce is inferred from body structure — a
`Loop` is a reduce Loop iff its body holds a carrier, an `Accum`. `is_reduce` (and axis threading
and the other carrier-agnostic checks) test exactly that — the carrier is a plain `Statement` with the
reduce-surface methods, no shared base class. `Accum` exposes `associative` / `commutative` /
`has_identity` traits, forwarded to its scalar `op`. A reduce `Loop` carries no annotation — it
folds iff its body carries an `Accum` — and NO algebra payload (the fold's ⊕ lives on the `Fold` node's
stored `combine`). Commutativity
is unused — split/reorder legality is a future cooperative-tier concern, recorded
structurally when it returns.

A recurrence's state is NOT a fold, and has its own two statements. A fold's meaning is its op: ⊕ is a monoid, so
its steps may run in any order, its seed is the op's identity, and the carrier is one scalar inside the loops over
the cells. A carried state is the opposite on each count: its steps run in order, its step is any computation, its
seed is named, and it has cells — a delta rule's step reads OTHER cells of its own state (`k @ S`), which needs the
loop over the steps OUTSIDE the cells and the state kept across them. `Carry` (`S[i, j] <- next`, with its `seed`)
defines the NEXT value of one cell, and `Pre` (`v = pre S[k, j]`) reads what the PREVIOUS step left at any cell: the
seed on the first step, the last step's value once the loop has closed. No read of a step sees what that step
defines, so the reads and the definition of one step are order-free. Nothing is declared: the state is carried by the
nearest enclosing loop whose axis the `Carry` index does not read (`carried_cells`), which is therefore a reduce loop
the free-axis order never sorts under the cells, and its shape is the loops over its cells. The Loop IR rendering
keeps two slots of the cell shape and commits the second after each step; how many slots survive and where they are
stored is a schedule's question, not the statement's. The Tile lift keeps the state in the term: a `Fold` whose ⊕ is
the action `next`, with the `Carry` index as its `cells` and a `Pre` as a carrier-read slab (`ir/tile/ARCHITECTURE.md`).
The classic schedule realizes the loop as a serial launch axis over a state buffer; the register schedule retains the
state inside a CTA and realizes the same axis as a loop.

**The algebra is in the term, not a tag.** There is no stored / derived `AlgebraKind` and no op-tree node zoo. The
stored tile IR has exactly **ONE node kind**, `Fold` — `reduce(⊕) ∘ map(f)` in the λ-foldMap spelling:

- an OPTIONAL iteration `axis` (`None` = the zero-axis node) — a NAME, derived: the lift's first param when there is a
  combine. A term carries no extent at all: the coordinates it reads are the lift's trailing params, and the extent
  and window of every axis, bound or read, live in the kernel's AXIS TABLE, `TileOp.axes` (the free axes, each
  reduce axis, a split's slice and partition, a sweep). `Fold.lower` takes the table whole — a reduce loop reads its
  axis from it, and only the closed program opens the free coordinates' loops — and every kernel-side reader asks
  `Sched.axis_of` rather than the term. So a term is its function, whatever domain it is evaluated on; a sum over
  128 and one over 256 are one term under two tables, like a slab under two M.
- a pure `lift` `Lambda` `λ(k, v₁…vₙ) → S` — the element's SINGLETON state (ι is spelled in the lift;
  softmax's is `(x, 1)`);
- the monoid's flat `(init, combine)` fields — ONE program, whose results ARE the fold's accumulator names;
- a symmetric tuple of `operands` — the CLOSED inputs, each an edge, bound POSITIONALLY to the lift params. The
  params are the term's OWN names (`Fold.bindings` pairs each with its edge and component); nothing above an edge
  reads how the edge spells its results until the term is rendered — `Fold.applied` is the lift with the binding
  applied, and `step` / `lower` / the projection-region reader emit that form, so a lowered body reads producer
  names throughout while a rewrite that swaps an operand touches no name at all.

**`Map` and `Contraction` are DERIVED READINGS, not stored kinds.** Each is a reading of the stored params:
`axis is None` for the projection, `as_contraction()` (a `ContractionView`, or `None`) for the bilinear one, beside
`as_slab()` and `as_reduction()`. A reading cannot be constructed, subclassed or annotated, which is the point —
there is no type to dispatch on and no second place for a fact to live.

- A ZERO-AXIS fold is what `Map` was: no iteration and no monoid, its `lift` IS the per-cell projection. So
  softmax's normalize and RMSNorm's are one kind composed at two depths.
- The BILINEAR shape — operands `(b₀, a, b₁…)` under a `multiply` lift with a componentwise-additive
  combine — is what `Contraction` was, exposing `a` / `channels` / `b_trans` off `operands`. The roles and the
  orientation are read off the first channel's streamed edge and serve every channel, so a fold whose channels
  read different B spaces, or the same one the other way round (a Gated DeltaNet gate's per-head `X[h, d, k]`
  beside its `W[k, h]` projections), is no contraction and lowers through the generic path. The `⊗` and the
  additive fold `Accum` appear in the DERIVED `Fold.loop`, never as stored loop syntax.
- Every ROLE derives from arity (`Fold.role`, never stored): `FREE` with no axis, `TWISTED` off the combine's
  claiming family, `CONTRACTION` off the bilinear reading alone, `PLANAR` otherwise. `ops.head` reaches the node
  through the projection wrapper. Scheduling reads these facts directly from the Fold tree; `Fold.lower()` is
  reserved for callers that consume Loop IR.
- A SCAN is a fold with a per-step `observe` — a pure `λ(axis, *state)` run after each combine whose fresh results
  only boundary output writes consume. Its schedule preserves every prefix: a serial fold, or a cooperative warp
  prefix scan at a root without a provider chain. Register partials and cross-CTA splits remain unavailable.

`Fold.lower(bound, stores)` flattens the term to the loop nest: a plain loop for every free coordinate the caller left
unbound, outermost the one the most terms share, the reduce loop of each term innermost, every term placed at the
shallowest scope binding its free coordinates, and each boundary store right after the term defining its value. A value
of the tree a term or a store reads by name, such as the expert a gathered row belongs to, counts as the coordinates of
the term that defines it, since no loop binds a value. A term's coordinates there are those its own step reads, directly
or through the operands it reads: an operand it only carries to a store — a multi-output root's per-page and per-slot
results beside a count — takes its own loop beside the others, never one nested in another's. Loops carry NO algebra and
no annotation, so the derived nest depends only on what is stored, which is what makes every identity of the term a
digest of its lowered body — there is no separate term hasher. The `TileOp`'s body identity is the canonical digest of
the nest `lower()` derives (the body is the term's normal form); the variant key (`identity_key(with_io=True,
with_knobs=True)`) folds the schedule-free body identity with the knobs; and the deploy join key (the deploy identity
(`identity_key(with_io=True)`), over `TileOp.loop_body`) types the roles the body reads its buffers through, so term
re-spellings and cluster-sibling ops that lower alike share schedule evidence. A read walk STOPS at a statement whose
**`deps_deep` trait** is set (a `Statement`-protocol member beside `pure`, conservative `False` default; `Call` opts
in): its `deps()` already answers for the whole subtree, scope-correctly, so the walk does not re-walk `nested()`,
whose flat namespace cannot see the subroutine's own params.

A reduce is a contraction not by "two loads" but by the genuine algebra — the lift ⊗
**distributes over** the fold ⊕ (`multiply` over `add`; *not* `add` over `add`, a sum of two
operands) and exposes two distinct free-axis operand roles (`x[m, k]·x[m, k]` is a squared reduce,
not a contraction). Tile IR canonicalization constructs that form from a flat Fold when the
semiring and operand roles prove it; the mma atom tier reads the resulting Fold operands.

**The `Algebra` bundle is retired** — the stored term keeps exactly ONE spelling of ⊕, the
`Fold` node's flat `(init, combine)` pair, and everything else derives where it is consumed.
`Lambda.componentwise` builds a plain fold's combine and `Lambda.components` reads the shape back off any
stored combine (the componentwise op vector, or `None` for a twisted program — no family annotation); the `Fold`
rewrite handler renames the combine through `Lambda.rename` in lockstep with the body, and `Fold.canonical`
renumbers the combine's own names (its second operand, its temps) after the term's, so how a fold spelled its
accumulators never reaches the form. The state⊕state combine's one statement realization is the term's
own `Fold.merge(other)`, of which `Fold.step` is the instance at the injected singleton; the kernel
materializer reads the algebra through `Fold.as_reduction()` (the `ReductionView`: states, the
second operand's names, the terms, the componentwise op vector or `None` for a twisted combine) and
`merge` for cross-thread partitions. A *degenerate* fold is a plain `sum`/`max`/`mean` reduce; a
*twisted* one is online-softmax; a contraction's algebra is the degenerate algebra of its additive
fold.

The neutral element IS stored, as `Fold.init` — a monoid is `(S, ⊕, e)`, and a term that kept only
`combine` would be storing a semigroup while calling it a monoid. What is NOT stored is any
emitter's use of it: a degenerate fold dissolves into its `Accum`s and takes each fold's seed from
its `op.identity`, and a twisted fold's merge derives its own (`Fold.merge` spells the combine as
`Accum`s, never reading the stored `init`'s `−inf`). So `init` is algebra the term owes
its own definition, not a value the lowering path consults — which is why removing it would change every
`structural_key` and, with it, every variant key (`identity_key(with_io=True, with_knobs=True)`) used by tune DB
measurement replay and the cubin cache, in exchange for a field nothing reads.

**The twisted combine — a recipe, never hand-authored on a term.** Transport of structure: a monoid `(·, e)`
conjugated by a bijection ψ gives the twisted combine `x ⊕ y = ψ(ψ⁻¹(x) · ψ⁻¹(y))`, associative because the base
monoid is. A **recipe** (`ir/pure/twist.py`) states exactly that — the base's componentwise ⊕ per state, its
per-element lift, ψ and ψ⁻¹ — and beside the definition stores what conjugation does not give stably: one pattern per
channel (the per-element map a dependent reduce's lift must spell, over ROLES — `exp(s − g)` for a denominator,
`exp(s − g)·v` for an expectation, `(s − g·c)²` for Welford's deviation), what each state is at the singleton (`1`,
`v`, `0`), any state the two-pass form never had (Welford's count and running mean), and the fused ⊕ in its stable
spelling: two lambdas over roles for an open channel count (softmax's pivot advance and the one-sided scale each
channel takes to the advanced pivot, the two sides then joined by the channels' shared ⊕ — one recipe for softmax
and flash attention alike) or one lambda over every state pair (Welford's fixed carrier
`(sum, count, mean, M2)`). `Recipe.program(states)` instantiates either over a fold's state names by renaming, and
the definition certifies the data: the program is the conjugate of the base on random states, the seeds are the base
identities under ψ⁻¹, the injections are the lift seen through ψ. `Fold.fuse(recipe)`
fuses a reduce onto the reduce it reads, found among its operands, and the fused fold stores the recipe in its
`twist` field so the stable ⊕ derives rather than being baked in: the pivot's state is the lift param bound to it,
the score is the sub-cone of the lift equal to the pivot's own per-element map, through each projection's components.
Operand values compare by their lowered statement identities and buffer roles under the actual coordinate scopes;
the candidate's stream coordinate is substituted by the pivot's before comparison. Captured-coordinate parameter order
does not change the value, while a different buffer or index does. What remains, in role order, must equal a channel's
pattern by canonical form. A click gives the role-to-name map and the recipe instantiates itself by renaming; no recipe
names a term's variables. Online softmax and flash attention are one recipe: the expectation channel joins by the same
call, the pivot then being the fused fold. A shared dependent reached again through another score cone reuses its
carried state only when its channel, seed and injected value agree under the actual operand bindings. Conflicting
definitions of one state name are an error. **Example** — the online-softmax carrier: state `(m, d)`, partial
`(score, 1)`, identity `(−inf, 0)`, merge `m_new=max(m,s); d=d·exp(m−m_new)+exp(s−m_new); m=m_new`.

**The λ-foldMap primitives** (`ir/pure/lam.py`) — the finished algebra vocabulary the tile IR
stores against (see the tile-lowering ARCHITECTURE for the storage story). `Lambda(params, body, results)` is the ONE
binder kind over the reused stmt vocabulary — a `Body` of PURE stmts only (ANF ≙ a let-chain), validated in
`__post_init__` via the **`Statement.pure` trait** (declared on the `Statement` interface, conservative `False` default;
`Load`/`Assign`/`Select` and the structural `Fold` node opt in; `Accum`/`Write`/`Init`/`Loop`
never do — no
isinstance whitelist), with results-defined checked there too and α-invariance by canonical renumbering
(`Lambda.canonical` — free names never renumbered). A term is closed over its coordinates by construction — values
arrive through operand edges, and only the enclosing iteration axes are read from outside — so `Fold.canonical`
(and `Lambda.canonical` for a lambda) is the one cross-scope equivalence the Tile canonical forms and the lowering
passes (cone sharing, twisted-pair recognition, seam value clustering) all consult.
`Lambda.__post_init__` installs a
dependency-safe body order and commutative argument order, so these context-independent storage invariants do not
belong to `Fold`, `TileOp`, or the structural-key path. Contraction operand roles live on Fold edges, so sorting a
commutative product's arguments does not change them. Formation is strict: a kernel's writes ride
`TileOp.output_specs`, and synthesized split-reduce loops remain Loop IR until the new kernel re-enters total lift. A
result
may be a bare
`float` literal — ι is spelled in the lift (softmax's singleton
is `(x, 1)`). The monoid is the `(init, base)` pair stored directly on the `Fold` beside the optional `twist`
recipe (the `Monoid` wrapper class dissolved at 1r) — `base : S × S → S` a pure `Lambda`, always the componentwise ⊕
that `Lambda.componentwise` builds, whose results carry the fold's REAL accumulator names. The ⊕ the fold folds with
is DERIVED from that pair (`Fold.combine` — `base` itself when `twist` is `None`, the recipe's stable conjugate
`psi(psi_inv(x) base psi_inv(y))` otherwise), and the serial streaming step is derived from it in turn (combine
specialized at the singleton), so there is one stored spelling of the algebra and update-vs-combine consistency holds
by construction. A `Fold` carries NO
precision: accumulator dtype is a KERNEL-IR fact, stamped on the lowered `Accum` by the Init-placement pass, and a
reduce `Loop` arriving with a typed `Accum` is not canonical input to total lift. A twisted monoid's combine is the
recipe's program, reached by NAMING the recipe the fold instantiates rather than by restating it; the fusion that
names it recognizes the pair by canonical form (`Fold.fuse`), never by a stored family name;
`tests/compiler/ir/pure/test_twist.py` pins its associativity on random states.

### `loop/ir.py` — LoopOp types

| Symbol                       | Role                                                                                                              |
|------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `Axis`                       | Named iteration variable (`name`, `extent`). Defined in `ir/axis.py`, re-exported here. Carries an optional `window` (`Window` — the `parent` axis this one is a slice of, and whether the slice is a cross-CTA `partition`; its `base` / `bound` are declared but unused, kept only because recorded split identities spell them): the ONE windowing concept, and never a range — a loop's data-independent `start` / `end` is the kernel loop's own (`StridedLoop`), derived where it opens; `source_axis` is the derived compat read (`window.parent`). Excluded from equality / hashing. |
| `LoopOp`                     | One kernel. Stored field: `body` (nested `Loop` tree). Computed: `axes`, `loads`, `accums`.                       |
| `Load`                       | Body-form external read: `name = load(input)[index...]`. `input` matches the producing graph node's id.           |
| `Assign`                     | The one scalar definition: `name = value`, `value` one operation over names and literals (`exp(x)`, `2 * x - y`: an affine region is one node), with an optional `dtype`. |
| `Accum`                      | Reduce accumulator: `name = op(name, value)` inside a reduce `Loop`. Initialized to its op's identity. ``axes`` lists the reduction axis names — propagated through Sigma renames (including σ-splits via `Expression.free_vars()`); the escape-analysis helper derives cross-thread cooperativity from ``axes ∩ enclosing ThreadTile.axes``. |
| `Carry`                      | The next value of one cell of a carried state: `name[index...] <- value`, holding `seed` before the first step. No op: a state folds nothing. |
| `Pre`                        | Read of one cell of a carried state: `name = pre carrier[index...]`, the previous step's value inside the loop that carries it. |
| `Init`                       | Explicit `<dtype> name = identity;` seed at this scope (`name` + scalar `identity` + `dtype`). Used for a carried state's seed (one per component), emitted above the streaming `Loop`. Scope-bound (never hoisted); shadows a deeper same-named `Accum` init. |
| `Let`                        | Pure binding of one `Expression` to a name: a scalar literal (the twisted carrier's injected `1`), a precomputed integer index, a `FlatIndex` offset. Scoped like an `Assign`; legal inside a stored `Lambda`. |
| `Write`                      | Write an SSA value to output at `index`.                                                                          |
| `Select` + `SelectBranch`    | Coord-predicated binding (replaces the old Mux).                                                                  |
| `Loop`                       | Serial iteration block: `axis` + nested `body`.                                                                   |
| `StridedLoop`                | Strided iteration (`start`, `step`) — cooperative thread-stride loop reused by Tile/Kernel IR.                    |
| `Cond`                       | If/else block over an `Expression` predicate.                                                                           |
| `Statement`                       | Base class — every body statement subclasses it. Leaves and control-flow nodes live in `ir/stmt/`.               |

Body walkers: `iter_body(body)` (pre-order; powers `for s in loop_op`),
`map_body(body, fn)` (transformer), `Statement.rewrite(rename_ssa, sigma)`
(per-stmt copy with SSA rename + Expression substitution),
`str(stmt)` (the statement's text, the lines kernel dumps show; block stmts
recurse via `pretty_body`).

CUDA scalar rendering goes through `op_to_expression` in `ir/render.py`. Boolean masks retain the historical f32 SSA
convention, so Torch's `bitwise_not` spelling renders as logical zero-test (`mask == 0`); explicitly bool-stamped values
use the same semantics. Integer complement is not inferred from that name and fails closed until it has a typed
consumer. The target-aware render of an `Assign` is responsible for conversions such as `__half2float` when an
argument's stamped dtype differs from the result dtype.

`Select` uses the common dtype of its branch values, and type propagation gives its consumers that same dtype.
Selecting between two FP16 values must preserve the rounding of a subsequent FP16 product; promoting the selection
to FP32 would silently change the computation.

A typed `Let` also declares its value's dtype. Operand-edge and cone-statistic readers preserve that declaration,
including when a static scalar load becomes a literal binding. An untyped binding remains unknown; the readers do
not infer a storage dtype from the Python literal. Cuts and schedules consume these facts without changing policy.
Fragment epilogues retain these bindings through the shared statement renderer, including their coordinate
substitution and storage dtype. An atomic split's projection may use a binding independent of the carried state;
binding a state-dependent expression does not by itself prove that the projection distributes over addition.

Dependence cones: `Body.backward_cone(roots)` builds a `Cone` —
the subset of the body's immediate stmts closed under SSA dependence (a wrapper joins as a unit; internally-bound
axes excluded), plus `external_reads`, the names read from outside (axis vars and enclosing/sibling scopes alike).
Construction never fails: unresolved names are data, and chaining scope levels means seeding the next level's
`backward_cone` with the previous one's `external_reads`. This is
the shared substrate behind the rules that slice cones (the demoted-operand producer cut in
`tile/cut/030_cut`) — eligibility judgments stay in the rules, per
`pipeline/passes/ARCHITECTURE.md`.

The splicer folds the def-use graph counting the accumulators on each value's path. External
buffer depths let composition carry the result across producer/consumer boundaries without expanding either body.
Counting accumulators gives a sufficient independence proof for early reduction sharing: equal-depth reductions
cannot read each other's finalized result. The splicer supplies graph edges and keeps coordinate substitution local.

`Body.coalesce` assembles repeated shared SSA definitions once and joins adjacent independent reduction loops with
identical headers. Fold lowering and computed-operand assembly use this same operation. Its input is shared value
cones, not arbitrary repeated accumulator updates. Effects retain their order and multiplicity and stop reuse across
them; a read of a finalized reduction keeps the corresponding loops separate. Two lowerings of one shared fold need
not be equal statements — one cone edge lowers the fold alone, another fuses it with a sibling fold into one pass — so
the first declaration also wins per accumulator: a later seeded reduce loop drops the folds of accumulators the scope
already declared and keeps only what its other accumulators read.
Fold lowering also remembers each term placed at each scope before visiting its operands, so a shared term and its
boundary stores are emitted once instead of relying on statement cleanup to remove duplicate effects.

`backward_cone` resolves reads by NAME over a body it assumes is SSA, so it is only sound where one name has one
def. `Lambda.cone` is the caller that cannot assume it: a stored combine takes its states in as params and writes
them back on the way out to spell its results (`Recipe.program`), so a read of the INCOMING state would resolve
forward to the write-back. A param names itself, and `cone` keeps it that way by hiding the re-bindings from the
walk. Any other caller that cones a non-SSA body owes itself the same reading.

`rewrite` has two distinct rename channels that must stay disjoint:
`rename_ssa` carries **SSA-name** renames, `sigma` carries **axis**
substitutions. `Load`/`Write` index expressions apply *both*
(`sigma.apply(e).rename(rename)`) so an indirect
(gather) index Var gets renamed exactly once. Putting the same name in
both maps renames it twice — and if the two passes form a chain (e.g.
`x → in5` and a pre-existing `in5 → in26`) the double application
collapses it transitively, silently wiring a gather to the wrong row.

`rewrite` is also **not scope-aware**: it descends into every nested body and maps a stmt's own bindings as well as
its reads, while `Assign` / `Load` / `Select` names bound inside a `Loop` / `Cond` body are scoped to that body. The
two are safe together only for a whole-subtree renumbering (`rename_ssa_sequential`). When the rename instead comes
from *dropping* a binding — load dedup, CSE — an inner scope that merely re-uses the dropped name's spelling is a
different variable, and renaming it both redeclares the survivor inside the scope and rewires the inner arithmetic to
the outer value. `passes.rename_free(stmt, alias)` is the hygienic form: it prunes the alias of whatever each child
scope re-binds before descending. It rewrites the wrapper with empty child bodies, then visits each child once;
rewriting the full subtree first would repeat and discard work at every enclosing level.
If an alias's destination is shadowed, the local binding is renamed before substitution to prevent capture.
`Body.local_defs` distinguishes bindings at this scope from definitions in deeper scopes; a deeper shadow must not
hide an available enclosing value. Copy elimination and CSE both use the hygienic rewrite.
`normalize.place_values` threads its own per-scope availability. σ has the same hazard with axis names, which collide
across a tree by design (a cone statistic's axis may spell the same as the enclosing contraction's):
`fold.subst_free(stmt, sigma)` is σ's hygienic form — it stops at
a `Loop` / reducing `Fold` binder that re-binds a substituted name, and is what the smem compute fill substitutes
cell coordinates through.

### Body normalization — `emmy-ir`'s `normalize/`

Normalization is a canonical form, not a pipeline pass. It runs the same on every body, at construction and inside
every identity digest, and answers to no knob, pin or evidence; nothing under `ir/` imports the pipeline that holds
them. Every function in the module runs inside `normalize_body`. A body → body transform that something decides — a
knob the search sets, a pin, a pass stage — is a pipeline pass and lives in the pass file owning that decision, even
when another pass could reuse it from here. A pass may call a normalization step (kernel lowering's
`040_split_invariant_divides` owns the hoist it applies); normalization never calls a pass.
`tests/architecture/test_layering.py` holds the module to one exported entry point, to no transform that no
normalization step calls, and `ir/` to no pipeline import.

Pure `body → body` passes run from `LoopOp.__post_init__` so every
constructed `LoopOp` (including intermediate fusion results) is
canonicalized before validation:

While it builds a region, fusion holds a compact body containing scalar `Call` statements. Each references one
read-only `Subroutine` with explicit coordinate parameters and captured input buffers. A symbolic dim an index names
is no coordinate, so the definition reads it free, as it reads a buffer. Fusion runs `prepare_body` on the compact
body: ordering, placement and coordinate simplification operate on the calls, and on each shared definition once.
Identical calls to the same definition share by argument structure without searching their bodies. Fusion then
expands the calls with the demand reconstruction that built the region. Independent reductions with equal extents
and reduction depths share axes; coordinate substitutions keep different arguments distinct. Terminal values also
seed expansion, so a body without output writes expands too. The expanded body is the only form that leaves fusion:
normalization refuses a body that holds a call, so validation, executable identity, serialization and Tile IR
lifting never meet one. Placement then shares common producers across definitions, and operation clustering starts
from that CSE form. Subroutine boundaries never limit fusion.

- `topological_sort` — stable Kahn reorder so SSA defs precede their uses
  within each body (fixes splicer-produced use-before-def).
- `drop_size_one_free_axes` — inline extent-1 free Loops.
- `drop_size_one_reduce_axes` — collapse a canonical extent-1 reduction to its single monoid update. This includes
  decode-softmax values that fusion hoists into the enclosing scope; copy-alias elimination then removes the identity
  update before total reduction lifting.
- `canonicalize_free_axis_order` — sort outer free Loops by their row-major position in boundary writes, so output
  storage geometry rather than axis spelling decides the nest. When the writes cannot totally order the chain, axis
  roles and the least complete alpha-renamed form decide the order. Role expressions use a fixed coordinate order,
  independent of the current nest, so affine expression ordering cannot make the loops swap on every round.
  A cross-CTA partition coordinate occupies the workspace's leading index, so the same rule keeps it outside the axes
  it partitions without a naming convention.

- `fold_constants` — an operation whose operands are all literals (a literal is a `Let` binding a number at the
  default precision) becomes the `Let` of the value it computes, at the statement's own dtype, so a rounding the
  program performed still happens where it did and nothing computes past it. With `x - x → 0` from the affine fold,
  the one-key softmax weight `exp(s - s) / exp(s - s)` is the literal `1` and the product by it folds to nothing.
- `fold_trivial_selects` — a `Select` whose branches all bind one value is a copy of it.
- `drop_unread_values` — a pure statement whose values nothing reads is dropped, and a loop or branch left with
  no statement goes with it: the work reached no output. The one-key softmax weight leaves its whole score behind
  this way, the dot product that fed it and the loop that summed it included: a dropped value leaves its operands
  unread in turn, and one call drops the whole chain. A body that stores nothing is a fragment whose values are all
  its reader's, so nothing is dropped from it.
- `one_operation_per_definition` — every definition is one operation over names and literals: a nested node of an
  `Assign`'s value that is no name, literal or node of the value's own affine region (a comparison, a select, a cast,
  any operation) is a definition of its own ahead of its reader, and an `Accum` sums a name or a literal. An affine
  region is one node, as a sum is one operation: `canonicalize_expressions` spells every expression in affine form
  — like terms summed, zero terms dropped, the constant last, in the coefficient domain of its field's class (an index
  folds integer literals alone and keeps a quotient a term, a value folds any literal and a division by a float
  literal into the product by its reciprocal) — and this step joins an affine definition read exactly once, inside an
  affine reader, into the reader's region, so `x + x`, `2 * x` and `t = 3 * x; s = t + x; o = s - x` are one body,
  `x / 2.0` and `0.5 * x` another, and `x - x` is `0`. The affine form reassociates a sum and rounds a reciprocal
  once, which every reduce schedule and fast-math division do already. A chain of named temps and one tree are one
  body up to the single-reader guard — a sum two statements read stays a definition — so the readers of the normal
  form (the tile lift, the kernel passes, the renderer) read one operation per statement; the renderer spells a nested
  value a kernel pass built as temps of its own.
- `merge_sibling_reduce_loops` — unify sibling reduce axes whose Load positions overlap, then merge matching Loops
  before descending into their children. A parent merge therefore exposes child reductions to the same walk.
  Overlapping reductions share one canonical axis name (softmax's max + sum sweeps; the two matmul reductions in
  `silu(x@Wg) * (x@Wu)` that both index `x` at the same K slot). A position is `(source, dim, anchor, coefficient)`,
  read through `affine_form`: a blocked reduce indexes its stream at `o·B + i` and still walks that dimension, while
  the anchor keeps `o·B + i` apart from `o·B + 32 + j`, which walk different halves. Union-find groups all
  transitively-overlapping Loops at one scope.
  Sibling reduce Loops that share `axis.name` / `extent` concatenate into one Loop body.
  Every gate is phrased over what the second Loop reads from its ENCLOSING scope (`free_names` — what it uses and does
  not bind itself): it must read no name the first body defines (blocking softmax-style sequential reduces where
  sum-exp reads `acc_max`). A statement between the loops that the incoming Loop waits on, directly or through other
  statements between them (`ordering_constraints`: names, a buffer's version, accumulator and carried state, protocols),
  moves above the first Loop, unless it waits on the first Loop itself — the form must not depend on where a value was
  spelled. An edge from the incoming Loop to the first that is not a name (the first writes a buffer the second
  loads) is a dependence too, and keeps them apart. Names both bodies merely happen to bind are a COLLISION, not a
  dependence — two alpha-equal copies of one cone always share every spelling — so the incoming body's copies rename
  apart. Only a name the incoming Loop still binds after it closes (its immediate carriers, which `Loop`'s render
  declares ahead of the loop) refuses, because the rename cannot reach that name's readers outside. Eliminates the
  duplicate K traversal in patterns like `silu(x@Wg) * (x@Wu)`, and the duplicate score pass between the channels of a
  blocked twisted carrier; subsequent normalization collapses the duplicate loads, and the lowering passes stage both
  weight tensors symmetrically.
- `place_values` — every pure value once, at the shallowest loop scope on its consumers' path that binds its
  coordinates and holds its operands. Values are told apart by number (the value numbering below): the same function
  of the same coordinate expressions is one value whatever it is called, so a second computation a scope can already see
  reads the first, a copy aliases its source, and a block without a side effect moves with everything it still holds.
  This one step is loop-invariant motion, load and expression dedup, copy-alias elimination and common-branch motion:
  a value both branches of a condition compute moves ahead of the condition, a branch alone never speculates. A read of
  a buffer or carried state the body writes carries the version it read — the writes before it on its path, with a
  loop's write counted on the back edge and a branch's not — so it never merges with a read of another version nor
  moves above the write that made its own, and everything computed from it inherits the version. Stores, reductions,
  seeds and carried updates stay where they are, an ordered execution protocol holds everything after it behind it,
  and a binding rebound later represents no other value. Two seeded reductions of one value in one loop, each updated
  once and read only after the loop, are one reduction. Division keeps its own rounding even when its denominator is
  invariant; reciprocal multiplication is not a normalization. Names are distinct across scopes when it runs, as the
  sequential renaming leaves them.
- `rename_ssa_sequential` — `Load` names become `in0, in1, …`, accumulator state becomes `acc0, …`, and
  every other definition becomes `v0, v1, …`, in lexical definition order. Names stay globally unique while each
  nested body tracks its own binders, so sibling scopes may reuse the same source spelling without collapsing. Scalar
  accumulators bind beside their own reduction loop, without exporting a nested loop's private state into its
  ancestors. Conditions and unseeded loops preserve the state they update, including an explicit enclosing seed.
  Axis renames reach conditions, reduction metadata, and `Window` parent/base/bound metadata as well as indices; a
  reduction's axis tuple is canonicalized as a set. SSA values travel only through the rename channel, never `sigma`,
  so indirect indices cannot be renamed twice. The order reads a scope lexically: two bindings of one name, which
  motion can produce, keep their source order and a read binds the one before it (or the first, where none precedes:
  a scope is order-free); an accumulator is read where it stands — ahead of its update the step before, behind it the
  updated value — so the online softmax's rescale reads the old maximum beside the new one; a carried state orders
  nothing within its step, since a `Pre` reads the step before wherever it sits, while the loop that carries the state
  produces it. A scope whose bindings and reads wait on each other has no reading, and normalization refuses it.
- Normalization runs its steps once and checks nothing. The steps that feed each other run to their own fixed
  point where they sit: a value placed outside its loop can expose a sibling merge, two reads placement made one can
  leave a predicate constant, and a merge exposes values to place, so the merge, the expression simplification and
  placement run until the first two leave a placed body alone. The split feeds the steps ahead of it — a definition
  it makes can be placed, and a region it joins can leave a value unread — so the steps run again after a split that
  changed the body, and only then. The order runs last, once, with one exception: the unread-value drop reads by
  name, and a cut can splice one cone twice under the same names, so a value only its namesake's readers kept alive
  is unread only once the order has given every binding its own name. A body the order left such a value in goes
  through the steps again. What keeps this honest is a test, not a loop: the normal form of every kernel a repository
  golden or the realization corpus stores is one normalization leaves alone
  (`tests/compiler/ir/stmt/implementation/test_normal_form_is_final.py`). A form that was not final would give two
  spellings of one body two texts, and no error.
- Legal motion and merging expose availability; equality alone never moves a value across a scope or an effect. This
  is scoped placement, not general partial redundancy elimination: separately executed sibling scopes and equality
  through mutable loop-carried state may remain, and loads are invalidated by writes to any element of their buffer.
  Affine integer expressions over bound loop coordinates have a unique coefficient form in lexical axis order; for
  proven nonnegative indices, positive constant quotient chains collapse to one divisor, and `/` and `//` share that
  spelling. Unsupported expressions compare structurally after these rewrites.
- The canonical order comes from the value numbering. Within a scope, blocks without a side effect come first, then
  leaves, then blocks with one, each group by statement type; ties break by the value — a leaf by its number and its
  coordinates spelled by binding depth, a block by the hash of its scope tree — with every buffer spelled by the role
  the body gives it (`values.roles`: the rank of its cell of use), so renaming buffers moves nothing unless it renames
  two the body cannot tell apart, which then order by name. Dependency and effect constraints
  (`order.ordering_constraints`) bound the order; a heap-based Kahn sort takes the least-priority ready statement.
  Names are then given sequentially, a commutative operation lists its operands by value and then by those names, and
  a final axis unification restores shared reduction dimensions without merging dependent loops. The order is final
  on its own output because nothing it sorts by reads what it decides: a statement's shape and a block's spelled
  subtree read a sum's coefficients sorted and the block's children already ordered, and no number reads a name.

### Value numbering — `emmy-ir`'s `numbering/values.rs`

A statement's number is its statement type and payload (op, dtype, resource) over its operands' numbers, with every
maximal coordinate-only index expression a parameter numbered by first appearance across the statement and its operands
and spelled by the depth of the loop that binds it. A statement is thereby a function of coordinate expressions: a
weight read at `(h / 384) * 128 + d` in one nest and at a grid axis in another number alike, as do the same cone inlined
under two consumers. A reduce binds the parameters that mention its axis, retains their bound index expressions in its
number, and keeps the free coordinates those parameters read. Offsets and strides still distinguish reduced walks.
Commutative operands order by number, and operands tied on number by the layout and then the parameter order their
arrangement gives, so neither the source order nor the spelling reaches a number. A carried state an operation reads
ahead of its update — the previous step's value — is keyed by use, like every other read of a state; only a name free in
the body is spelled. The same number applied to the same coordinate expressions in one scope is one instance — what
placement computes once.

### Structural identity — `emmy-ir`'s `identity.rs`

`Body.identity()` takes the executable normal form (`normalize_body`) and keys its scope tree (`values.digest`):
every block a node described by its statement type and its extent or predicate, every leaf an instance with its
coordinates spelled by binding depth, each scope's members sorted, and an effect order two members must keep riding the
later member — so neither spelling nor dependency-valid order reaches the key. Clear external argument names remain on
executable bodies; the key is digest material. One normal form serves both, so a body keys the same whether it was held
bare or constructed as a Loop op.

- External buffers and carried states are keyed by use, never by spelling: colored by type (`identity_key(with_io=True)`)
  and by everything downstream of them, refined until the cells stop splitting. Two in one cell are told apart by
  individualizing each in turn and ranking first the one whose tree hashes smaller; two whose trees hash alike are
  interchangeable. The ranks are the roles `b0, b1, …`: two kernels whose typed argument lists read alike in
  declaration order but assign the types to different roles key apart, and declaring the same roles in another order
  keys the same. The identity material names which buffer fills each role (`Op.canonical_buffers`), which is how the
  kernel cache rebinds a hit.
- Optional operation clustering replaces each elementwise operation with its compute-unit representative before
  numbering. It is the only operation rewrite owned by identity; all executable canonicalization stays in
  `normalize_body`.

The exact and compute-unit-clustered forms are cached on each immutable `Body`. Two bodies that differ only by SSA or
axis names, argument spelling and discovery order, dependency-valid statement order, or equivalent commutative and
affine expression spelling therefore share a structural key. Use it when deduplicating candidate bodies in search.

### `ir/expression.py` — Expression simplification

`simplify` (called inside `normalize_body`). Generic bottom-up Expression rewriter:
constant folding, algebraic identities, range-based comparison folding
(`(k0 > 2047 ? 2047 : k0) < 0 ? 0 : k0` → `k0`). `SimplifyCtx`/`Interval`
track integer ranges from axis extents (`axis.extend_simplify_ctx` pushes
each loop axis into the ctx). `SimplifyCtx.bounds` additionally tracks a
*symbolic* exclusive upper bound per var (`i < seq_len`) so a modulo by a
non-literal divisor folds — `i % seq_len → i` when `i`'s loop extent is
`seq_len` (`_mod_below_divisor`). This collapses the delinearized seq
coordinate `((i*stride + feat) / stride) % seq_len` that compose-indexmaps
emits back to `i`, the symbolic-shape counterpart of the literal-divisor
`_div_mod_decompose` cleanup (a static `seq_len` already constant-folds it).
Symbolic-extent axes get `[0, sentinel]` ranges (non-negativity for the inner
`(i*c + …)//c → i` div fold) instead of being dropped.

`_div_mod_decompose` also sees through a division standing in its way (`A / d`
decomposes by `n` once `A` decomposes by `n·d`), and through a sum whose one
addend is a clean multiple of the divisor (the partner then owns the whole
remainder, so restating it as `n·(x/n) + x%n` suffices). Together those separate
a sub-byte-packed operand address: an NVFP4 weight spells `((row·K + k)/2) %
(K/2)`, holding the row axis inside a division, and the decomposition puts the
row on the quotient side where a consumer asking "does this index still mention
the row outside a div/mod" can see it. The free-axis re-fusion every kernel forms through
is that consumer, and its answer decides whether a packed matmul binds a
contraction at all.

### `loop/splicer.py` — LoopOp merger

The machinery `pipeline/passes/loop/fusion/010_merge_loop_ops.py` calls to splice a DAG of `LoopOp` nodes. `Sigma`
(from `ir/sigma.py`) is the axis-substitution bookkeeping threaded through the merge.

`splice_graph` resolves each internal Load through `Graph.producer(buffer)`, so primary and secondary output buffers
use the same path. Every graph output supplies an explicit `(loop tag, Write.output)` root. Separate terminal loops
therefore seed one worklist; its one binding table shares equal upstream demands across output ports instead of
inlining a shared producer per consumer. The single-sink convenience form still derives the unique terminal loop and
selects all its Writes. Every `NotSupported` carries a reason string, logged at DEBUG by `splice_loops` —
`compile -vv` shows which pattern a rejected edge hit.

The splicer forms one subroutine per demanded source reduction, with nested reductions represented by calls to their
own shared definitions. Equal calls reuse the same demand before expanding their bodies; different coordinates retain
distinct calls. `compile -vv` prints this intermediate form as `sub name(buffers, coordinates):` and one-line calls.
The final Loop IR listing is expanded and fully CSE'd, as are all persisted kernels.

Body analysis, mutable construction and demand reconstruction are fusion's body assembly (`loop/assembly.py`). It
consumes bodies and splice edges, and uses the statement module through its exported interface. The splicer chooses
graph regions and wraps the reconstructed body in a validated `LoopOp`. The same engine expands the subroutine calls
before the body leaves, sharing equal demands before constructing duplicate cones. Fresh names and coordinate
substitution prevent capture; equal reduction depth keeps a reduction that reads another's finalized value in a
separate sweep. Every legal fusion region is still built whole; subroutine boundaries do not affect final CSE.

Before dependency reconstruction, `splice_graph` finds output equivalence clusters: single-owner copy chains ending
at a terminal graph output, with the same dtype and element count and an exact symbolic proof that the source and
destination coordinates are related by a reshape and axis permutation. Equal element count alone is insufficient;
slices, broadcasts, and conversions remain ordinary edges. The proof compares each source coordinate with one
mixed-radix digit of the destination's dense flat address, then composes those inverse layouts across the chain. The
splicer retargets the computed source's `Write` through that inverse and removes the copy roots from reconstruction.
This preserves the producer's loop geometry through terminal reshape/transpose chains without enumerating the output
domain. A leading dimension both shapes share as the same symbol (the token axis of a serving prefill program) has
no dense flat address to digitize; the proof strips it, proves the static trailing shapes, and the retarget carries
the source's leading index through unchanged. Without that the symbolic prefill twin kept its reshape copies as
ordinary edges, and the fused half took a different form from its static twins — projections recomputed under
every per-column statistic.

A `Write` that observes an `Accum` inside that accumulator's own reduce scope is an ordered prefix output. The
splicer refuses that shape whether it is the merged root or a producer edge: dependency reconstruction would freshen
the reduce loop and move the `Write` after it, changing every prefix value into the final reduction. Such an
effectful inner loop is not valid input to total lift.

Before splicing, `loop/lowering/090_spell_store_rounding` turns a public store that narrows an `Accum` into an ordinary
typed `copy` statement. A decomposition may route that accumulator through one transient, shape-only buffer before a
pass-through LoopOp writes the public buffer; that direct load retains the accumulator's implicit f32 dtype, so the
same rule spells its public conversion. An actual `Assign` computation over private reduction state remains untyped:
normalization and softmax therefore retain f32 internal state rather than narrowing it at an inferred projection edge.
`splice_graph` then preserves the explicit conversion through its ordinary statement path and reconstructs no dtype
boundary from source provenance or graph topology.

Construction is bounded as a ratio: the dedup table shares each `(stmt, emit scope, σ)` binding, and a legitimate
splice emits a few bindings per source statement at most (a value read at several offsets takes one each). A
recurrence left unrolled — each stage re-demanded under compositions of σs, DeepSeek-V4's 20-iteration Sinkhorn chain
being the live case — multiplies bindings per stage instead of deduplicating, and such a merge cannot be constructed
at any budget. The first binding past `_BINDING_RATIO` times the region's statement count raises `UnfusableStmt`,
which names the loop. Fusion never catches it: its regions are decided before any is spliced, so a region that
multiplies is a recurrence for `loop/fusion/005_roll_recurrence` to roll, not a region to shrink — shrinking made the
kernel set depend on the order the producers were visited in. This is a termination bound, not a fusion-quality gate:
placement still owns every cut on a merge that CAN be built.

Each splice memoizes `Expression.free_vars()` by expression identity while placing dependencies. Sigma expressions
remain live for the splice, and identity avoids both repeated coordinate-tree walks and the recursive structural hashing
a global cache would require; the memo is discarded with the splicer.

`Sigma` computes canonical expression text once for each initial substitution. Derived substitutions created by
`extend` and `restrict` retain the applicable canonical entries from their parent, so dependency placement neither
reformats deep coordinate trees nor retains duplicate canonical strings.

### `loop/runner.py` — C++ JIT executor

`execute_loop_op_cpp(loop, input_arrays, out_shapes)` renders the LoopOp body to a C++ source string and JIT-compiles
it in-process via cppyy / Cling (cached by the rendered source), then calls it with raw pointers to the input and every
output array. One output returns an array; multiple outputs return a tuple in the operation's graph-populated ABI
order. Each Write's own scope determines its output shape, so independent sibling nests may reuse axis names with
different extents. This powers `LoopOp.forward`, so post-fusion graphs run through the default `Backend.run` topo-walk
like any pre-fusion graph.

### `loop/builder.py` — body construction

`BodyBuilder` constructs bodies for fusion and subroutine expansion without spelling out every `Loop(Axis(…))`
nest. Construction is mutable — descent is a dict lookup per scope level and a prepend is an append to a
reverse-ordered list — and the immutable body is materialized once by `finish()`. Rebuilding the tuple tree per
insert is quadratic in program size and re-runs each level's `Loop` construction normalization per insert, which is
invisible on small graphs and decisive on large ones. Fresh SSA names retain the lowest available deterministic
suffix while a per-hint monotonic cursor ensures each occupied suffix is tested at most once; the used-name set
remains authoritative when another hint claims a future suffix.

## `tile/`

Tile IR stores the complete inner loop nest as one tree of `Fold` terms. The Loop IR boundary peels the outer parallel
axes, converts every reduction from its explicit `Accum` statements, and leaves each nested reduction in the same
position inside its parent lambda. A root zero-axis Fold holds the per-cell statement sequence. An output loop's
per-cell projection becomes a zero-axis term evaluated over the sweep axis — a sibling operand of the root — and its
writes live in `TileOp.output_specs` as sweep specs.

A nonzero-axis Fold exposes its combine result names through `Fold.defines()`, so later sibling statements and outer
folds may consume its result without hoisting it to an operand edge. `Fold.loop` mechanically lowers the tree back to
the corresponding nested Loop IR.

The total-lift invariant is that no raw inner `Loop` survives. A bilinear term orients itself at formation, its
shared argument in the contraction's canonical A slot; `TileOp.__post_init__` then applies the tree-wide
canonicalization — an identity projection dissolves into its operand, and same-value cones become one shared object. Scoped lambda equivalence is an analysis over the
canonical Folds. A separate pre-scheduling rewrite fuses every reduce that reads a reduce into the twisted carrier a
recipe recognizes — `(maximum, denominator, expectations…)` for the exp family — hoisting the factors constant along
the axis (attention's `1/l`) out of the fold first; softmax and masked or unmasked SDPA are arity variants of one
recipe, not separate matchers. Placement and cross-CTA split are structural phases before site construction. Classic scheduling
classifies the resulting Fold tree, assigns each node once and every consumer operand edge independently, then stores
one complete typed schedule on `TileOp.schedule`. Unsupported shapes remain unmapped; scheduling never annotates or
rewrites the Fold tree.

See [`tile/ARCHITECTURE.md`](tile/ARCHITECTURE.md) for the exact storage and boundary contract.

## `kernel/`

### `kernel/ir.py` — fully-scheduled kernel form

Reuses `Tile` + leaf stmts from Tile IR; adds hardware primitives
materialized from scheduling decisions. `KernelOp` carries the body
directly (no separate AST class).

A Kernel IR statement type is data: a declared statement type (see the statement module's interface), stated in the
crate's declaration list beside the shared statement types and imported here as a class. The statement type holds fields
with their roles, how it prints and at most a constructor check; what rendering or the CUDA lowering asks of a statement
beyond that (a grid's size, a fragment's register count, a store's element addresses) is a function over the statement,
beside the statement type's render function, never a member attached to the class. Every field has a type from one
closed set — strings, numbers, booleans, expressions, axes, dtypes, elementwise ops, bodies, statements, records of such
fields (`FragmentArg`, one operand of a `FragmentApply`), and tuples and optionals of those — which is the set the
crate's `Field` trait is implemented for. A fragment layout is therefore named, not held: the statement stores the
atom's layout name and rendering looks the geometry up. `tests/architecture/test_layering.py` guards the members.

| Symbol             | Role                                                              |
|--------------------|-------------------------------------------------------------------|
| `KernelOp`         | Graph-op wrapper around a `Tile`-rooted body. One per kernel.     |
| `Smem`             | `__shared__` array allocation (name + dtype + extents + optional `align`). Swizzled TMA operand slabs align to their full swizzle atom (`8 × swizzle_width` B: B128→1024, B64→512, B32→256) — the coordinate-only `ldmatrix` XOR only reproduces the hardware's absolute-address swizzle when the base zeroes the swizzle's source-address bits; non-swizzled TMA keeps 128 B, fp16 16 B. `pack_smem` (the shared pool packer used by `smem_bytes` and the renderer) pads each buffer to `max(sizeof(dtype), align)` so the static-vs-dynamic gate and the launch-time dynamic-pool size agree. |
| `Let` (a `stmt/` leaf) | Pure binding of one expression to an SSA name: a scalar literal, an integer index bound once outside a nested hot loop, or a flattened buffer coordinate (`FlatIndex`) reused across a copy's trips. |
| `Sync`             | Thread barrier: CTA-wide `__syncthreads()`, warp-scope `__syncwarp()`, or a named `bar.sync` over a thread subset inside a warp-specialized branch. |
| `TreeHalve`        | Cross-thread tree reduction over a smem buffer.                   |
| `WarpShuffle`      | Lane butterfly reduction or inclusive prefix scan over an aligned lane group. |
| `WarpBroadcast`    | Broadcast the last lane's scan state within each aligned lane group. |
| `RegFragment`      | Per-thread `mma.sync` register array declaration, zero-initialized for C. The established m16n8k16 layout uses A/B/C counts 4/2/4 for f16/f16/f32; the Volta m8n8k4 layout carries explicit A/B counts 2/2 and C counts 8 f32 or 4 packed f16 because one instruction realizes four PTX cells arranged as one logical 16×16 tile. Carries instruction shape, dtype, and an optional explicit register count. The opaque `nvcuda::wmma` nodes remain retired. |
| `LdmatrixLoad`     | Load one operand into a `RegFragment`. The m16n8k16 layout can use `ldmatrix.sync.aligned.m8n8.x{4,trans}.b16` from shared memory or a global-memory-direct gather with the same lane map. SM70 has no `ldmatrix`, so gmem-direct and unpaired staged Volta paths use cooperative gathers. A materialized canonical-B tile with even M/N fragment counts instead derives paired crosswise-A and B-congruous shared layouts; each 128-bit load drains two logical fragments. Its four computation groups duplicate the appropriate A or B quadrant. `b_trans=True` marks a `[N, K]` weight and selects the corresponding transposed gather. Guards clamp M/N lanes and zero masked K elements in both layouts. A 1-byte staged slab (`byte_slab=True`) has no `ldmatrix` below sm_100a and drains through the cooperative gather too; when it also carries a `scale_buffer` the slab holds a weight stored as bytes with one scale per k block in that companion slab — PACKED PAIRS (an NVFP4 weight: one byte, two K elements, decoded through the value table) or fp8 bytes (one element each, converted by the hardware cvt and multiplied by an f32 scale before the round to the fragment). |
| `MmaSyncPtx`       | Inline PTX for either `mma.sync.aligned.m8n8k4.row.{col,row}.{f32,f16}.f16.f16.{f32,f16}` on the Volta fragment layout or the established `mma.sync.aligned.m16n8k16.row.col.{f32,f16}.{f16,bf16}.{f16,bf16}.{f32,f16}` family. Paired B-congruous loads select row/row; the other Volta paths retain row/col. The renderer includes only the selected family's prelude, so SM70 never parses newer `ldmatrix` or m16n8k16 assembly. The BLOCK-SCALED fp4 form (`m16n8k64`, `kind::mxf4nvf4`) additionally carries `sfa_frag` / `sfb_frag`: both multiplicands are packed e2m1 pairs and the instruction applies one ue4m3 scale per 16 K elements itself, so the call passes those two scale registers where the others repeat the accumulator. Its data fragments reuse the fp8 byte loaders — the k64 4-bit lane map is the k32 8-bit one, over a row of K/2 bytes — leaving only the scale loaders new. It assembles only for the arch-suffixed consumer-Blackwell target, which the plan requests through `KernelSpec.arch_specific` (the flag TMA also sets). |
| `WgmmaDescriptor`  | The 64-bit shared-memory matrix descriptor a `wgmma` operand is read through: start address, leading and stride byte offsets and the swizzle mode, built by `emmy_wgmma_desc` from the slab address the same swizzled TMA or cp.async fill deposited. |
| `WgmmaMma`         | One `wgmma.mma_async.sync.aligned.m64nNk16` cell over N/8 of the per-warp 16×8 C fragments (instruction register d[4j+k] is fragment j, register k); A from a descriptor or a 4-register fragment, B always from a descriptor; the transpose bits are template arguments of the generated wrapper. |
| `WgmmaFence` / `WgmmaCommit` / `WgmmaWait` | The asynchronous discipline around a chunk of `WgmmaMma` cells: fence before the first cell after the accumulators were touched, commit after the chunk, wait before any accumulator read and before the ring slot is released. |
| `FragmentPromote`  | Fold a packed f16-accumulate C fragment into its f32 shadow fragment and rezero it (`emmy_mma_promote_f16acc`: PTX `cvt.f32.f16` + add per element, with warp shuffles on Volta to align its distinct f16 and f32 C layouts) — the chunked-accumulation promote pairing the f16-acc `MmaSyncPtx`. The mma chain accumulates in f16 at full rate; each K chunk (the staged bk slab, every `_F16ACC_STEPS` gmem-direct atom steps) folds into the f32 shadow, bounding the f16 rounding to one chunk while the store/epilogue read f32. |
| `FragmentApply`    | The one pointwise node over a C fragment. Each argument resides in another fragment, a per-row register pair, a cell-uniform scalar, a predicate over the element's absolute coordinates (`COORD`), or a global-memory load template at those coordinates (`GMEM`). A coordinate mask is a `where` over a `COORD` predicate — the masked branch takes the carrier's finite identity, avoiding `-inf - -inf` in an all-masked chunk, and an additive mask applies its keep op first; an additive bias is an `add` of a `GMEM` operand. The atom's fragment-layout descriptor supplies the element count and row mapping, so the same leaf serves m16n8k16 and Volta m8n8k4. |
| `FragmentRowReduce` | Fold one warp's C fragments along the atom's N direction, per ROW: in-lane columns combine first, then the layout's `__shfl_xor` masks combine the column-group lanes. The resulting register pair is what a `FragmentApply` broadcasts as a `ROW` operand. The chunk tier's pivot and summed channel partials are exactly this, which is why the tier needs the chunk inside one warp column. |
| `FragmentRepack`   | Convert C fragments into a 16-bit operand in registers. The m16n8k16 f16 B form exchanges packed column pairs across the warp; the A form uses the existing lane layout. m16n8k16 consumes two adjacent fragments; Volta m8n8k4 selects a four-column A slice or four-row B slice from one logical 16×16 C fragment with warp shuffles of packed FP16 pairs. |
| `RegStore`         | Layout-aware per-lane epilogue store: four C elements for m16n8k16 or eight elements covering the four Volta output quadrants for m8n8k4. A paired Volta tile derives the matching interleaved 32×32 accumulator map from its cell position; it is not a schedule field. Adjacent elements leave as one packed pair when N is contiguous, including under an M-only tail guard; an N guard or strided physical orientation keeps scalar stores. Stores f32 directly or downconverts to f16. An optional epilogue is a pure term over the projection tail's own `Load` / `Assign` / `Select` stmts, stored as its body with the names bound to the store's fragments and the name of the stored value; it is evaluated at each element's own coordinates. Floating loads widen to f32, while integer loads stay typed so shifts and bitwise operations on packed codes render as integer CUDA expressions. `run` holds the next three N-adjacent cells of a 16-bit `wgmma` output: the quad's lanes exchange packed column pairs so each lane stores one cell's eight columns of a row as one 16-byte store (`097_widen_fragment_stores`). |
| Shared from `tile` | `Tile` (launch geometry); from `ir/stmt/`: `Loop`, `StridedLoop`, `Load`, `Assign`, `Accum`, `Init`, `Let`, `Write`, `Select`, `Cond`, `ZeroPrologue`. |

Volta direct A and transposed-B loads use the shared four-value loader when the flattened base and row stride
are provably four-element aligned and K needs no mask. FP16 uses one 64-bit load. FP32 uses two 64-bit loads,
each coupled to FP16 conversion in inline PTX, keeping temporary float lifetimes inside the packed load. The
alignment proof reuses the expression-divisibility check used by swizzled addresses. M/N clamps preserve aligned
rows; K tails, unknown alignment, other source types, and strided canonical B retain the scalar gather. No schedule
field or model-specific choice is involved.

## `cuda/ir.py`

| Symbol    | Role                                                                        |
|-----------|-----------------------------------------------------------------------------|
| `CudaOp`  | Graph-op carrying `kernel_source`, `kernel_name`, `arg_order`, `grid`, `block`, `smem_bytes`, `zero_outputs`, `comment`. Produced by `pipeline/passes/lowering/cuda` (renders the `KernelOp` body to a `__global__` source string). |

## Graph as the single program form

There is no separate program type. A `Graph` is the execution plan:
node ids are buffer names, `node.output.shape` is the buffer shape,
`graph.topological_order()` is the launch order, and
`graph.inputs` / `graph.outputs` / `ConstantOp` membership gives each
buffer its role (input / output / constant / scratch).
