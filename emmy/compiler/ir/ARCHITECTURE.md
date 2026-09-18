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
| `tile/ir`         | after `lowering/tile`           | `TileOp` holding the structural root `op`, output specifications, placement, workers, knobs, a typed classic schedule, and its materialization |
| `kernel/ir`       | after `lowering/kernel`         | `KernelOp` + hardware stmts (`Tile`, `Smem`, `Sync`, `TreeHalve`)                                     |
| `cuda/ir`         | after `lowering/cuda`           | `CudaOp` (rendered `__global__` source)                                                               |

## Pure terms vs statements

Two vocabularies, and exactly one direction between them.

A **statement** (`ir/stmt/`) occupies a position in an instruction stream: it has an order, a
scope, and — for a carrier — a seed the enclosing scope has to declare. A **pure term**
(`ir/pure/`) denotes a value: it binds names, carries an algebra, substitutes and compares up to
α-renaming, and has no position at all. `Lambda`, the `Fold` term and the twist recipes (`ir/pure/twist.py` — a
twisted monoid as data, which `Fold.fuse` fuses a reduce and the reduce it reads into) all live on the term side.

**A pure class is never a `Stmt` subclass and never occupies a statement position.** When a term
has to reach the instruction stream it is RENDERED into statements at the point of use — never
spliced in as one. `Fold.merge(other)` is the shape of that: the cross-partition state⊕state
combine IS the fold's stored `combine` applied with its second operand naming the partial being
merged, and it becomes `Assign` rescale temps plus one `Accum` per state component wherever the
lowering needs statements (the REG-tree merge, the cooperative tail, the cross-CTA finalize loop);
the serial step is the same derivation at the injected singleton. There is no `StateMerge` type: the term is the
`Lambda` the fold already stores,
and a rendering function is not a kind.

The invariant is what stops facts from acquiring a second home. A term that renders itself needs
no private spelling of anything the statements already carry:

- the neutral elements come from `Accum.op.identity` through the ONE identity placement
  (`Loop.render` / `StridedLoop.render`), so no `identities` field rides on the term and no
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
`ir/pure/fold.py` and is not a `Stmt`.

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
the argument for `Stmt`-hood: it has to appear at a POSITION in the emitted step stream. It does not
need to be a statement to get there. The tree already carries it: a composed node is an entry in
`operands`, and its position is produced by the derivation — `Fold.lower` places every term of the
tree at the shallowest scope binding its free coordinates, operands ahead of the term that reads
them. A loop-invariant scale `Load` ahead of attention's score contraction is that rule at work, not
a special case: it reads no coordinate the reduce loop binds, so it lands ahead of the loop even
though the cone that reads it rides inside. The only place terms become statements is `Fold.lower()`.

`Fold` does keep a small structural protocol whose names it shares with `Stmt` — `nested()` for its
children, `rewrite()` for α-renaming, and `defines()` for its result names.
These are term operations spelled the same way so one canonicalizer and one deep walk serve both
vocabularies; they are not statement behaviour, and `Fold` has no `render`. Impure computation stays
in Loop IR until total lift; there is no impure `Lambda` construction path.

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
  Tensor-IR ops survive only *inside* `LoopOp.body` as `Assign.op` or
  `Accum.op` (`ElementwiseOp` only — `ReduceOp` is not a valid body
  op; reductions are `Accum` statements inside a reduce `Loop`). `LoopOp` construction orders a free-loop chain by
  the row-major coordinate depth in its boundary writes; axis spelling is only the fallback when output storage does
  not totally order the chain. The resulting geometry, rather than source names, reaches Tile IR placement.
- **Loop → tile** (after `lowering/tile`): `LoopOp` nodes are replaced by
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

**Stmt subclasses are `@dataclass(frozen=True)`** — every concrete Loop-IR
/ Tile-IR / Kernel-IR statement (`Loop`, `Cond`, leaves, `Tile`, `Smem`, `Sync`,
`CpAsyncCopy`, `TmaDescriptor`, …) is immutable + hashable. `Body` is a `tuple[Stmt, ...]`
subclass, so ordinary body equality and hashing work end-to-end. `Body.structural_key()` uses the complete
`structural.form`, including identity-relevant fields such as `Axis.window` that may deliberately be excluded from
general equality. Its exact and clustered keys, and its executable normal forms, are cached properties on that Body;
there is no equality-keyed shared cache that could alias distinct metadata. To "edit" a frozen
Stmt, return a fresh instance via `dataclasses.replace(stmt, field=value)`;
`__post_init__` coercions use `object.__setattr__`. Ops, by contrast,
are frozen and unhashable — rewrites replace the op and rebind its graph node. Op fields stored inside Stmts (e.g.
`Assign.op`) must be lightweight value objects (e.g. `ElementwiseImpl`,
not `ElementwiseOp`) so the surrounding Stmt's hashability isn't poisoned.

## `base.py`

Cross-cutting root. Imported by every dialect, imports nothing from
them.

| Symbol          | Role                                                                           |
|-----------------|--------------------------------------------------------------------------------|
| `Op`            | Base class. Subclasses implement `infer_output_shape` and `forward` (numpy).   |
| `InputOp`       | Sentinel: graph input tensor. Value supplied by the executor.                  |
| `ConstantOp`    | Sentinel: weights / scalar constants. Scalars carry `value`; tensors carry `source_path` / `source_shape` / `source_dtype` (the safetensors / `nn.Module` address) plus `load_ops` — a chain of frontend ops applied at bind time by the loader. `source_parts` is the multi-source alternative: `(path, shape)` pairs the loader reads and concatenates along axis 0 before running the chain. `source_graph` is the N-source bind record (`032_fold_constant_subgraphs`' collapsed static cone): a mini-graph whose external leaves name source paths and whose scalar leaves carry values — the loader binds/evaluates it through the NumPy backend, then runs the chain. Exactly one of `source_path` / `source_parts` / `source_graph` is set on a loadable constant. |
| `_keepdim_axis` | Shape helper shared by `ReduceOp` (tensor) and `MeanOp` (frontend).            |

## `expr.py`

Shared expression sublanguage used by every IR layer: `Load.index`,
`Write.index`, `SelectBranch.select`, `IndexMapOp.coord_map`,
`StridedLoop.start`/`step`, `Cond.cond`, etc. Imports nothing from
other IR files.

| Symbol                                                                       | Role                                                                     |
|------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `Var`, `Literal`, `BinaryExpr`, `Builtin`, `FuncCallExpr`, `TernaryExpr`, `CastExpr`, `FlatIndex` | Expression nodes. Each has `eval(env) → value/ndarray`, `pretty()`, `substitute(mapping)`, `free_vars()`, plus the generic tree walks `subterms()` / `rebuild(fn)`. `FlatIndex` is a buffer coordinate's row-major offset, flattened against the buffer's shape at render time; it only appears in a kernel body. |
| `_ExprOps`                                                                   | Mixin: Python operator overloading for expression building; default `NotImplementedError` for `pretty`/`substitute`/`free_vars`. |
| `Expr`                                                                       | Union type alias.                                                        |
| `PLACEHOLDER_PREFIX`, `placeholder`, `is_placeholder`                        | Convention for output-coord placeholders in coord maps.                  |

## `frontend/ir.py`

Ops captured directly from PyTorch. Every one has a decomposition rule
under `pipeline/passes/frontend/decomposition/`; after that pass none of these
remain.

| Group         | Ops                                                                                                      |
|---------------|----------------------------------------------------------------------------------------------------------|
| Layout-only   | `TransposeOp`, `ReshapeOp`, `SliceOp`, `CatOp`, `UnsqueezeOp` — rewrite to `IndexMapOp`.                 |
| Compound math | `LinearOp`, `MatmulOp`, `SdpaOp`, normalization/reduction ops — rewrite to elementwise + reduce chains. |

## `tensor/ir.py`

Minimal IR fusion consumes. `IndexMapOp` is the unified layout-only op;
it replaces the frontend layout ops via `coord_map` expressions.

| Symbol                               | Role                                                           |
|--------------------------------------|----------------------------------------------------------------|
| `ElementwiseOp`                      | Per-element scalar function (`add`/`mul`/`where`/`exp`/`sin`/`cos`/…); unary `pad` is the frontend's exact zero-width identity. |
| `CastOp`, `BitcastOp`                | Numeric conversion and same-width bit reinterpretation.       |
| `RangeOp`                            | Static one-dimensional integer sequence.                       |
| `ReduceOp`                           | Collapse one axis via associative binary op.                   |
| `ScanOp`                             | Cumulative variant of reduce.                                  |
| `GatherOp`, `ScatterOp`              | Data-dependent reads / writes.                                 |
| `IndexMapOp` + `IndexSource`         | Unified layout-only op over `Expr`.                            |

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
behind the four sites that used to switch on the reduce op name (`Accum.render`'s
`+=` / `*=` / `fmax` / `fmin`, `kernel/ir._binary_combine_expr`, and
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
and the other carrier-agnostic checks) test exactly that — the carrier is a plain `Stmt` with the
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
stored is a schedule's question, not the statement's. The Tile lift realizes it as a serial launch axis over a state
buffer (`lowering/tile/010_lift`).

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
  combine — is what `Contraction` was, exposing `a` / `channels` / `b_trans` off `operands`. The `⊗` and the
  additive fold `Accum` appear in the DERIVED `Fold.loop`, never as stored loop syntax.
- Every ROLE derives from arity (`Fold.role`, never stored): `FREE` with no axis, `TWISTED` off the combine's
  claiming family, `CONTRACTION` off the bilinear reading alone, `PLANAR` otherwise. `ops.head` reaches the node
  through the projection wrapper. Scheduling reads these facts directly from the Fold tree; `Fold.lower()` is
  reserved for callers that consume Loop IR.
- A SCAN is a fold with a per-step `observe` — a pure `λ(axis, *state)` run after each combine whose fresh results
  only boundary output writes consume. Observation makes the
  stream order-visible, so an observed fold schedules as the serial fold only.

`Fold.lower(bound, stores)` flattens the term to the loop nest: a plain loop for every free coordinate the caller
left unbound, outermost the one the most terms share, the reduce loop of each term innermost, every term placed at
the shallowest scope binding its free coordinates, and each boundary store right after the term defining its value. Loops carry NO algebra and no annotation, so the derived nest depends
only on what is stored, which is what makes every identity of the term a digest of its lowered body — there is no
separate term hasher.
The `TileOp`'s body identity is the canonical digest of the nest `lower()` derives (the body is the
term's normal form); the variant key (`identity_key(with_io=True, with_knobs=True)`) folds the schedule-free body
identity with the knobs; and the deploy join key (the deploy identity (`identity_key(with_io=True)`), over
`TileOp.loop_body`) types the roles the body reads its buffers through, so term re-spellings and cluster-sibling ops
that lower alike share schedule evidence.
`Fold.deps()` exposes names captured outside the lift params, including captures reached recursively through operand
edges. A contraction deliberately hides its pure lift body from generic nested-body walks, so this direct dependency
surface is what keeps an operand's captured statistic ordered before the contraction that reads it. A read walk
STOPS at this rollup — the **`Stmt.deps_deep` trait** (a `Stmt`-protocol member beside `pure`, conservative `False`
default; `Fold` opts in) tells `_member_reads` that `deps()` already answers for the whole subtree, scope-correctly.
Re-walking the lift's flat namespace cannot see its params, so an operand-supplied name read inside the lift leaked
out of every enclosing lambda as a phantom capture: an operand-supplied name is not an enclosing capture.

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
the score is the sub-cone of the lift alpha-equal to the pivot's own per-element map (operand for operand, through a
projection's
components), and what remains, in role order, must equal a channel's pattern by canonical form. A click gives the
role-to-name map and the recipe instantiates itself by renaming; no recipe names a term's variables. Online softmax
and flash attention are one recipe: the expectation channel joins by the same call, the pivot then being the fused
fold. **Example** — the online-softmax carrier: state `(m, d)`, partial `(score, 1)`, identity `(−inf, 0)`, merge
`m_new=max(m,s); d=d·exp(m−m_new)+exp(s−m_new); m=m_new`.

**The λ-foldMap primitives** (`ir/pure/lam.py`) — the finished algebra vocabulary the tile IR
stores against (see the tile-lowering ARCHITECTURE for the storage story). `Lambda(params, body, results)` is the ONE
binder kind over the reused stmt vocabulary — a `Body` of PURE stmts only (ANF ≙ a let-chain), validated in
`__post_init__` via the **`Stmt.pure` trait** (declared on the `Stmt` interface, conservative `False` default;
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
| `Assign`                     | SSA body stmt: `name = op(args)` with `op: ElementwiseImpl`.                                                      |
| `Accum`                      | Reduce accumulator: `name = op(name, value)` inside a reduce `Loop`. Initialized to its op's identity. ``axes`` lists the reduction axis names — propagated through Sigma renames (including σ-splits via `Expr.free_vars()`); the escape-analysis helper derives cross-thread cooperativity from ``axes ∩ enclosing ThreadTile.axes``. |
| `Carry`                      | The next value of one cell of a carried state: `name[index...] <- value`, holding `seed` before the first step. No op: a state folds nothing. |
| `Pre`                        | Read of one cell of a carried state: `name = pre carrier[index...]`, the previous step's value inside the loop that carries it. |
| `Init`                       | Explicit `<dtype> name = identity;` seed at this scope (`name` + scalar `identity` + `dtype`). Used for a carried state's seed (one per component), emitted above the streaming `Loop`. Scope-bound (never hoisted); shadows a deeper same-named `Accum` init. |
| `Let`                        | Pure binding of one `Expr` to a name: a scalar literal (the twisted carrier's injected `1`), a precomputed integer index, a `FlatIndex` offset. Scoped like an `Assign`; legal inside a stored `Lambda`. |
| `Write`                      | Write an SSA value to output at `index`.                                                                          |
| `Select` + `SelectBranch`    | Coord-predicated binding (replaces the old Mux).                                                                  |
| `Loop`                       | Serial iteration block: `axis` + nested `body`.                                                                   |
| `StridedLoop`                | Strided iteration (`start`, `step`) — cooperative thread-stride loop reused by Tile/Kernel IR.                    |
| `Cond`                       | If/else block over an `Expr` predicate.                                                                           |
| `Stmt`                       | Base class — every body statement subclasses it. Leaves and control-flow nodes live in `ir/stmt/`.               |

Body walkers: `iter_body(body)` (pre-order; powers `for s in loop_op`),
`map_body(body, fn)` (transformer), `Stmt.rewrite(rename_ssa, sigma)`
(per-stmt copy with SSA rename + Expr substitution),
`Stmt.pretty(indent)` (rendered lines for kernel dumps; block stmts
recurse via `pretty_body`).

CUDA scalar rendering goes through `stmt.base.op_to_expr`. Boolean masks retain the historical f32 SSA convention,
so Torch's `bitwise_not` spelling renders as logical zero-test (`mask == 0`); explicitly bool-stamped values use the
same semantics. Integer complement is not inferred from that name and fails closed until it has a typed consumer.
The optional readable-source fold keeps a single-use `Assign` named when any argument's stamped dtype differs from
the result dtype, so the target-aware `Assign.render` path remains responsible for conversions such as
`__half2float`.

Dependence cones (`ir/stmt/body.py`): `Body.backward_cone(roots)` builds a `Cone` —
the subset of the body's immediate stmts closed under SSA dependence (a wrapper joins as a unit; internally-bound
axes excluded), plus `external_reads`, the names read from outside (axis vars and enclosing/sibling scopes alike).
Construction never fails: unresolved names are data, and chaining scope levels means seeding the next level's
`backward_cone` with the previous one's `external_reads`. `Body.defs_die_at(members, roots=…, allowed=…)` is the
matching escape check (may the cone be cut out, with only the designated consumers reading its roots?). This is
the shared substrate behind the rules that slice cones (the demoted-operand producer cut in
`lowering/tile/030_cut`) — eligibility judgments stay in the rules, per
`pipeline/passes/ARCHITECTURE.md`.

`backward_cone` resolves reads by NAME over a body it assumes is SSA, so it is only sound where one name has one
def. `Lambda.cone` is the caller that cannot assume it: a stored combine takes its states in as params and writes
them back on the way out to spell its results (`Recipe.program`), so a read of the INCOMING state would resolve
forward to the write-back. A param names itself, and `cone` keeps it that way by hiding the re-bindings from the
walk. Any other caller that cones a non-SSA body owes itself the same reading.

`rewrite` has two distinct rename channels that must stay disjoint:
`rename_ssa` carries **SSA-name** renames, `sigma` carries **axis**
substitutions. `Load`/`Write` index exprs apply *both*
(`_rename_ssa_vars_in_expr(sigma.apply(e), rename)`) so an indirect
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
scope re-binds before descending. `normalize.dedup_loads` applies the same rule while threading its own per-scope
environment. σ has the same hazard with axis names, which collide across a tree by design (a cone statistic's axis
may spell the same as the enclosing contraction's): `fold.subst_free(stmt, sigma)` is σ's hygienic form — it stops at
a `Loop` / reducing `Fold` binder that re-binds a substituted name, and is what the smem compute fill substitutes
cell coordinates through.

### `ir/stmt/normalize.py` — executable body normalization

Pure `body → body` passes run from `LoopOp.__post_init__` so every
constructed `LoopOp` (including intermediate fusion results) is
canonicalized before validation:

- `topo_sort_siblings` — stable Kahn reorder so SSA defs precede their uses
  within each body (fixes splicer-produced use-before-def).
- `drop_size_one_free_axes` — inline extent-1 free Loops.
- `drop_size_one_reduce_axes` — collapse a canonical extent-1 reduction to its single monoid update. This includes
  decode-softmax values that fusion hoists into the enclosing scope; copy-alias elimination then removes the identity
  update before total reduction lifting.
- `canonicalize_free_axis_order` — sort outer free Loops by their row-major position in boundary writes, so output
  storage geometry rather than axis spelling decides the nest. When the writes cannot totally order the chain, axis
  roles and the least complete alpha-renamed form decide the order. A cross-CTA partition coordinate occupies the
  workspace's leading index, so the same rule keeps it outside the axes it partitions without a naming convention.

- `eliminate_copy_aliases` — drop `y = copy(x)` Assigns. Each nested body owns its alias map, so source spellings
  reused by sibling scopes remain separate binders.
- `unify_sibling_reduce_axes` — rename sibling reduce Loops whose reduce-axis Load positions overlap so they share one
  canonical axis name (softmax's max + sum sweeps; the two matmul reductions in `silu(x@Wg) * (x@Wu)` that both index
  `x` at the same K slot). A position is `(source, dim, anchor, coefficient)`, read through `affine_form`: a blocked
  reduce indexes its stream at `o·B + i` and still walks that dimension, while the anchor keeps `o·B + i` apart from
  `o·B + 32 + j`, which walk different halves. Union-find groups all transitively-overlapping Loops at one scope.
- `merge_sibling_reduce_loops` — concatenate sibling reduce Loops that share `axis.name` / `extent` into one Loop body.
  Every gate is phrased over what the second Loop reads from its ENCLOSING scope (`free_names` — what it uses and does
  not bind itself): it must read no name the first body defines (blocking softmax-style sequential reduces where
  sum-exp reads `acc_max`), and no between-stmt def. Names both bodies merely happen to bind are a COLLISION, not a
  dependence — two alpha-equal copies of one cone always share every spelling — so the incoming body's copies rename
  apart. Only a name the incoming Loop still binds after it closes (its immediate carriers, which `Loop.render`
  declares ahead of the loop) refuses, because the rename cannot reach that name's readers outside. Eliminates the
  duplicate K traversal in patterns like `silu(x@Wg) * (x@Wu)`, and the duplicate score pass between the channels of a
  blocked twisted carrier; subsequent normalization collapses the duplicate loads, and the lowering passes stage both
  weight tensors symmetrically.
- `split_invariant_divides` — rewrite `divide(x, y)` into
  `reciprocal(y) + multiply(x, recip)` when `y` is loop-invariant
  w.r.t. some axis `x` depends on, so the rcp can hoist out of the
  inner loop and the per-iter cost drops from XU divide to FMA
  multiply.
- `hoist_loop_invariants` — pull loop-invariant Assigns out of reduce
  Loops. The hoisted set is closed under the scope's ordering constraints, the same ones the sibling order respects:
  the consumer of an accumulator a pinned reduction exports, a read of a buffer the loop writes, and anything behind
  a barrier or a declaration stay in the loop. Effect summaries are cached on immutable statements, and
  `Body.axis_dependencies` retains only the axes reachable from each definition. Long SSA chains therefore remain
  linear in definitions × loop depth instead of materializing the quadratic full SSA dependency closure.
- `dedup_loads` — after expression simplification, keep one `Load` for each identical
  `(input, index, width, dtype)` read in a scope and rewire every scalar or vector lane. A write invalidates retained
  reads of that buffer, including around a nested scope with a write. The same walk keeps one `Assign` per identical
  operation over identical arguments and one `Accum` per identical accumulation — a value the loop tree computes
  twice (a contraction spelled on both sides of a cut seam, a repeated pure expression) folds to one definition, and
  an accumulator alias carries out of the loop that defined it to the scope that reads the sum. This is
  canonicalization for every Loop / Tile body, not a fusion profitability decision; the structural key inherits it,
  so two bodies that differ by a repeated computation key alike.
- `rename_ssa_sequential` — cosmetic: `Load` names become `in0, in1, …`, accumulator state becomes `acc0, …`, and
  every other definition becomes `v0, v1, …`, in lexical definition order. Names stay globally unique while each
  nested body tracks its own binders, so sibling scopes may reuse the same source spelling without collapsing. Axis
  renames reach conditions, reduction metadata, and `Window` parent/base/bound metadata as well as indices; a
  reduction's axis tuple is canonicalized as a set. SSA values travel only through the rename channel, never `sigma`,
  so indirect indices cannot be renamed twice.
- `sort_commutative_args` — sort `Assign.args` for commutative ops
  (`add` / `multiply` / `maximum` / `minimum`) so two bodies that
  differ only by argument order land in the same canonical form.
  Runs last so the sort key is the post-rename canonical SSA / buffer
  names.
- The final ordering pass canonicalizes integer coordinate expressions, builds one colored relation graph for the
  complete body tree, and chooses one dependency- and effect-valid statement order. Vertices represent scopes,
  statements, lexical definitions, axes, source axes, and external buffers; colored relations retain operand
  positions, captures, aliases, nesting, resource hazards, and ordered execution protocols. The graph is independent
  of source order and spelling, and it rides the normalized body: structural identity labels the same graph again
  under its own buffer coloring instead of building it a second time. A scope's definitions bind its reads in any
  order and shadow an enclosing binding of the same spelling; a deeper scope's definition binds nothing read above
  it, so the block still depends on the enclosing definition it reads.
- A standard smaller-half worklist computes the equitable partition in
  `O((vertices + relations) log vertices)` relation visits. Exact individualization is isolated to partitions that
  refinement cannot distinguish; no exact near-linear worst-case graph-canonization algorithm is known. Canonical
  vertex ranks then serve as the optional tie-break for `Body.topological_order`, a heap-based Kahn sort. Ready nested
  scopes stay ahead of leaf epilogues so normalization does not widen schedule search or obscure contractions.

### `ir/stmt/identity.py` — structural identity

`Body.identity()` takes the executable normal form (`normalize_body`), labels its relation graph with the external
buffers colored by type, assigns the buffers canonical names by rank, and optionally collapses operations to their
compute-unit cluster; `Body.structural_key()` is its digest. Clear external argument names remain on executable
bodies; the identity body is digest material only and must never be executed. One normal form serves both, so a
body keys the same whether it was held bare or constructed as a Loop op.

- The same relation graph that orders statements ranks external buffers without using their spelling. Identity assigns
  `b0`, `b1`, … by those ranks, preserving aliasing while making discovery order irrelevant, and materializes the
  labeled order directly — no second ordering pass after the rename.
- The typed identity (`identity_key(with_io=True)`) colors each buffer vertex with its dtype and hint-free shape, so
  the types bind to the ROLE a buffer plays. Two kernels whose typed argument lists read alike in declaration order
  but assign the types to different roles key apart; declaring the same roles in another order keys the same. The
  identity material also names which buffer fills each role (`Op.canonical_buffers`), which is how the kernel cache
  rebinds a hit.
- Optional operation clustering replaces each elementwise operation with its compute-unit representative before
  normalization. It is the only operation rewrite owned by identity; all executable canonicalization stays in
  `normalize_body`.

The key is `digest(form(canonical_body))`, not the human `pretty()` rendering. The exact and compute-unit-clustered
forms are cached on each immutable `Body`. Two bodies that differ only by SSA or axis names, argument spelling and
discovery order, dependency-valid statement order, or equivalent commutative and affine expression spelling
therefore share a structural key. Use it when deduplicating candidate bodies in search.

### `ir/expr.py` — Expr simplification

`simplify` (called inside `normalize_body`). Generic bottom-up Expr rewriter:
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
the row outside a div/mod" can see it. The `loop/canonicalize` axis re-fusion is
that consumer, and its answer decides whether a packed matmul binds a
contraction at all.

### `loop/splicer.py` — LoopOp merger

The machinery `pipeline/passes/loop/fusion/010_merge_loop_ops.py` calls to splice a DAG of `LoopOp` nodes. `Sigma`
(from `ir/sigma.py`) is the axis-substitution bookkeeping threaded through the merge.

`splice_graph` resolves each internal Load through `Graph.producer(buffer)`, so primary and secondary output buffers
use the same path. Every graph output supplies an explicit `(loop tag, Write.output)` root. Separate terminal loops
therefore seed one worklist; its one binding table shares equal upstream demands across output ports instead of
inlining a shared producer per consumer. The single-sink convenience form still derives the unique terminal loop and
selects all its Writes. Every `_NotSupported` carries a reason string, logged at DEBUG by `splice_loops` —
`compile -vv` shows which pattern a rejected edge hit.

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

Before splicing, `loop/lifting/090_spell_store_rounding` turns a public store that narrows an `Accum` into an ordinary
typed `copy` statement. A decomposition may route that accumulator through one transient, shape-only buffer before a
pass-through LoopOp writes the public buffer; that direct load retains the accumulator's implicit f32 dtype, so the
same rule spells its public conversion. An actual `Assign` computation over private reduction state remains untyped:
normalization and softmax therefore retain f32 internal state rather than narrowing it at an inferred projection edge.
`splice_graph` then preserves the explicit conversion through its ordinary statement path and reconstructs no dtype
boundary from source provenance or graph topology.

Construction is bounded per statement: the dedup table shares each `(stmt, emit scope, σ)` binding, and in
every legitimate splice no single statement takes more than a handful of distinct bindings. A recurrence-shaped
region — each stage re-demanded under compositions of σs, DeepSeek-V4's 20-iteration Sinkhorn chain being the live
case — multiplies bindings per stage instead of deduplicating, and such a merge cannot be constructed at any budget.
The first statement past the cap stops the splice. The doom is structured (`UnfusableStmt` names the offending
loop) and surfaced to the fusion pass on request, which drops that loop plus its downstream closure from the region
and retries — so one doomed chain costs only itself, not every other merge in its region. This is a termination
bound, not a fusion-quality gate: placement still owns every cut on a merge that CAN be built, and the refusal must
stay cheap because the greedy policy re-runs fusion on every candidate graph it prices.

Each splice memoizes `Expr.free_vars()` by expression identity while placing dependencies. Sigma expressions remain
live for the splice, and identity avoids both repeated coordinate-tree walks and the recursive structural hashing a
global cache would require; the memo is discarded with the splicer.

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

### `loop/builder.py` — fluent construction

`LoopBuilder` constructs merged `LoopOp` bodies for the fusion splicer without spelling out every `Loop(Axis(…))`
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

| Symbol             | Role                                                              |
|--------------------|-------------------------------------------------------------------|
| `KernelOp`         | Graph-op wrapper around a `Tile`-rooted body. One per kernel.     |
| `Smem`             | `__shared__` array allocation (name + dtype + extents + optional `align`). Swizzled TMA operand slabs align to their full swizzle atom (`8 × swizzle_width` B: B128→1024, B64→512, B32→256) — the coordinate-only `ldmatrix` XOR only reproduces the hardware's absolute-address swizzle when the base zeroes the swizzle's source-address bits; non-swizzled TMA keeps 128 B, fp16 16 B. `pack_smem` (the shared pool packer used by `smem_bytes` and the renderer) pads each buffer to `max(sizeof(dtype), align)` so the static-vs-dynamic gate and the launch-time dynamic-pool size agree. |
| `Let` (a `stmt/` leaf) | Pure binding of one expression to an SSA name: a scalar literal, an integer index bound once outside a nested hot loop, or a flattened buffer coordinate (`FlatIndex`) reused across a copy's trips. |
| `Sync`             | Thread barrier: CTA-wide `__syncthreads()`, warp-scope `__syncwarp()`, or a named `bar.sync` over a thread subset inside a warp-specialized branch. |
| `TreeHalve`        | Cross-thread tree reduction over a smem buffer.                   |
| `RegFragment`      | Per-thread `mma.sync` register array declaration, zero-initialized for C. The established m16n8k16 layout uses A/B/C counts 4/2/4 for f16/f16/f32; the Volta m8n8k4 layout carries explicit 2/2/8 counts because one instruction realizes four PTX cells arranged as one logical 16×16 tile. Carries instruction shape, dtype, and an optional explicit register count. The opaque `nvcuda::wmma` nodes remain retired. |
| `LdmatrixLoad`     | Load one operand into a `RegFragment`. The m16n8k16 layout can use `ldmatrix.sync.aligned.m8n8.x{4,trans}.b16` from shared memory or a global-memory-direct gather with the same lane map. SM70 has no `ldmatrix`, so gmem-direct and unpaired staged Volta paths use cooperative gathers. A materialized canonical-B tile with even M/N fragment counts instead derives paired crosswise-A and B-congruous shared layouts; each 128-bit load drains two logical fragments. Its four computation groups duplicate the appropriate A or B quadrant. `b_trans=True` marks a `[N, K]` weight and selects the corresponding transposed gather. Guards clamp M/N lanes and zero masked K elements in both layouts. A 1-byte staged slab (`byte_slab=True`) has no `ldmatrix` below sm_100a and drains through the cooperative gather too; when it also carries a `scale_buffer` the slab holds a weight stored as bytes with one scale per k block in that companion slab — PACKED PAIRS (an NVFP4 weight: one byte, two K elements, decoded through the value table) or fp8 bytes (one element each, converted by the hardware cvt and multiplied by an f32 scale before the round to the fragment). |
| `MmaSyncPtx`       | Inline PTX for either `mma.sync.aligned.m8n8k4.row.{col,row}.f32.f16.f16.f32` on the Volta fragment layout or the established `mma.sync.aligned.m16n8k16.row.col.{f32,f16}.{f16,bf16}.{f16,bf16}.{f32,f16}` family. Paired B-congruous loads select row/row; the other Volta paths retain row/col. The renderer includes only the selected family's prelude, so SM70 never parses newer `ldmatrix` or m16n8k16 assembly. The BLOCK-SCALED fp4 form (`m16n8k64`, `kind::mxf4nvf4`) additionally carries `sfa_frag` / `sfb_frag`: both multiplicands are packed e2m1 pairs and the instruction applies one ue4m3 scale per 16 K elements itself, so the call passes those two scale registers where the others repeat the accumulator. Its data fragments reuse the fp8 byte loaders — the k64 4-bit lane map is the k32 8-bit one, over a row of K/2 bytes — leaving only the scale loaders new. It assembles only for the arch-suffixed consumer-Blackwell target, which the plan requests through `KernelSpec.arch_specific` (the flag TMA also sets). |
| `WgmmaDescriptor`  | The 64-bit shared-memory matrix descriptor a `wgmma` operand is read through: start address, leading and stride byte offsets and the swizzle mode, built by `emmy_wgmma_desc` from the slab address the same swizzled TMA or cp.async fill deposited. |
| `WgmmaMma`         | One `wgmma.mma_async.sync.aligned.m64nNk16` cell over N/8 of the per-warp 16×8 C fragments (instruction register d[4j+k] is fragment j, register k); A from a descriptor or a 4-register fragment, B always from a descriptor; the transpose bits are template arguments of the generated wrapper. |
| `WgmmaFence` / `WgmmaCommit` / `WgmmaWait` | The asynchronous discipline around a chunk of `WgmmaMma` cells: fence before the first cell after the accumulators were touched, commit after the chunk, wait before any accumulator read and before the ring slot is released. |
| `FragmentPromote`  | Fold a packed f16-accumulate C fragment into its f32 shadow fragment and rezero it (`emmy_mma_promote_f16acc`: PTX `cvt.f32.f16` + add per element) — the chunked-accumulation promote pairing the f16-acc `MmaSyncPtx`. The mma chain accumulates in f16 at full rate; each K chunk (the staged bk slab, every `_F16ACC_STEPS` gmem-direct atom steps) folds into the f32 shadow, bounding the f16 rounding to one chunk while the store/epilogue read f32. |
| `FragmentApply`    | The one pointwise node over a C fragment. Each argument resides in another fragment, a per-row register pair, a cell-uniform scalar, a predicate over the element's absolute coordinates (`COORD`), or a global-memory load template at those coordinates (`GMEM`). A coordinate mask is a `where` over a `COORD` predicate — the masked branch takes the carrier's finite identity, avoiding `-inf - -inf` in an all-masked chunk, and an additive mask applies its keep op first; an additive bias is an `add` of a `GMEM` operand. The atom's fragment-layout descriptor supplies the element count and row mapping, so the same leaf serves m16n8k16 and Volta m8n8k4. |
| `FragmentRowReduce` | Fold one warp's C fragments along the atom's N direction, per ROW: in-lane columns combine first, then the layout's `__shfl_xor` masks combine the column-group lanes. The resulting register pair is what a `FragmentApply` broadcasts as a `ROW` operand. The chunk tier's pivot and summed channel partials are exactly this, which is why the tier needs the chunk inside one warp column. |
| `FragmentRepack`   | Convert score C fragments into one 16-bit A fragment in registers for a paired contraction. m16n8k16 consumes two adjacent fragments; Volta m8n8k4 selects one four-column slice from a logical 16-column fragment with warp shuffles. |
| `RegStore`         | Layout-aware per-lane epilogue store: four C elements for m16n8k16 or eight elements covering the four Volta output quadrants for m8n8k4. A paired Volta tile derives the matching interleaved 32×32 accumulator map from its cell position; it is not a schedule field. Adjacent elements leave as one packed pair when N is contiguous, including under an M-only tail guard; an N guard or strided physical orientation keeps scalar stores. Stores f32 directly or downconverts to f16. An optional epilogue is a pure `Lambda` over the projection tail's own `Load` / `Assign` / `Select` stmts, its leading params bound to the store's fragments; it is evaluated at each element's own coordinates. |
| Shared from `tile` | `Tile` (launch geometry); from `ir/stmt/`: `Loop`, `StridedLoop`, `Load`, `Assign`, `Accum`, `Init`, `Let`, `Write`, `Select`, `Cond`, `ZeroPrologue`. |

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
