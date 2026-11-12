//! The statement types, declared as data: every statement type is a struct whose fields carry a type and the roles
//! they play. The declarations are one macro ([`with_statement_types!`]) so the Python extension generates its classes
//! from the same list, and [`define_statement_types!`] turns the list into the structs, the `Statement` enum and the
//! protocol walks over the roles.
//!
//! A role says what a field is to the rest of the compiler: `DEFINES` / `READS` / `UPDATES` / `LOCAL` for SSA names,
//! `BUFFER_READ` / `BUFFER_WRITE` / `LOCAL_BUFFER` for buffers, `EXPRESSION`, `AXIS` and `AXIS_NAME`, `BODY` for a
//! child body, `STMTS` for carried statements, `TERM` for the body of a pure term whose names are private,
//! and `RECORD` for records whose own fields play roles. A number where a name, a buffer or an expression is expected
//! is plain data ([`NumOr`]). A field with no role is plain data.

/// The roles a field's values play, as a set.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Roles(u32);

impl Roles {
    pub const NONE: Roles = Roles(0);

    pub const fn or(self, other: Roles) -> Roles {
        Roles(self.0 | other.0)
    }

    /// Whether any role of `other` is one of these.
    pub const fn contains_any(self, other: Roles) -> bool {
        self.0 & other.0 != 0
    }
}

impl std::ops::BitOr for Roles {
    type Output = Roles;

    fn bitor(self, other: Roles) -> Roles {
        self.or(other)
    }
}

pub const DEFINES: Roles = Roles(1 << 0);
pub const READS: Roles = Roles(1 << 1);
pub const UPDATES: Roles = Roles(1 << 2);
pub const LOCAL: Roles = Roles(1 << 3);
pub const BUFFER_READ: Roles = Roles(1 << 4);
pub const BUFFER_WRITE: Roles = Roles(1 << 5);
pub const LOCAL_BUFFER: Roles = Roles(1 << 6);
pub const EXPRESSION: Roles = Roles(1 << 7);
pub const AXIS: Roles = Roles(1 << 8);
pub const AXIS_NAME: Roles = Roles(1 << 9);
pub const BODY: Roles = Roles(1 << 10);
pub const STMTS: Roles = Roles(1 << 11);
pub const TERM: Roles = Roles(1 << 12);
pub const RECORD: Roles = Roles(1 << 13);
/// The field's expressions are of the `Index` class (the declaration's `: Index`).
pub const INDEX: Roles = Roles(1 << 14);
/// The field's expressions are of the `Value` class (the declaration's `: Value`).
pub const VALUE: Roles = Roles(1 << 15);

/// The SSA-name roles a rename maps.
pub const NAMES: Roles = DEFINES.or(READS).or(UPDATES).or(LOCAL);
pub const DEFINING: Roles = DEFINES.or(UPDATES);
pub const READING: Roles = READS.or(UPDATES);
pub const BUFFERS: Roles = BUFFER_READ.or(BUFFER_WRITE);
pub const CARRIED: Roles = STMTS.or(TERM);

/// What a statement type is as a whole, as a set of flags.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StatementFlags(u32);

impl StatementFlags {
    pub const NONE: StatementFlags = StatementFlags(0);

    pub const fn or(self, other: StatementFlags) -> StatementFlags {
        StatementFlags(self.0 | other.0)
    }

    pub const fn contains(self, other: StatementFlags) -> bool {
        self.0 & other.0 != 0
    }
}

pub const PURE: StatementFlags = StatementFlags(1);
pub const SIDE_EFFECTS: StatementFlags = StatementFlags(2);
pub const DEPS_DEEP: StatementFlags = StatementFlags(4);
/// The statement type checks its fields on construction (`validate`, in the protocol's checks).
pub const CHECKED: StatementFlags = StatementFlags(8);

/// Every statement type and record, declared once. `$callback` receives the list and generates from it.
#[macro_export]
macro_rules! with_statement_types {
    ($callback:ident) => {
        $callback! {
            statement_types {
                /// Read a value (or N consecutive values) from an external input buffer.
                ///
                /// Scalar (`width == 1`): one SSA binding `names[0]`. Vector (`width > 1`): N SSA bindings; lane k reads
                /// `index[:-1] + (index[-1] + k,)` into `names[k]`. The `050_vectorize_loads` pass widens a run of consecutive
                /// scalar Loads into one Load with `len(names) > 1`. Every pass before that produces scalar Loads and can keep
                /// using `s.name` / `s.index`.
                ///
                /// A load takes its fields as declared: `names` is a tuple, one name for a scalar load. Use `.is_vector` /
                /// `.is_scalar` / `.width` / `.name` (asserts scalar) / `.names` to test shape.
                ///
                /// A scalar Load is rendered as a literal binding (`float name = <value>;`) when `ctx.literal_constants` carries
                /// a value for `input` — the scalar-constant-inlining path populates that map at the cuda lowering boundary so
                /// kernels can embed `ConstantOp` values directly instead of taking them as `float*` parameters.
                ///
                /// `dtype` is optional and, when set, names the source-buffer element dtype that this Load produces. The
                /// `030_stamp_types` Kernel-IR pass populates it once before any analytical pass runs; downstream passes
                /// (vectorize_loads, demote, etc.) read it instead of reaching for the matcher-populated
                /// `KernelOp.inputs`/`outputs` side channels. `None` keeps the legacy render-time inference path working for
                /// tests that construct Loads by hand without dtype.
                ///
                /// `carried` marks a staged copy that crosses the chunk loop's back edge. The names and their packed vector are
                /// the ones an earlier Load of the same names DECLARED ahead of the loop. `"load"` re-assigns the packed vector,
                /// through the read-only (non-coherent) path — the operand is one the kernel never writes — so the loaded chunk
                /// survives into the next iteration as the few registers the vector holds. From there a read of one of the
                /// names reads its lane of the vector where the read stands: nothing spreads the vector into the names.
                /// Measured on the V100: spreading it at the load keeps every element live across the mma stream and ptxas
                /// then serializes the loads behind it (a 128x128 partial 84 us against 67).
                shared Load {PURE | CHECKED} {
                    names: Names [DEFINES],
                    input: Name [BUFFER_READ],
                    index: Exprs [EXPRESSION | READS] : Index,
                    dtype: Option<DType> [] = None,
                    carried: Name [] = Name::from(""),
                }
                /// The one scalar definition: `name = value`, a pure value of the `Value` class over SSA names and
                /// literals, one operation of the table: the normal form gives every nested operation node a definition
                /// of its own, and a sum in affine form is one node. `dtype` overrides the promotion of the operands when set: the
                /// `demote_to_write_dtype` Kernel IR pass stamps it on an `Assign` whose result only feeds dtype-narrower
                /// consumers, so a softmax or RMSNorm epilogue feeding an `__half*` store computes in `__half` end to
                /// end without spurious promotions.
                shared Assign {PURE} {
                    name: Name [DEFINES],
                    value: Expression [EXPRESSION | READS] : Value,
                    dtype: Option<DType> [] = None,
                }
                /// Reduce accumulator — declares-and-folds in one statement.
                ///
                /// Semantics: `name = op(name, value)` inside the enclosing reduce `Loop`, `value` a `Value` expression, so
                /// `acc += a * b` is one statement; the reduce combine `op` is the statement's effect and stays its own field. Before the first iteration `name` is
                /// initialized to `op.identity` (the combine's neutral element). After the Loop completes, `name` is an SSA
                /// binding visible in the enclosing scope, carrying the finalized reduced value.
                ///
                /// `op` is an `ElementwiseImpl` — typically one of `ADD` / `MAX` / `MIN` / `MUL`. It defines both the combine
                /// operation and the accumulator's identity value. Multiple `Accum` stmts targeting the same `name` in one reduce
                /// Loop must agree on `op`.
                ///
                /// Default op is `add` — fixtures that sum values can omit `op=`; `max` / `min` / `mul` must be passed
                /// explicitly.
                ///
                /// `axes` is the tuple of axis names this Accum reduces over — populated by the lifting pass (one axis name from
                /// the wrapping reduce `Loop`) and propagated by every rewrite that renames axes via Sigma. Used by the escape-
                /// analysis helper to derive cross-thread cooperativity (`Accum.axes ∩ enclosing ThreadTile.axes`); empty tuple =
                /// legacy/handwritten path with no reduction-axis info.
                ///
                /// The accumulator is read where it stands: within one step of its loop, a read of `name` ahead of the update
                /// is the value the step started from (the step before, or the seed), a read behind it the updated value. The
                /// order keeps every read on its side of the update.
                shared Accum {} {
                    name: Name [DEFINES],
                    value: Expression [EXPRESSION | READS] : Value,
                    op: ElementwiseOp [] = ElementwiseOp::Known($crate::elementwise::Operation::Add),
                    dtype: Option<DType> [] = None,
                    axes: Names [] = Names::from([]),
                    base: Option<Name> [READS] = None,
                }
                /// The next value of one cell of a CARRIED STATE: `name[index] <- value`.
                ///
                /// A recurrence's state, which a fold is not: its steps run in order, its step is any computation, and it holds
                /// `seed` before the first. So it has no op — an op would declare an algebra (a reorderable sum) the state does
                /// not have — and nothing else to declare either: the loop that carries it is the nearest enclosing `Loop` whose
                /// axis `index` does not read (`protocol::shared::carried_cells`), and its shape is those of
                /// the loops over its cells, each entry of `index` a bare axis. The loop over the steps therefore sits OUTSIDE
                /// the loops over the cells, which is what lets a step read OTHER cells of its own state (a delta rule's `k @ S`)
                /// through `Pre`; a scalar `Accum` lives inside the cell loops and cannot. One statement defines a state;
                /// a cell it skips keeps its value.
                shared Carry {} {
                    name: Name [DEFINES],
                    value: Name [READS],
                    index: Exprs [EXPRESSION] : Index,
                    seed: NumOr<Name> [BUFFER_READ],
                    dtype: Option<DType> [] = None,
                }
                /// A read of one cell of a carried state: `name = carrier[index]`.
                ///
                /// Inside the `Loop` that carries it, the value the PREVIOUS step left (the seed on the first step) — no read of
                /// a step sees what that step defines; after that loop closes, the last step's. `index` is any coordinate
                /// expression, so a step may read a cell other than the one it defines — under a reduce over it, a contraction
                /// over the state. See :class:`Carry`.
                ///
                /// A read like a `Load`, so a term may hold one: a carrier read is a slab over one `Pre` (`Fold.carrier_read`),
                /// the way a gmem read is a slab over one `Load`.
                shared Pre {PURE} {
                    name: Name [DEFINES],
                    carrier: Name [READS],
                    index: Exprs [EXPRESSION | READS] : Index,
                }
                /// Explicit accumulator / carried-state seed at this scope: `<dtype> <name> = <identity>;` — a scope-local
                /// declaration.
                ///
                /// A fold's OWN seed does not ride here — it lives on its `Accum` and is derived by `Loop`'s render from
                /// `op.identity`. What produces an `Init` is the CLAMP-TO-IDENTITY masking: a masked tail's `Select` needs the
                /// identity as an SSA name to select, so the masked ILP copy (`lowering/kernel/_factor`) and the compute fill's K
                /// mask (`lowering/kernel/_atom._k_masked`) each bind one. `identity` is the neutral element (one scalar — 0 / 1
                /// / -inf), held directly.
                shared Init {} {
                    name: Name [DEFINES],
                    identity: Float [],
                    dtype: DType [KW_ONLY],
                }
                /// A pure binding of one expression to an SSA name: `<dtype> <name> = <value>;`.
                ///
                /// Three things reach a body this way — a scalar literal (the twisted carrier's injection binds the `1` online
                /// softmax folds each element with, so the lift can return it by name), an integer index precomputed once outside
                /// a nested hot loop, and a buffer coordinate's flattened offset
                /// (:class:`~emmy.compiler.ir.expression.FlatIndex`) reused across a copy's trips. Pure, so it is legal inside a
                /// stored `Lambda` body. A float literal's `dtype` is stamped at kernel lowering like an `Assign`'s (f32 when
                /// nothing narrows it); an untyped index renders as `auto` and takes its expression's type. A bare number coerces
                /// to a float `Literal`.
                shared Let {PURE} {
                    name: Name [DEFINES],
                    value: Expression [EXPRESSION | READS] : Value,
                    dtype: Option<DType> [] = None,
                }
                /// Write an SSA value (or N consecutive values) to `output`.
                ///
                /// Scalar (`width == 1`): stores one SSA value at `index`. Vector (`width > 1`): stores N values; `values[0]`
                /// goes to `index`, `values[k]` goes to `index[:-1] + (index[-1] + k,)`. The `080_vectorize_stores` pass widens a
                /// run of consecutive scalar Writes into one Write with `len(values) > 1`. Every pass before that produces scalar
                /// Writes and can keep using `s.value` / `s.index`.
                ///
                /// A write takes its fields as declared: `values` is a tuple, one name for a scalar write. Use `.is_vector` /
                /// `.is_scalar` / `.width` / `.value` (asserts scalar) / `.values` to test shape.
                ///
                /// `output` is the destination buffer's name (matches the owning graph node's id, or — for multi-output kernels —
                /// one of its output buffer names). `index` uses axis Vars to compute the per-dim offset.
                ///
                /// `value_dtype` is optional; when set, names the SSA-value dtype being stored. Stamped by `030_stamp_types`;
                /// downstream passes read it instead of querying `ctx.ssa_dtypes` at render time.
                shared Write {SIDE_EFFECTS | CHECKED} {
                    output: Name [BUFFER_WRITE],
                    index: Exprs [EXPRESSION | READS] : Index,
                    values: Names [READS],
                    value_dtype: Option<DType> [] = None,
                    atomic: bool [] = false,
                    swizzle: Name [] = Name::from("NONE"),
                }
                /// Coord-predicated value binding — replaces Mux.
                ///
                /// At each iteration coord, exactly one branch's `select` predicate should be True; its `value` is bound to
                /// `name` in the SSA scope. Branches are expected to be disjoint; later branches act as catch-alls when no
                /// earlier predicate matches.
                shared Select {PURE | CHECKED} {
                    name: Name [DEFINES],
                    branches: Branches [RECORD],
                }
                /// Zero another launch's atomic accumulator from THIS kernel — the delegated zero-init
                /// (`lowering/cuda/005_delegate_zero_init`).
                ///
                /// An atomic accumulator (atomic `Write` / `RegStore` output) must be zero before its kernel launches; the
                /// runtime's per-launch `zero_outputs` memset costs a CUDA-graph MEMSET node per site. This stmt rides a kernel
                /// that launches strictly BEFORE the accumulator's kernel in the same stream (a dataflow predecessor —
                /// topological launch order guarantees it), so stream serialization makes the zero happen-before the atomics and
                /// the MEMSET node disappears. Every thread of the grid writes a stride of it, ahead of the kernel's own work:
                /// one CTA alone left the grid waiting on a serial tail (130 us for a 4 MB accumulator on the H100). The zero is
                /// over raw 32-bit words (`words = nbytes / 4` — all-zero bytes are 0.0 in every buffer dtype, the same argument
                /// the runtime memset makes), so the render never needs the buffer's dtype.
                ///
                /// `dst` names the target buffer — it joins the carrying kernel's `outputs` (and thus its signature /
                /// `arg_names`) WITHOUT a graph edge: an edge would be a cycle (the target is the downstream kernel's own
                /// output). The slab planner learns the earlier first-write through `CudaOp.zero_prologues` instead.
                shared ZeroPrologue {SIDE_EFFECTS} {
                    dst: Name [BUFFER_WRITE],
                    words: i64 [],
                }
                /// Explicit iteration block — one loop over an axis.
                ///
                /// `body` executes `axis.extent` times, once per axis value. Reduce-loops fold any `Accum` statements in
                /// their body into the named accumulator (one sweep over the axis per accumulator). Free loops run in
                /// parallel with no folding.
                ///
                /// SSA scoping: `Assign` / `Select` names defined inside `body` are scoped to that body — invisible to statements
                /// outside the Loop. Only `Accum` targets cross the Loop boundary, carrying the finalized reduced value.
                ///
                /// Used by Loop IR for general iteration; reused by Kernel IR for serial (post-materialization) loops inside
                /// cooperative blocks.
                ///
                /// `unroll=True` annotates the loop for `#pragma unroll` at render time. Set by scheduling passes
                /// (`mark_unroll`); has no effect on the IR's iteration semantics.
                ///
                /// The loop carries NO algebra — the fold's ⊕ lives on the :class:`~emmy.compiler.ir.pure.fold.Fold` node's
                /// stored `combine`, and the dissolved fold `Accum`\ s in the body are the loop-level spelling; a loop folds iff
                /// its body carries one.
                shared Loop {} {
                    axis: AxisRef [AXIS],
                    body: Body [BODY],
                    unroll: bool [] = false,
                    seed: bool [] = true,
                }
                /// Strided iteration: `for (axis = start; axis < axis.extent; axis += step)`.
                ///
                /// Cooperative variant of `Loop` — used at Tile IR to express "threads of the CUDA block stride through this
                /// axis" (typical `start = Var('t'), step = BLOCK_SIZE`). The body uses the original axis Var directly; the
                /// strided iteration shape is encoded by the loop construct itself rather than via affine indexing in the body.
                ///
                /// `end` overrides the `<` bound with an arbitrary `Expression` over in-scope names (default: the axis extent) —
                /// a data-independent early stop such as the causal flash stream's triangular kv bound. The axis extent stays the
                /// analysis-visible range; `end` must never exceed it.
                ///
                /// Reduction detection mirrors `Loop`: a `StridedLoop` is a reduce-loop iff its body contains an `Accum`.
                shared StridedLoop {} {
                    axis: AxisRef [AXIS],
                    start: Expression [EXPRESSION] : Index,
                    step: Expression [EXPRESSION] : Index,
                    body: Body [BODY],
                    unroll: bool [] = false,
                    end: Option<Expression> [EXPRESSION | READS] : Index = None,
                    seed: bool [] = true,
                }
                /// Conditional block — `if (cond) { body } [else { else_body }]`.
                ///
                /// `cond` is an `Expression` over axis Vars and previously-defined SSA names; `body` and `else_body` are stmt
                /// sequences executed when the predicate evaluates true / false respectively. `else_body` empty means a bare
                /// `if`.
                ///
                /// SSA scoping mirrors `Loop`: names defined inside either body are scoped to that body, except `Accum` targets
                /// which cross the boundary with their finalized value (matching Loop semantics).
                ///
                /// `deps` are the SSA names referenced inside `cond` — the splicer / dataflow analyses need them to thread the
                /// predicate's reads through. Names referenced inside `body` / `else_body` are the body stmts' own deps; the
                /// recursive walker picks them up.
                shared Cond {} {
                    cond: Expression [EXPRESSION | READS] : Value,
                    body: Body [BODY],
                    else_body: Body [BODY] = Body::empty(),
                }
                /// A fusion-only scalar call; its definition stays outside generic statement-tree walks.
                ///
                /// Fusion expands its calls before a LoopOp is built. They are not members of Tile IR lambdas, and a body that
                /// holds one has no normal form.
                shared Call {DEPS_DEEP | CHECKED} {
                    name: Name [DEFINES],
                    target: SubroutineRef [],
                    args: Exprs [EXPRESSION | READS] : Index,
                }
                /// Declare a per-block `__shared__` array.
                ///
                /// Renders to `__shared__ [__align__(N)] <dtype> <name>[<prod(extents)>];`. `extents` is the multi-dim shape used
                /// to flatten `Load` / `Write` indices against this buffer. smem_bytes for `CudaOp` is computed by walking the
                /// KernelOp body and summing `prod(extents) * sizeof(dtype)` across distinct `Smem` declarations.
                ///
                /// `align` is an optional explicit byte alignment. TMA destinations (`cp.async.bulk.tensor`) require 16-byte
                /// aligned smem; the materializer sets `align=16` on TMA-target slabs. Default 0 means "no explicit alignment" —
                /// falls back to the dtype's natural alignment.
                kernel Smem {} {
                    name: Name [LOCAL_BUFFER],
                    extents: Ints [],
                    dtype: Name [] = Name::from("float"),
                    align: i64 [] = 0,
                }
                /// Thread-group barrier.
                ///
                /// `barrier_id == 0` (default): `__syncthreads();` — CTA-wide.
                ///
                /// `warp=True`: `__syncwarp();` — warp-scope only. For ordering a warp's own smem writes before its own reads of
                /// a WARP-PRIVATE buffer (the flash C→A handoff slab): post-Volta independent thread scheduling means even intra-
                /// warp write→read needs the sync, but a CTA-wide barrier there convoys the block's warps for no ordering
                /// benefit.
                ///
                /// `barrier_id > 0` and `count` set: `bar.sync <id>, <count>;` — named-barrier synchronizing exactly `count`
                /// threads on barrier id `<id>` (one of 1..15). Used inside warp-specialized branches where only a subset of CTA
                /// threads execute the sync — `__syncthreads()` would be CUDA UB because the enclosing `if (warp < P)` condition
                /// diverges across warps. The count must equal the number of threads that actually arrive at this barrier in
                /// flight.
                kernel Sync {} {
                    barrier_id: i64 [] = 0,
                    count: Option<i64> [] = None,
                    warp: bool [] = false,
                }
                /// Tile coordinate binding — one GPU thread per output element.
                ///
                /// The kernel-IR realization of a :class:`~emmy.compiler.ir.tile.TileOp`'s thread schedule (geometry only — the
                /// combine, if any, lives in the wrapped `body`): it maps the kernel's iteration space (the product of its `axes`
                /// extents — the tile's `grid_axes`) onto a 1-D linear thread grid. Each thread derives a global linear id `_gid
                /// = blockIdx.x * blockDim.x + threadIdx.x`, guards `_gid < N` (`N` = ∏ extents), decodes its per-axis indices
                /// from `_gid`, and runs the `body` once. Index motion (broadcast, transpose, gather) is already encoded in the
                /// body's `Load` index Expressions, so the decode only has to bind the axis induction variables.
                ///
                /// Renders to::
                ///
                /// int _gid = blockIdx.x * blockDim.x + threadIdx.x; if (_gid < N) { int a0 = _gid / s0;          // outermost
                /// (s0 = ∏ inner extents) int a1 = _gid / s1 % e1; ... <body> }
                ///
                /// The cuda lowering reads `N` to size the launch grid (`ceil(N / blockDim)` CTAs of `blockDim` threads). Static
                /// extents only for now — a symbolic axis raises (the runtime-arg / masked-tile forms land as the skeleton
                /// grows).
                ///
                /// `block_threads` is the per-CTA thread count for a **cooperative** tile — the materializer sets it to `coop ·
                /// ∏block-cells` so the cuda lowering derives `blockDim = block_threads` and `gridDim = N / block_threads` (the
                /// linear `_gid` decode then groups `block_threads` consecutive cells per CTA, the innermost being the
                /// cooperative lanes). `None` is the scalar tier (one thread per cell, `blockDim = _BLOCK_SIZE`).
                ///
                /// `aux_threads` is the warp-specialized producer band appended PAST the cooperative cells (`blockDim =
                /// block_threads + aux_threads`, the grid still `N / block_threads`). The decode becomes `_gid = blockIdx.x ·
                /// block_threads + threadIdx.x % block_threads`: a compute thread decodes its own cell exactly as before, while
                /// an aux thread wraps onto the CTA's first cells — its BLOCK-axis vars (functions of `blockIdx` alone) are
                /// correct (the producer fill needs the CTA tile origin), its unit/lane vars alias a compute cell and are never
                /// read (the producer branch guards on the raw `threadIdx.x`, and the store side is guarded to compute threads).
                kernel Tile {CHECKED} {
                    axes: Axes [AXIS],
                    body: Body [BODY],
                    block_threads: Option<i64> [] = None,
                    aux_threads: i64 [] = 0,
                    raster_axes: Option<(Name, Name)> [AXIS_NAME] = None,
                    raster_group: Option<i64> [] = None,
                    raster_orient: Name [] = Name::from("m"),
                    spread: bool [] = false,
                }
                /// Issue one `cp.async.{ca,cg}.shared.global` instruction.
                ///
                /// Replaces the per-thread `Load(reg) + Write(smem)` pair in cooperative loads on sm_80+. The hardware copies
                /// `nbytes` (one contiguous vector of `nbytes / sizeof(dtype)` elements) directly from global to shared without a
                /// register staging slot, freeing the thread register and removing the LDG → STS dependency.
                ///
                /// `nbytes` ∈ {4, 8, 16} (the cp.async copy sizes). 16-byte copies use the `.cg` (cache-global, bypass-L1)
                /// qualifier — the streaming form CUTLASS / cuBLAS use, requiring 16-byte-aligned smem + global addresses — and
                /// 4/8-byte copies use `.ca`. `smem_index` / `src_index` address the *start* of the contiguous chunk;
                /// `_stage_expand` picks the widest legal `nbytes` so e.g. an fp16 tile stages 8 halves per `cp.async`.
                ///
                /// Renders to inline PTX. The asm reads the smem address via `cvta.to.shared.u32` and the global pointer as a
                /// 64-bit value; indices flatten via `render_index` against the buffer's declared shape (same as `Load` /
                /// `Write`).
                kernel CpAsyncCopy {} {
                    smem: Name [BUFFER_WRITE],
                    smem_index: Exprs [EXPRESSION | READS] : Index,
                    src: Name [BUFFER_READ],
                    src_index: Exprs [EXPRESSION | READS] : Index,
                    nbytes: i64 [] = 4,
                    swizzle: Name [] = Name::from("NONE"),
                    lane_index: Option<Exprs> [EXPRESSION | READS] : Index = None,
                    lane_rows: i64 [] = 0,
                    valid: Option<Expression> [EXPRESSION | READS] : Index = None,
                }
                /// `cp.async.commit_group;` — finalize the preceding cp.async copies issued by this thread into a commit group.
                /// Pairs with `CpAsyncWait` to wait for that group to drain.
                kernel CpAsyncCommit {} {}
                /// `cp.async.wait_group N;` — block this thread until ≤ N cp.async groups remain in flight. `group=0` waits for
                /// everything (synchronous style); larger values stagger waits for software pipelining.
                kernel CpAsyncWait {} {
                    group: i64 [] = 0,
                }
                /// Host-built `CUtensorMap` descriptor for a TMA box copy.
                ///
                /// Declarative — renders nothing inside the kernel. The CUDA backend walks the `KernelOp` body, calls
                /// `cuTensorMapEncodeTiled` once per distinct `TmaDescriptor` at compile time, and binds the resulting 128-byte
                /// descriptor as a `__grid_constant__` kernel arg. `name` matches the parameter name referenced by sibling
                /// `TmaLoad` stmts. `src_buf` names the global buffer the descriptor addresses (used by the kernel-arg pipeline
                /// to add `src_buf` to the parameter list).
                kernel TmaDescriptor {} {
                    name: Name [],
                    src_buf: Name [BUFFER_READ],
                    src_shape: Ints [],
                    box_extents: Ints [],
                    swizzle: Name [] = Name::from("NONE"),
                    dtype: Name [] = Name::from("float"),
                }
                /// `cp.async.bulk.tensor.<rank>d` — single-thread box copy.
                ///
                /// Issued by exactly one thread of the CTA (the materializer wraps this in `Cond(tid==0, [...])`). Source coords
                /// come from the staging Statement's `origin` (CTA-uniform); destination is the dynamic-phase smem slab. Pairs
                /// with `MbarrierArriveExpectTx` (issued immediately before) and `MbarrierWait` (issued by every consumer
                /// thread).
                kernel TmaLoad {} {
                    smem: Name [BUFFER_WRITE],
                    smem_index: Exprs [EXPRESSION | READS] : Index,
                    desc: Name [],
                    coords: Exprs [EXPRESSION | READS] : Index,
                    mbar: Name [],
                    mbar_slot: Option<Expression> [EXPRESSION | READS] : Index = None,
                }
                /// `mbarrier.init.shared.b64 [&mbar[slot]], count;` — one-shot init.
                ///
                /// With `slot=None`: initializes a scalar mbarrier (single-slot path). With `slot=Expression`: initializes one
                /// element of a per-slot mbarrier array. The materializer emits one `MbarrierInit` per literal slot index in the
                /// buffer count, hoisted to the kernel prologue inside `Cond(threadIdx.x == 0, [...]) + Sync()`.
                kernel MbarrierInit {} {
                    mbar: Name [],
                    count: i64 [] = 1,
                    slot: Option<Expression> [EXPRESSION] : Index = None,
                }
                /// `mbarrier.arrive.expect_tx.shared.b64` — declare the expected transaction byte count for the in-flight TMA
                /// copy on this barrier. With per-slot mbarrier arrays, `slot` selects the ring buffer slot.
                kernel MbarrierArriveExpectTx {} {
                    mbar: Name [],
                    bytes_: i64 [],
                    slot: Option<Expression> [EXPRESSION] : Index = None,
                }
                /// `mbarrier.arrive.shared.b64` — simple arrive (no transaction-byte count). Used by warp-specialized consumer
                /// warps to signal "slot empty" so the producer can refill it on the next ring iteration. The producer side uses
                /// `MbarrierArriveExpectTx` instead (which declares the incoming TMA byte count).
                kernel MbarrierArrive {} {
                    mbar: Name [],
                    slot: Option<Expression> [EXPRESSION] : Index = None,
                }
                /// `setmaxnreg.{inc,dec}.sync.aligned.u32 N;` — Hopper+ register-budget redistribution. `direction="dec"` shrinks
                /// the calling warp's max register count to `count` (returning registers to the pool); `direction="inc"` claims
                /// up to `count` registers. Used by warp- specialized kernels so producer warps drop registers and consumer warps
                /// claim them, decoupling occupancy from the consumer's pressure.
                ///
                /// Requires sm_90+. On older targets NVCC rejects the instruction at compile time — only a warp-specialized
                /// materialization emits this Statement, and that path is already TMA-gated to sm_90+ anyway.
                kernel SetMaxNReg {} {
                    count: i64 [],
                    direction: Name [],
                }
                /// `mbarrier.try_wait.parity.shared.b64` — block until the barrier's parity flips for `phase`. Replaces
                /// `CpAsyncWait + Sync` for TMA transports — mbarrier arrival already provides CTA-wide visibility. `slot`
                /// selects the ring buffer slot when the mbarrier is an array.
                kernel MbarrierWait {} {
                    mbar: Name [],
                    phase: Expression [EXPRESSION] : Index,
                    slot: Option<Expression> [EXPRESSION] : Index = None,
                }
                /// Cross-thread combine of a (possibly multi-component) monoid state over a PRE-POPULATED power-of-two smem tree
                /// — the > warp_size fallback for :class:`WarpShuffle`.
                ///
                /// Each state component has its own smem slab `bufs[i][0..length)`, populated by the caller (`emit_combine`
                /// writes the per-thread partials, or the hierarchical path writes one per-warp partial per slot). This stmt
                /// tree-reduces each slab into `bufs[i][0]` applying the carrier's `combine_states` at every halving step
                /// (rendered at `dtype`), then broadcast-reassigns every `state[i]` from `bufs[i][0]` **in place** (no `_b`
                /// rename — every thread ends holding the full reduction in the carried SSA names). `length` is a power of two ≤
                /// `blockDim.x`. A scalar reduce is the 1-component case (`state=("acc",)`, `combine_states=(Assign("acc", op,
                /// ("acc","acc__o")),)`).
                ///
                /// Named-barrier support: `barrier_id > 0` routes the per-iter sync to `bar.sync <id>, <count>` instead of
                /// `__syncthreads()` — required inside a warp-specialized consumer branch (a __syncthreads on a warp-divergent
                /// cond is UB).
                kernel TreeHalve {CHECKED} {
                    bufs: Names [],
                    state: Names [UPDATES],
                    state_b: Names [LOCAL],
                    combine_states: Stmts [STMTS],
                    length: i64 [],
                    tid_var: Expression [EXPRESSION] : Index,
                    dtype: DType [] = DType::F32,
                    barrier_id: i64 [] = 0,
                    barrier_count: Option<i64> [] = None,
                    inner: Option<(Expression, i64)> [EXPRESSION] : Index = None,
                }
                /// Cross-thread combine of a (possibly multi-component) monoid state over `length` lanes via `__shfl_xor_sync` —
                /// a register-only butterfly (no smem, no syncthreads).
                ///
                /// Each lane holds a full per-thread partial `state` tuple (a scalar reduce is the 1-component case
                /// `state=("acc",)`). Each butterfly step shuffles **every** component down into `state_b` and applies the
                /// carrier's `combine_states` (the state-merges-state monoid op, rendered at `dtype`) to fold the shuffled
                /// neighbor into this lane's state, reassigning `state` **in place**. After the last step every lane holds the
                /// full reduction in the SAME SSA names — so the post-reduce epilogue reads them with no rename (kills the old
                /// scalar `<name>_b` broadcast).
                ///
                /// `length` must be a power of two ≤ `warp_size`; the XOR butterfly never crosses an aligned `length`-lane group,
                /// so this is also the SEGMENTED per-row combine for strided-cooperative rows. `commutative` (required — the
                /// butterfly reorders) is checked at the carrier.
                ///
                /// The shuffle mask is `__activemask()`, NOT a hard-coded `0xffffffff`: a whole-CTA cooperative reduce with `BR <
                /// warp_size` launches a **partial warp** (e.g. `block=8`), and naming the absent lanes `8..31` in the mask is
                /// undefined behavior — on the e2e captured-graph path it manifested as a multi-millisecond stall (a per-
                /// token×head RMSNorm went 28 ms instead of µs). The combine is reached with the warp converged (the per-CTA
                /// reduce has no data-dependent divergence and the only enclosing guard is warp-uniform), so `__activemask()`
                /// returns exactly the participating lanes — correct for a partial warp, a full warp, AND a non-warp-multiple
                /// block's partial final warp (where no single static literal mask is correct per warp).
                kernel WarpShuffle {CHECKED} {
                    state: Names [UPDATES],
                    state_b: Names [LOCAL],
                    combine_states: Stmts [STMTS],
                    length: i64 [],
                    dtype: DType [] = DType::F32,
                    prefix: bool [] = false,
                }
                /// Broadcast the last lane's state within each aligned lane group, in place.
                kernel WarpBroadcast {} {
                    state: Names [UPDATES],
                    length: i64 [],
                }
                /// The one per-element pointwise node over an `mma.sync` C fragment — the fragment-tier sibling of the scalar
                /// `Assign`.
                ///
                /// Writes `out` (a layout-sized f32 C fragment) `= op(args…)` per element `i` via the same `op_to_expression`
                /// translation the scalar `Assign` uses — so ANY elementwise op reaches the tensor-core tier, not just softmax's
                /// `exp` / scale. Each arg (:class:`FragmentArg`) is one of five forms, named by its `kind`:
                ///
                /// - `FRAG` — a C-fragment, indexed `arg[i]`; - `ROW` — a per-row scalar pair, broadcast by the layout's element-
                /// to-row map; - `UNIFORM` — a cell-uniform scalar name or `Literal`, the same value for every element; - `COORD`
                /// — a predicate `Expression` over the element's ABSOLUTE coordinates, written with the reserved :data:`FRAG_ROW`
                /// / :data:`FRAG_COL` vars (causal `__fcol > __frow`, a symbolic boundary `__fcol >= seq_len`); the render
                /// substitutes tile origin + layout offset; - `GMEM` — a `(buffer, index)` load template over the same reserved
                /// vars, read at each element's absolute coordinates and converted to f32 (SDPA's additive `attn_mask` bias; the
                /// gathered score tile of the softmax@V carrier).
                ///
                /// `row_base` / `col_base` are the tile origin the `COORD` / `GMEM` args add to the layout's per-element
                /// offsets; `row_base` is required iff a template references `__frow`.
                ///
                /// Realizations: `exp(s − m)` = a `subtract` (FRAG, ROW) then an `exp` (FRAG); `O *= α` = an in-place `multiply`
                /// (FRAG, ROW); `S *= scale` = an in-place `multiply` (FRAG, UNIFORM); a coordinate mask = an in-place `where`
                /// (COORD, UNIFORM fill, FRAG) — the masked branch takes the carrier's finite identity instead of producing `-inf
                /// - -inf` in an all-masked chunk; a gmem bias = an in-place `add` (FRAG, GMEM). `in_place` reassigns `out` (no
                /// `float out[4]` decl — `out` must be a FRAG arg).
                ///
                /// `layout` names the atom's C-fragment geometry (:func:`frag_layout`).
                kernel FragmentApply {} {
                    out: Name [DEFINES],
                    op: ElementwiseOp [],
                    args: FragmentArgs [RECORD],
                    in_place: bool [] = false,
                    layout: Name [] = Name::from("m16n8k16"),
                    post: ElementwiseOps [] = ElementwiseOps::from([]),
                    row_base: Option<Expression> [EXPRESSION] : Index = None,
                    col_base: Option<Expression> [EXPRESSION] : Index = None,
                }
                /// Per-ROW reduction over one warp's `mma.sync` C fragments — the chunk pivot and the per-row channel partials
                /// the twisted carrier's atom tier folds (`_atom._FlashOps`).
                ///
                /// A chunk-wide score tile is `len(frags)` C fragments side by side. Reducing over the chunk combines every
                /// element assigned to each of the lane's two rows, then applies the layout's `__shfl_xor` butterfly masks across
                /// the lanes that hold columns of the same row. Afterwards every lane in the group holds the full per-row value,
                /// so `top` and `bot` are the :data:`ROW` operands :class:`FragmentApply` broadcasts.
                ///
                /// Distinct from :class:`WarpShuffle`, which folds a whole per-thread monoid state over a cooperative-K lane set;
                /// this folds WITHIN one warp's fragments along the atom's N direction, keyed on the PTX C layout. It is why the
                /// tier requires the chunk to sit inside one warp column (`n_units == 1` on the score): a reduction that crossed
                /// warps would need smem.
                kernel FragmentRowReduce {} {
                    top: Name [DEFINES],
                    bot: Name [DEFINES],
                    frags: Names [READS],
                    op: ElementwiseOp [],
                    layout: Name [] = Name::from("m16n8k16"),
                    dtype: DType [] = DType::F32,
                }
                /// Declare an mma.sync per-thread register array.
                ///
                /// mma.sync multiplicands are explicit per-lane register arrays — `unsigned a[4]` / `unsigned b[2]` (f16, two
                /// halfs per 32-bit reg) — and the accumulator is `float c[4]` (f32) or, on the f16-accumulate atom, packed
                /// `unsigned c[2]` (two halfs per reg — the same element map, pair-packed, on m16n8k16). Volta uses explicit
                /// counts and distinct f16/f32 accumulator lane maps. `shape` is the cell `(M, N, K)`; the count derives from
                /// `shape` + `role` (+ the C dtype) via :func:`_mma_sync_nregs`. The `c` array is zero-initialised at
                /// declaration, so the mma.sync path needs no separate fill node.
                ///
                /// `count` (default 1) arrays a whole fragment FAMILY into one declaration — `float O_i_f[count][n_regs]` — so a
                /// run of `count` sibling fragments (`O_i_f0 … O_i_f{count-1}`) collapses to a single arrayed decl indexed
                /// `O_i_f[t]`. Only `100_loopify` sets `count > 1` (the pin-gated re-roll); `count == 1` renders the flat `float
                /// name[n_regs]` form, byte-identical. A family of SCALAR registers (`nregs == 0` — the fp4 block-scale
                /// fragments) arrays to one dimension, `unsigned name[count]`, because its member is the register itself and not
                /// an array of them.
                kernel RegFragment {} {
                    name: Name [DEFINES | LOCAL_BUFFER],
                    role: Name [],
                    shape: Shape3 [],
                    dtype: DType [],
                    count: i64 [] = 1,
                    nregs: Option<i64> [] = None,
                }
                /// Load one lane's b32 of ue4m3 block scales for the block-scaled fp4 mma.
                ///
                /// The instruction takes its per-16-element scales as two ordinary register operands beside the multiplicands,
                /// one per side, and applies them itself. A supplying lane holds four K blocks as the four bytes of one b32 —
                /// exactly the `k = 64` an atom step spans — and the lane→row map differs per side and is NOT the data
                /// fragment's: SFA's supplying lanes are picked by `(lane & 3)` against a thread-id selector, SFB's by equality
                /// with one. Both maps were found by probing the hardware, and the render's prelude records them beside the PTX
                /// (the PTX ISA's documented thread-id range is wrong for `scale_vec::4X` — ptxas takes 0..1, not 0..3).
                ///
                /// `src_buffer` is the scale slab, `src_index` the fragment's `(row, k-block)` base and `ldm` the slab's row
                /// stride in bytes. Non-supplying lanes read nothing and the instruction ignores their register, so there is no
                /// guard here.
                ///
                /// Distinct from :class:`LdmatrixLoad`'s `scale_buffer`, which is the OTHER way to spend a block scale: there the
                /// k16 drain folds it into a 16-bit fragment and no scale reaches the instruction. These two never appear on one
                /// operand.
                kernel BlockScaleLoad {} {
                    frag: Name [UPDATES],
                    src_buffer: Name [BUFFER_READ],
                    src_index: Exprs [EXPRESSION] : Index,
                    role: Name [],
                    ldm: i64 [] = 0,
                }
                /// `ldmatrix.sync.aligned.m8n8.x{2,4}[.trans].b16` — load one operand fragment from smem into a per-thread
                /// register array.
                ///
                /// `frag` is the destination :class:`RegFragment`; `src_buffer` / `src_index` are the cell's smem tile base (the
                /// same `(buffer, base-offset)` `_mma_src_index` computes); `ldm` is the smem row stride in elements. Each lane
                /// derives its own row address from its warp lane id (`threadIdx.x & 31`): the 16×K `a` tile uses `x4` (`row =
                /// lane%16`, K-col block `(lane/16)*8`); the K×8 `b` tile uses `x2.trans` (`row = lane%16`) so a row-major smem
                /// slab feeds the mma's col-major B operand.
                ///
                /// `staged` (default `True`) selects the transport. The modern fragment layout's `ldmatrix` is **smem only**,
                /// while the Volta layout uses its cooperative gather against either address space. When the operand was NOT
                /// staged into shared memory (`staged=False`, `src_buffer` is the gmem operand) the render emits the
                /// `emmy_mma_load_{a,b}_gmem` helper instead — a gmem-direct fragment load that replicates the same lane→element
                /// map without ldmatrix. Slower (no smem reuse) but correct; the atom lowering picks per operand based on whether
                /// an enclosing `StageBundle` staged it.
                ///
                /// `gmem_guard` (gmem-direct only) carries a masked-tile boundary as `(base Expression, bound Expression)` on the
                /// operand's lane-varying output axis (A's M rows, B's N cols): the render switches to the `_mclamp` / `_nclamp`
                /// helper, which clamps the lane coordinate to the `bound - base` in-range elements — the gmem-direct analogue of
                /// the staged slab-fill clamp, so an unstaged masked cell still lowers instead of reading past the runtime-sized
                /// buffer. Clamped lanes read duplicates; their stores are masked by the RegStore guards.
                kernel LdmatrixLoad {} {
                    frag: Name [UPDATES],
                    src_buffer: Name [BUFFER_READ],
                    src_index: Exprs [EXPRESSION] : Index,
                    role: Name [],
                    ldm: i64 [] = 0,
                    swizzle: Name [] = Name::from("NONE"),
                    staged: bool [] = true,
                    gmem_guard: Option<(Expression, Expression)> [EXPRESSION] : Index = None,
                    k_zero: Option<(Expression, Expression)> [EXPRESSION] : Index = None,
                    b_trans: bool [] = false,
                    pair_frag: Option<Name> [UPDATES] = None,
                    byte_slab: bool [] = false,
                    scale_buffer: Option<Name> [BUFFER_READ] = None,
                    scale_index: Exprs [EXPRESSION] : Index = Exprs::from([]),
                    scale_ldm: i64 [] = 0,
                    fragment_layout: Name [] = Name::from("m16n8k16"),
                    fragment_index: i64 [] = 0,
                }
                /// `mma.sync.aligned.m{M}n{N}k{K}.row.col.f32.{ab}.{ab}.f32` — one tensor-core MMA via inline PTX (the `s16816`
                /// instruction).
                ///
                /// `c_frag` is the accumulator array (both input and output — `d = a·b + c`); `a_frag` / `b_frag` are the
                /// multiplicand arrays filled by :class:`LdmatrixLoad`. `shape` spells the M/N/K. `ab_dtype` (`"f16"` / `"bf16"`
                /// / `"e4m3"` / `"e5m2"` — the PTX dtype field, :attr:`AtomKind.ab_dtype`) tags the operand element type — same-
                /// width dtypes share a fragment layout and differ only in the PTX instruction's dtype field, so the render just
                /// picks the matching `emmy_mma_…` wrapper. `c_dtype` is the accumulator element type: `"f32"` (the default —
                /// `float c[4]`) or `"f16"` (the f16-accumulate atom — packed `unsigned c[2]`, full HMMA rate on the consumer
                /// dies; the lowering pairs it with a periodic :class:`FragmentPromote` into an f32 shadow fragment).
                kernel MmaSyncPtx {} {
                    c_frag: Name [UPDATES],
                    a_frag: Name [READS],
                    b_frag: Name [READS],
                    shape: Shape3 [],
                    ab_dtype: Name [] = Name::from("f16"),
                    c_dtype: Name [] = Name::from("f32"),
                    b_row_major: bool [] = false,
                    sfa_frag: Option<Name> [READS] = None,
                    sfb_frag: Option<Name> [READS] = None,
                }
                /// Fold an f16-accumulate mma **C fragment** into its f32 **shadow fragment** and rezero it — the chunked-
                /// accumulation promote of the `_f16acc` atom family: the mma chain accumulates at the full f16-accumulate HMMA
                /// rate, and every K chunk this promote-adds the packed f16 partials into the f32 shadow (`cvt.f32.f16` + add per
                /// element) and re-zeros the f16 fragment, so the accumulation error stays bounded by one chunk's length. `dst`
                /// is the f32 shadow the store / epilogue reads; `src` is the packed f16 mma accumulator. The fragment layout
                /// selects their register counts and, on Volta, the warp shuffles that align their different lane maps. Both are
                /// defined here (`src` is rezeroed), so reorderings keep the promote pinned between the mma chain and the store.
                kernel FragmentPromote {} {
                    dst: Name [UPDATES],
                    src: Name [UPDATES],
                    fragment_layout: Name [] = Name::from("m16n8k16"),
                }
                /// Convert mma **C fragments** into one 16-bit operand fragment in registers.
                ///
                /// The m16n8k16 layout takes two k-adjacent C fragments whose lanes already align with A. The Volta m8n8k4 layout
                /// takes one logical 16×16 C fragment and selects a four-column A slice or four-row B slice with warp shuffles.
                /// The emitter gates on the atom's C→A/C→B repack capabilities at schedule time. The m16n8k16 f16 B layout takes
                /// one C fragment and exchanges its packed column pairs between lanes. Every lane of the warp must participate in
                /// that exchange.
                kernel FragmentRepack {} {
                    frag: Name [DEFINES],
                    srcs: Names [READS],
                    ab_dtype: Name [] = Name::from("f16"),
                    fragment_layout: Name [] = Name::from("m16n8k16"),
                    part: i64 [] = 0,
                    role: Name [] = Name::from("a"),
                }
                /// `unsigned long long name = emmy_wgmma_desc(...)` — the shared-memory matrix descriptor a `wgmma.mma_async`
                /// operand is read through (:func:`wgmma_descriptor_bits`). Replaces :class:`LdmatrixLoad` on a descriptor-fed
                /// operand: the tensor core reads the slab itself, so no per-lane registers are filled. `smem_index` (elements
                /// into `smem`) selects the ring slot and the k step; `lbo_bytes` / `sbo_bytes` are the leading / stride byte
                /// offsets between core matrices, which the lowering derives from the slab's layout.
                kernel WgmmaDescriptor {} {
                    name: Name [DEFINES],
                    smem: Name [BUFFER_READ],
                    smem_index: Option<Expression> [EXPRESSION] : Index,
                    swizzle: Name [],
                    lbo_bytes: i64 [],
                    sbo_bytes: i64 [],
                }
                /// `wgmma.mma_async.sync.aligned.m64n{N}k16.f32.{ab}.{ab}` — one warp-group tensor-core MMA, issued
                /// asynchronously by the four warps of a group together.
                ///
                /// `c_frags` is the accumulator: per warp, the `m64nN` instruction's `N / 2` f32 registers are the `N / 8`
                /// ordinary `m16n8` C fragments (:class:`RegFragment`, `float c[4]`) side by side — instruction register `d[4j +
                /// k]` is fragment `j`, register `k` — so the epilogue, the softmax rescale and the P→A repack keep their lane
                /// maps. Exactly one of `a_desc` (the shared-memory form, `_ss`) and `a_frag` (the register form, `_rs` — the
                /// `m16n8k16` A fragment, K-major only) names the A operand; `b_desc` is always a :class:`WgmmaDescriptor`.
                /// `scale_d = 0` discards the old accumulator (`d = a·b`, the first step of a fresh accumulator); `trans_a` /
                /// `trans_b` are the descriptor operands' MN-major transpose bits.
                ///
                /// The discipline around the cells (:class:`WgmmaFence`, :class:`WgmmaCommit`, :class:`WgmmaWait`) is the
                /// lowering's: nothing here orders the accumulator read.
                kernel WgmmaMma {CHECKED} {
                    c_frags: Names [UPDATES],
                    b_desc: Name [READS],
                    shape: Shape3 [],
                    ab_dtype: Name [] = Name::from("f16"),
                    a_desc: Option<Name> [READS] = None,
                    a_frag: Option<Name> [READS] = None,
                    scale_d: i64 [] = 1,
                    trans_a: i64 [] = 0,
                    trans_b: i64 [] = 0,
                }
                /// `wgmma.fence.sync.aligned` — orders the group's earlier register and shared-memory accesses before the
                /// `wgmma.mma_async` that follows. The lowering issues it before the first cell after anything else touched the
                /// accumulator (the epilogue, the softmax rescale).
                kernel WgmmaFence {} {}
                /// `wgmma.commit_group.sync.aligned` — closes the `wgmma.mma_async` cells issued since the last commit into one
                /// group for :class:`WgmmaWait` to count. The lowering commits after a chunk's cells.
                kernel WgmmaCommit {} {}
                /// `wgmma.wait_group.sync.aligned N` — block until at most `group` committed groups are still in flight. Carries
                /// no deps: the accumulator it completes is the `c_frags` of the cells before it, and the lowering places the
                /// wait before any read of those fragments and before the slot's release on the "empty" mbarrier. The redundant-
                /// sync and ldmatrix-pairing passes treat it as a barrier.
                kernel WgmmaWait {} {
                    group: i64 [] = 0,
                }
                /// Store an mma.sync f32 accumulator array to the output buffer with a per-lane epilogue downconvert.
                ///
                /// mma.sync has no `store_matrix_sync` — each lane owns 4 elements of the M×N=16×8 C tile (rows `g` / `g+8` with
                /// `g = lane/4`, cols `(lane%4)*2 + {0,1}`) and writes them directly. `frag` is the f32 `float c[4]`; when the
                /// destination buffer is narrower (`__half*`) each value is converted via `__float2half`. `ldm` is the output row
                /// stride (N) — `0` auto-resolves from the buffer's inner extent.
                ///
                /// `epilogue` optionally carries a fused pointwise chain — residual adds, bias / scale broadcasts, activations, a
                /// coordinate `Select` — as a pure term over the ordinary `Load` / `Assign` / `Select` stmts of the projection
                /// tail, stored as its body: `epilogue_params` bind, in order, the accumulator names to `frag` and `extra_frags`
                /// (a multi-fold node's additional C fragments, so the chain combines the channels per cell — SwiGLU et al.),
                /// every other name the body reads is a coordinate, and `epilogue_result` names the stored value. The lowering
                /// forms the body through `Lambda.closing`, which fixes its statement order. Evaluated per fragment element right
                /// before the downconvert (the CUTLASS epilogue-visitor pattern): floating loads widen to f32, while integer
                /// loads retain their type for arithmetic on packed codes. Each param substitutes to that element, and the
                /// reserved :data:`ELEM_ROW` / :data:`ELEM_COL` vars in a `Load` index or `Select` predicate substitute to the
                /// element's row / col offset within the fragment (the cell base is already in the expression).
                ///
                /// `m_guard` / `n_guard` carry a masked-tile boundary as `(base Expression, bound Expression)` — the tile's row /
                /// col coordinate of fragment element (0,0) and the axis's real extent (a `Literal` for a static overhang axis,
                /// the symbolic `Var` for a runtime-sized one). The enclosing boundary `Cond` only gates on the tile base (the
                /// atom-lane offsets `_g` / `_t` are render-local, invisible to σ), so an atom tile *straddling* the bound passes
                /// the Cond while its trailing rows / cols are out of range. A guard predicates each fragment element's store
                /// (and its epilogue gmem reads) at its own coordinate: `base + _g (+8) < bound` per row, `base + _t*2 (+1) <
                /// bound` per col. `None` (the default) renders the unguarded fast path unchanged. `m_guard` alone keeps the
                /// vectorized row-pair stores (a pair shares a row); an `n_guard` forces per-element scalar stores (the pair
                /// straddles the column bound).
                ///
                /// `row_dim` / `col_dim` identify the output-index dimensions carrying the mma fragment's algebraic M/N axes.
                /// Their row-major strides may be either physical orientation: a contraction whose shared operand varies along
                /// the output's trailing axis stores a transposed fragment with scalar strided writes, while the ordinary
                /// contiguous-N case keeps packed pairs.
                ///
                /// A staged Volta tile using paired operand layouts also derives `volta_interleaved` and its register-cell
                /// `fragment_index`. Together they select the coupled 32×32 accumulator map; neither is a schedule choice or a
                /// wire-codec field.
                ///
                /// `atomic` renders each store as an `atomicAdd` accumulate instead of a plain assign — `030_cut`'s atomic
                /// finalize on the mma tier: every split partition's C fragment adds into the (per-launch zero-init'd) output.
                /// f16 / bf16 keep the vectorized row pair (native packed-pair `atomicAdd`); an f32 destination has no packed
                /// atomic below sm_90, so it drops to per-element scalar `atomicAdd`\ s.
                kernel RegStore {CHECKED} {
                    dst_buffer: Name [BUFFER_WRITE],
                    dst_index: Exprs [EXPRESSION] : Index,
                    frag: Name [READS],
                    shape: Shape3 [],
                    ldm: i64 [] = 0,
                    ldn: i64 [] = 0,
                    epilogue: Option<Body> [TERM] = None,
                    epilogue_params: Names [] = Names::from([]),
                    epilogue_result: Option<Name> [] = None,
                    extra_frags: Names [READS] = Names::from([]),
                    m_guard: Option<(Expression, Expression)> [EXPRESSION] : Index = None,
                    n_guard: Option<(Expression, Expression)> [EXPRESSION] : Index = None,
                    atomic: bool [] = false,
                    fragment_layout: Name [] = Name::from("m16n8k16"),
                    volta_interleaved: bool [] = false,
                    fragment_index: Int2 [] = Int2(0, 0),
                    swizzle: Name [] = Name::from("NONE"),
                    row_dim: Option<i64> [] = None,
                    col_dim: Option<i64> [] = None,
                    run: Stmts [STMTS] = Stmts::from([]),
                }
            }
            records {
                /// One branch of a `Select` body statement.
                shared SelectBranch {
                    value: Name [READS],
                    select: Expression [EXPRESSION | READS] : Value,
                }
                /// One operand of a :class:`FragmentApply`. What it holds follows its `kind`:
                ///
                /// - `FRAG` — `names` is the one C fragment; - `ROW` — `names` are the two per-row scalars, stored explicitly so
                /// an SSA rename keeps them consistent with their definitions; - `UNIFORM` — `names` is the one cell-uniform
                /// scalar, or `expressions` its one literal; - `COORD` — `expressions` is the one predicate; - `GMEM` — `buffer`
                /// is the buffer read and `expressions` its index template.
                kernel FragmentArg {
                    kind: Name [],
                    names: Names [READS] = Names::from([]),
                    expressions: Exprs [EXPRESSION] : Value = Exprs::from([]),
                    buffer: Option<Name> [BUFFER_READ] = None,
                }
            }
        }
    };
}

/// Where a statement type is exported from: the shared statement module or the Kernel IR.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layer {
    Shared,
    Kernel,
}

/// One field of a declared statement type: its name, its roles and whether it has a default.
#[derive(Clone, Copy, Debug)]
pub struct FieldInfo {
    pub name: &'static str,
    pub roles: Roles,
    pub has_default: bool,
    /// Given by keyword only in the Python constructor (an `Init`'s dtype): `KW_ONLY` in the declaration, no role.
    pub kw_only: bool,
}

/// The class an expression field's roles declare, when they declare one.
pub fn class_of(roles: Roles) -> Option<crate::expr::ExprClass> {
    if roles.contains_any(INDEX) {
        Some(crate::expr::ExprClass::Index)
    } else if roles.contains_any(VALUE) {
        Some(crate::expr::ExprClass::Value)
    } else {
        None
    }
}

/// Every expression `field` holds, visited under `roles`, is of the class its roles declare (a record's fields
/// under their own), or the first that is not is the error, named by `label` (`Load.index`).
pub fn classed<T: crate::field::Field>(field: &T, roles: Roles, label: &str) -> crate::Result<()> {
    struct Check(crate::Result<()>);
    impl crate::field::Visitor for Check {
        fn expression(&mut self, roles: Roles, expression: &crate::expr::Expression) {
            if self.0.is_err() || !roles.contains_any(EXPRESSION) {
                return;
            }
            if let Some(class) = class_of(roles) {
                self.0 = class.check(expression);
            }
        }
    }
    let mut check = Check(Ok(()));
    field.visit(roles, &mut check);
    check.0.map_err(|err| match err {
        crate::Error::Value(message) => crate::Error::Value(format!("{label}: {message}")),
        other => other,
    })
}

impl FieldInfo {
    /// A block field: spelled as statements under the header, not on it.
    pub fn is_block(&self) -> bool {
        self.roles.contains_any(BODY.or(TERM).or(STMTS))
    }

    /// The expression class every expression the field holds must belong to: what its roles declare.
    pub fn class(&self) -> Option<crate::expr::ExprClass> {
        class_of(self.roles)
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! __class_role {
    () => {
        $crate::statement::Roles::NONE
    };
    (Index) => {
        $crate::statement::INDEX
    };
    (Value) => {
        $crate::statement::VALUE
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __has_default {
    () => {
        false
    };
    ($default:expr) => {
        true
    };
}

/// The core's callback: the structs, the `Statement` enum and the generic walks.
macro_rules! define_statement_types {
    (
        statement_types { $( $(#[doc = $doc:literal])* $layer:ident $StatementType:ident {$($flag:tt)|*} { $( $field:ident : $ty:ty [$($role:ident)|*] $(: $class:ident)? $(= $default:expr)? ,)* } )* }
        records { $( $(#[doc = $rdoc:literal])* $rlayer:ident $Record:ident { $( $rfield:ident : $rty:ty [$($rrole:ident)|*] $(: $rclass:ident)? $(= $rdefault:expr)? ,)* } )* }
    ) => {
        $(
            $(#[doc = $doc])*
            #[derive(Clone, PartialEq, Eq, Hash, Debug)]
            pub struct $StatementType {
                $( pub $field: $ty, )*
            }

            #[allow(non_upper_case_globals, dead_code, unused_variables, unused_mut)]
            impl $StatementType {
                pub const LAYER: Layer = define_statement_types!(@layer $layer);
                pub const FLAGS: StatementFlags = StatementFlags::NONE $(.or($flag))*;
                define_statement_types!(@fields $StatementType { $( $field : $ty [$($role)|*] $(: $class)? $(= $default)? ,)* });
            }

            impl<'a> TryFrom<&'a Statement> for &'a $StatementType {
                type Error = $crate::Error;

                /// The statement as this statement type, or a type error naming the type it is.
                fn try_from(stmt: &'a Statement) -> $crate::Result<&'a $StatementType> {
                    match stmt {
                        Statement::$StatementType(inner) => Ok(inner),
                        other => Err($crate::Error::Type(format!("not a {}: a {}", stringify!($StatementType), other.ty()))),
                    }
                }
            }
        )*

        $(
            $(#[doc = $rdoc])*
            #[derive(Clone, PartialEq, Eq, Hash, Debug)]
            pub struct $Record {
                $( pub $rfield: $rty, )*
            }

            #[allow(non_upper_case_globals, dead_code, unused_variables, unused_mut)]
            impl $Record {
                pub const LAYER: Layer = define_statement_types!(@layer $rlayer);
                define_statement_types!(@fields $Record { $( $rfield : $rty [$($rrole)|*] $(: $rclass)? $(= $rdefault)? ,)* });
            }

            /// A record is a field of the statement type that holds it: spelled on the header as its name over its own
            /// fields.
            impl $crate::field::Field for $Record {
                fn ty() -> $crate::field::FieldType {
                    $crate::field::FieldType::Record($Record::NAME)
                }
                fn visit(&self, _roles: Roles, visitor: &mut dyn $crate::field::Visitor) {
                    $Record::visit(self, visitor);
                }
                fn rewrite(&self, _roles: Roles, rewriter: &mut dyn $crate::field::Rewriter) -> Self {
                    $Record::rewrite(self, rewriter)
                }
                fn same(&self, other: &Self) -> bool {
                    $Record::same(self, other)
                }
                fn spell(&self, out: &mut String) -> $crate::Result<()> {
                    out.push_str($Record::NAME);
                    let fields = self.header_fields()?;
                    if !fields.is_empty() {
                        out.push('(');
                        for (index, (name, text)) in fields.iter().enumerate() {
                            if index > 0 {
                                out.push_str(", ");
                            }
                            out.push_str(name);
                            out.push('=');
                            out.push_str(text);
                        }
                        out.push(')');
                    }
                    Ok(())
                }
                fn from_value(value: $crate::field::Value) -> $crate::Result<Self> {
                    match value {
                        $crate::field::Value::Record(name, values) if name == $Record::NAME => {
                            $Record::from_values(values)
                        }
                        other => Err(other.mismatch(concat!("a ", stringify!($Record)))),
                    }
                }
            }
        )*

        /// A statement of any type: one variant per statement type.
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub enum Statement {
            $( $StatementType($StatementType), )*
        }

        /// Every statement type by name, in declaration order.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
        pub enum StatementType {
            $( $StatementType, )*
        }

        impl StatementType {
            /// Every statement type, in declaration order.
            pub const ALL: &[StatementType] = &[ $( StatementType::$StatementType, )* ];

            pub fn name(self) -> &'static str {
                match self { $( StatementType::$StatementType => $StatementType::NAME, )* }
            }

            pub fn layer(self) -> Layer {
                match self { $( StatementType::$StatementType => $StatementType::LAYER, )* }
            }

            pub fn fields(self) -> &'static [FieldInfo] {
                match self { $( StatementType::$StatementType => $StatementType::FIELDS, )* }
            }

            pub fn field_ty(self, field: &str) -> Option<$crate::field::FieldType> {
                match self { $( StatementType::$StatementType => $StatementType::field_ty(field), )* }
            }

            /// A statement of this type from its field values.
            pub fn from_values(self, values: $crate::field::Values) -> $crate::Result<Statement> {
                match self { $( StatementType::$StatementType => Ok(Statement::$StatementType($StatementType::from_values(values)?)), )* }
            }
        }

        impl std::str::FromStr for StatementType {
            type Err = $crate::Error;

            fn from_str(name: &str) -> $crate::Result<StatementType> {
                match name {
                    $( stringify!($StatementType) => Ok(StatementType::$StatementType), )*
                    _ => Err($crate::Error::Value(format!("no statement type {name}"))),
                }
            }
        }

        impl std::fmt::Display for StatementType {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.name())
            }
        }

        impl Statement {
            pub fn ty(&self) -> StatementType {
                match self { $( Statement::$StatementType(_) => StatementType::$StatementType, )* }
            }

            pub fn layer(&self) -> Layer {
                self.ty().layer()
            }

            pub fn flags(&self) -> StatementFlags {
                match self { $( Statement::$StatementType(_) => $StatementType::FLAGS, )* }
            }

            pub fn fields(&self) -> &'static [FieldInfo] {
                self.ty().fields()
            }

            /// The operation a statement with an `op` field holds.
            pub fn op(&self) -> Option<&$crate::elementwise::ElementwiseOp> {
                match self { $( Statement::$StatementType(k) => k.op(), )* }
            }

            /// This statement with another operation, for a statement type that holds one; itself for any other.
            pub fn with_op(&self, op: $crate::elementwise::ElementwiseOp) -> Statement {
                match self { $( Statement::$StatementType(k) => Statement::$StatementType(k.with_op(op)), )* }
            }

            /// The dtype a statement's `dtype` field holds: the value-defining shared statement types, and TreeHalve,
            /// WarpShuffle, FragmentRowReduce and RegFragment, whose field is typed `DType` too.
            pub fn dtype(&self) -> Option<&$crate::dtype::DType> {
                match self { $( Statement::$StatementType(k) => k.dtype(), )* }
            }

            /// The one name a statement's defining `name` field binds: the single-value shared statement types, Call,
            /// and RegFragment and WgmmaDescriptor.
            pub fn name(&self) -> Option<&$crate::Name> {
                match self { $( Statement::$StatementType(k) => k.name(), )* }
            }

            /// The checks a statement makes on construction: every expression a classed field holds is of its class,
            /// and `validate` of every `CHECKED` statement type.
            pub fn check(&self) -> $crate::Result<()> {
                match self { $( Statement::$StatementType(k) => {
                    $( $crate::statement::classed(&k.$field, Roles::NONE $(.or(define_statement_types!(@role $role)))*.or($crate::__class_role!($($class)?)), concat!(stringify!($StatementType), ".", stringify!($field)))?; )*
                    define_statement_types!(@check k {$($flag)|*})
                } )* }
            }

            pub fn visit(&self, visitor: &mut dyn $crate::field::Visitor) {
                match self { $( Statement::$StatementType(k) => k.visit(visitor), )* }
            }

            pub fn rewrite_fields(&self, rewriter: &mut dyn $crate::field::Rewriter) -> Statement {
                match self { $( Statement::$StatementType(k) => Statement::$StatementType(k.rewrite(rewriter)), )* }
            }

            pub fn same(&self, other: &Statement) -> bool {
                match (self, other) {
                    $( (Statement::$StatementType(a), Statement::$StatementType(b)) => a.same(b), )*
                    _ => false,
                }
            }

            pub fn header_fields(&self) -> $crate::Result<Vec<(&'static str, String)>> {
                match self { $( Statement::$StatementType(k) => k.header_fields(), )* }
            }

            pub fn block_fields(&self) -> Vec<(&'static str, Vec<std::sync::Arc<Statement>>)> {
                match self { $( Statement::$StatementType(k) => k.block_fields(), )* }
            }
        }

        $(
            impl From<$StatementType> for Statement {
                fn from(stmt: $StatementType) -> Statement {
                    Statement::$StatementType(stmt)
                }
            }
        )*
    };
    (@fields $Type:ident { $( $field:ident : $ty:ty [$($role:ident)|*] $(: $class:ident)? $(= $default:expr)? ,)* }) => {
        pub const NAME: &'static str = stringify!($Type);
        pub const FIELDS: &'static [FieldInfo] = &[
            $( FieldInfo { name: stringify!($field), roles: Roles::NONE $(.or(define_statement_types!(@role $role)))*.or($crate::__class_role!($($class)?)), has_default: $crate::__has_default!($($default)?), kw_only: define_statement_types!(@kw_only $($role)|*) }, )*
        ];

        /// Visit every role-playing value, in field order.
        pub fn visit(&self, visitor: &mut dyn $crate::field::Visitor) {
            $( <$ty as $crate::field::Field>::visit(&self.$field, Roles::NONE $(.or(define_statement_types!(@role $role)))*.or($crate::__class_role!($($class)?)), visitor); )*
        }

        /// The same with every role-playing value passed through `rewriter`.
        pub fn rewrite(&self, rewriter: &mut dyn $crate::field::Rewriter) -> $Type {
            $Type { $( $field: <$ty as $crate::field::Field>::rewrite(&self.$field, Roles::NONE $(.or(define_statement_types!(@role $role)))*.or($crate::__class_role!($($class)?)), rewriter), )* }
        }

        /// Whether `other` holds the same values, field by field, by identity where a field is shared.
        pub fn same(&self, other: &$Type) -> bool {
            true $( && <$ty as $crate::field::Field>::same(&self.$field, &other.$field) )*
        }

        /// Each field's header text, left out for a block field or one at its default.
        pub fn header_fields(&self) -> $crate::Result<Vec<(&'static str, String)>> {
            #[allow(unused_mut)]
            let mut out = Vec::new();
            $(
                if !<$ty as $crate::field::Field>::is_block() {
                    let at_default = define_statement_types!(@at_default self $field $ty $(= $default)?);
                    if !at_default {
                        let mut text = String::new();
                        <$ty as $crate::field::Field>::spell(&self.$field, &mut text)?;
                        out.push((stringify!($field), text));
                    }
                }
            )*
            Ok(out)
        }

        /// Each block field's statements, left out at its default.
        pub fn block_fields(&self) -> Vec<(&'static str, Vec<std::sync::Arc<Statement>>)> {
            #[allow(unused_mut)]
            let mut out = Vec::new();
            $(
                if <$ty as $crate::field::Field>::is_block() {
                    let at_default = define_statement_types!(@at_default self $field $ty $(= $default)?);
                    if !at_default {
                        out.push((stringify!($field), <$ty as $crate::field::Field>::block(&self.$field)));
                    }
                }
            )*
            out
        }

        /// The type a field is read by.
        pub fn field_ty(name: &str) -> Option<$crate::field::FieldType> {
            match name {
                $( stringify!($field) => Some(<$ty as $crate::field::Field>::ty()), )*
                _ => None,
            }
        }

        /// Build from the values the text reader collected, defaults filling what the text left out.
        pub fn from_values(mut values: $crate::field::Values) -> $crate::Result<$Type> {
            #[allow(unused_mut, unused_variables)]
            let out = $Type {
                $( $field: define_statement_types!(@take values $Type $field $ty $(= $default)?), )*
            };
            values.finish(Self::NAME)?;
            Ok(out)
        }

        /// The operation this statement holds: its `op` field, when it has one.
        pub fn op(&self) -> Option<&$crate::elementwise::ElementwiseOp> {
            $( if stringify!($field) == "op" { return <$ty as $crate::field::Field>::as_op(&self.$field); } )*
            None
        }

        /// The same with another operation; itself when it holds none.
        pub fn with_op(&self, op: $crate::elementwise::ElementwiseOp) -> $Type {
            $Type { $( $field: if stringify!($field) == "op" { <$ty as $crate::field::Field>::with_op(&self.$field, &op) } else { self.$field.clone() }, )* }
        }

        /// The dtype a value-defining statement stamps: its `dtype` field, when it has one of that type.
        pub fn dtype(&self) -> Option<&$crate::dtype::DType> {
            $( if stringify!($field) == "dtype" { return <$ty as $crate::field::Field>::as_dtype(&self.$field); } )*
            None
        }

        /// The one name a single-value statement defines: its `name` field, when that field defines.
        pub fn name(&self) -> Option<&$crate::Name> {
            $( if stringify!($field) == "name" && (Roles::NONE $(.or(define_statement_types!(@role $role)))*.or($crate::__class_role!($($class)?))).contains_any(DEFINES) { return <$ty as $crate::field::Field>::as_name(&self.$field); } )*
            None
        }
    };
    (@layer shared) => { Layer::Shared };
    (@layer kernel) => { Layer::Kernel };
    (@role KW_ONLY) => { Roles::NONE };
    (@role $role:ident) => { $role };
    (@kw_only) => { false };
    (@kw_only KW_ONLY $(| $rest:ident)*) => { true };
    (@kw_only $role:ident $(| $rest:ident)*) => { define_statement_types!(@kw_only $($rest)|*) };
    (@check $k:ident {}) => {{ let _ = &$k; Ok(()) }};
    (@check $k:ident {CHECKED $(| $rest:tt)*}) => { $k.validate() };
    (@check $k:ident {$flag:tt $(| $rest:tt)*}) => { define_statement_types!(@check $k {$($rest)|*}) };
    (@at_default $self:ident $field:ident $ty:ty = $default:expr) => {
        <$ty as $crate::field::Field>::equals(&$self.$field, &$default)
    };
    (@at_default $self:ident $field:ident $ty:ty) => { false };
    (@take $values:ident $Type:ident $field:ident $ty:ty = $default:expr) => {
        match $values.take(stringify!($field)) {
            Some(value) => <$ty as $crate::field::Field>::from_value(value)?,
            None => $default,
        }
    };
    (@take $values:ident $Type:ident $field:ident $ty:ty) => {
        match $values.take(stringify!($field)) {
            Some(value) => <$ty as $crate::field::Field>::from_value(value)?,
            None => return Err($crate::Error::Value(format!("{} requires {}", stringify!($Type), stringify!($field)))),
        }
    };
}

use crate::Name;
use crate::body::Body;
use crate::dtype::DType;
use crate::elementwise::ElementwiseOp;
use crate::expr::Expression;
use crate::field::{
    Axes, AxisRef, Branches, ElementwiseOps, Exprs, Float, FragmentArgs, Int2, Ints, Names, NumOr,
    Shape3, Stmts, SubroutineRef,
};

with_statement_types!(define_statement_types);

#[cfg(test)]
mod tests {
    use super::*;

    /// The declaration list, as statement types and records with their fields in order: what every stored body's
    /// text, every golden and every tune DB key spell. A change here is a change to all of them.
    #[test]
    fn the_declarations_are_pinned() {
        let mut listing = String::new();
        fn spelled(fields: &[FieldInfo]) -> String {
            fields
                .iter()
                .map(|f| {
                    f.class()
                        .map_or_else(|| f.name.to_string(), |class| format!("{}:{class}", f.name))
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
        for ty in StatementType::ALL {
            listing.push_str(&format!("{} {}\n", ty.name(), spelled(ty.fields())));
        }
        for (name, fields) in [
            (SelectBranch::NAME, SelectBranch::FIELDS),
            (FragmentArg::NAME, FragmentArg::FIELDS),
        ] {
            listing.push_str(&format!("{name} {}\n", spelled(fields)));
        }
        assert_eq!(
            blake3::hash(listing.as_bytes()).to_hex().as_str(),
            "f186c7d8285bf9b9363d7e57e0713b688fc510be96071ac709885b167e6079b7",
            "{listing}"
        );
    }

    /// A field has a class exactly when it plays `EXPRESSION`: the check reads the class from the roles, so an
    /// expression field without one would go unchecked, and a class on anything else would check nothing.
    #[test]
    fn a_field_has_a_class_exactly_when_it_holds_expressions() {
        let records = [
            (SelectBranch::NAME, SelectBranch::FIELDS),
            (FragmentArg::NAME, FragmentArg::FIELDS),
        ];
        let fields = StatementType::ALL
            .iter()
            .map(|ty| (ty.name(), ty.fields()))
            .chain(records);
        for (name, fields) in fields {
            for field in fields {
                assert_eq!(
                    field.roles.contains_any(EXPRESSION),
                    field.class().is_some(),
                    "{name}.{}",
                    field.name
                );
            }
        }
    }
}
