//! The statement kinds, declared as data: every kind is a struct whose fields carry a type and the roles they play.
//! The declarations are one macro ([`with_kinds!`]) so the Python extension generates its classes from the same
//! list, and [`define_kinds!`] turns the list into the structs, the `Statement` enum and the protocol walks over
//! the roles.
//!
//! A role says what a field is to the rest of the compiler: `DEFINES` / `READS` / `UPDATES` / `LOCAL` for SSA names,
//! `BUFFER_READ` / `BUFFER_WRITE` / `LOCAL_BUFFER` for buffers, `EXPRESSION`, `AXIS` and `AXIS_NAME`, `BODY` for a
//! child body, `STMTS` for statements the kind carries, `TERM` for the body of a pure term whose names are private,
//! and `RECORD` for records whose own fields play roles. A number where a name, a buffer or an expression is expected
//! is plain data ([`NumOr`]). A field with no role is plain data.

/// Roles, as bits.
pub const DEFINES: u32 = 1 << 0;
pub const READS: u32 = 1 << 1;
pub const UPDATES: u32 = 1 << 2;
pub const LOCAL: u32 = 1 << 3;
pub const BUFFER_READ: u32 = 1 << 4;
pub const BUFFER_WRITE: u32 = 1 << 5;
pub const LOCAL_BUFFER: u32 = 1 << 6;
pub const EXPRESSION: u32 = 1 << 7;
pub const AXIS: u32 = 1 << 8;
pub const AXIS_NAME: u32 = 1 << 9;
pub const BODY: u32 = 1 << 10;
pub const STMTS: u32 = 1 << 11;
pub const TERM: u32 = 1 << 12;
pub const RECORD: u32 = 1 << 13;
/// A constructor flag, not a role: the field is given by keyword only (an `Init`'s dtype).
pub const KW_ONLY: u32 = 1 << 20;

/// The SSA-name roles a rename maps.
pub const NAMES: u32 = DEFINES | READS | UPDATES | LOCAL;
pub const DEFINING: u32 = DEFINES | UPDATES;
pub const READING: u32 = READS | UPDATES;
pub const BUFFERS: u32 = BUFFER_READ | BUFFER_WRITE;
pub const CARRIED: u32 = STMTS | TERM;

/// Kind flags.
pub const PURE: u32 = 1;
pub const SIDE_EFFECTS: u32 = 2;
pub const DEPS_DEEP: u32 = 4;

/// Every statement kind and record, declared once. `$callback` receives the list and generates from it.
#[macro_export]
macro_rules! with_kinds {
    ($callback:ident) => {
        $callback! {
            kinds {
                /// Read a value, or N consecutive values, from an external buffer: lane `k` reads `index[:-1] + (index[-1] + k,)` into `names[k]`. `dtype` names the source element type once the type stamp set it. `carried` marks the two halves of a staged copy crossing a chunk loop's back edge (`load` re-assigns the packed vector, `unpack` the names from it).
                shared Load {PURE} {
                    names: Names [DEFINES],
                    input: Name [BUFFER_READ],
                    index: Exprs [EXPRESSION | READS],
                    dtype: Option<DType> [] = None,
                    carried: Name [] = Name::from(""),
                }
                /// Pure SSA statement `name = op(args)`. `dtype` overrides the promotion rule when set.
                shared Assign {PURE} {
                    name: Name [DEFINES],
                    op: Op [],
                    args: Names [READS],
                    dtype: Option<DType> [] = None,
                }
                /// Pure SSA statement `name = offset + Σ coefficient·arg`: the one statement for a sum, a difference, a negation or a product with a literal. `args` are SSA names, one per term, no coefficient zero.
                shared Affine {PURE} {
                    name: Name [DEFINES],
                    args: Names [READS],
                    coefficients: Floats [],
                    offset: Float [] = Float(0.0),
                    dtype: Option<DType> [] = None,
                }
                /// Reduce accumulator: `name = op(name, value)` inside the enclosing reduce loop, seeded with the op's identity. `axes` are the axis names it reduces over; `base` redirects the left operand of the fold.
                shared Accum {0} {
                    name: Name [DEFINES],
                    value: Name [READS],
                    op: Op [] = Op::new("add"),
                    dtype: Option<DType> [] = None,
                    axes: Names [] = Names::from([]),
                    base: Option<Name> [READS] = None,
                }
                /// The next value of one cell of a carried state: `name[index] <- value`, seeded by a constant or a buffer of the state's shape.
                shared Carry {0} {
                    name: Name [DEFINES],
                    value: Name [READS],
                    index: Exprs [EXPRESSION],
                    seed: NumOr<Name> [BUFFER_READ],
                    dtype: Option<DType> [] = None,
                }
                /// A read of one cell of a carried state: `name = carrier[index]`, the previous step's value.
                shared Pre {PURE} {
                    name: Name [DEFINES],
                    carrier: Name [READS],
                    index: Exprs [EXPRESSION | READS],
                }
                /// An explicit accumulator or carried-state seed at this scope: `<dtype> <name> = <identity>;`.
                shared Init {0} {
                    name: Name [DEFINES],
                    identity: Float [],
                    dtype: DType [KW_ONLY],
                }
                /// A pure binding of one expression to an SSA name: a literal, a precomputed index or a flattened buffer offset.
                shared Let {PURE} {
                    name: Name [DEFINES],
                    value: Expression [EXPRESSION | READS],
                    dtype: Option<DType> [] = None,
                }
                /// Write an SSA value, or N consecutive values, to `output` at `index`; `atomic` accumulates with `atomicAdd`, `swizzle` permutes a slab's address.
                shared Write {SIDE_EFFECTS} {
                    output: Name [BUFFER_WRITE],
                    index: Exprs [EXPRESSION | READS],
                    values: Names [READS],
                    value_dtype: Option<DType> [] = None,
                    atomic: bool [] = false,
                    swizzle: Name [] = Name::from("NONE"),
                }
                /// Coordinate-predicated value binding: the first branch whose predicate holds binds its value to `name`.
                shared Select {PURE} {
                    name: Name [DEFINES],
                    branches: Branches [RECORD],
                }
                /// Zero another launch's atomic accumulator from this kernel: `words` 32-bit words of `dst`.
                shared ZeroPrologue {SIDE_EFFECTS} {
                    dst: Name [BUFFER_WRITE],
                    words: i64 [],
                }
                /// One loop over an axis. A reduce loop folds the `Accum` statements of its body; `unroll` asks for `#pragma unroll`; `seed` emits the per-`Accum` identity seed ahead of the loop.
                shared Loop {0} {
                    axis: AxisRef [AXIS],
                    body: Body [BODY],
                    unroll: bool [] = false,
                    seed: bool [] = true,
                }
                /// Strided iteration `for (axis = start; axis < end or extent; axis += step)`: threads of a block striding through an axis.
                shared StridedLoop {0} {
                    axis: AxisRef [AXIS],
                    start: Expression [EXPRESSION],
                    step: NumOr<Expression> [EXPRESSION],
                    body: Body [BODY],
                    unroll: bool [] = false,
                    end: Option<Expression> [EXPRESSION] = None,
                    seed: bool [] = true,
                }
                /// Conditional block `if cond: body else: else_body`.
                shared Cond {0} {
                    cond: Expression [EXPRESSION | READS],
                    body: Body [BODY],
                    else_body: Body [BODY] = Body::empty(),
                }
                /// A fusion-only scalar call of a shared reduction body; fusion expands every call before a body is stored.
                shared Call {DEPS_DEEP} {
                    name: Name [DEFINES],
                    target: SubroutineRef [],
                    args: Exprs [EXPRESSION | READS],
                }
                /// A kernel-local shared-memory buffer.
                kernel Smem {0} {
                    name: Name [LOCAL_BUFFER],
                    extents: Ints [],
                    dtype: Name [] = Name::from("float"),
                    align: i64 [] = 0,
                }
                /// A block or warp barrier.
                kernel Sync {0} {
                    barrier_id: i64 [] = 0,
                    count: Option<i64> [] = None,
                    warp: bool [] = false,
                }
                /// Tile coordinate binding: one thread per cell of the product of `axes`, cooperative when `block_threads` is set, `aux_threads` producer threads past the cells, and the CTA raster order.
                kernel Tile {0} {
                    axes: Axes [AXIS],
                    body: Body [BODY],
                    block_threads: Option<i64> [] = None,
                    aux_threads: i64 [] = 0,
                    raster_axes: Option<(Name, Name)> [AXIS_NAME] = None,
                    raster_group: Option<i64> [] = None,
                    raster_orient: Name [] = Name::from("m"),
                    spread: bool [] = false,
                }
                /// One `cp.async` copy of `nbytes` from global `src[src_index]` into `smem[smem_index]`.
                kernel CpAsyncCopy {0} {
                    smem: Name [],
                    smem_index: Exprs [EXPRESSION],
                    src: Name [BUFFER_READ],
                    src_index: Exprs [EXPRESSION],
                    nbytes: i64 [] = 4,
                    swizzle: Name [] = Name::from("NONE"),
                    lane_index: Option<Exprs> [EXPRESSION] = None,
                    lane_rows: i64 [] = 0,
                    valid: Option<Expression> [EXPRESSION] = None,
                }
                /// Commit the pending `cp.async` copies as one group.
                kernel CpAsyncCommit {0} {}
                /// Wait until at most `group` `cp.async` groups are pending.
                kernel CpAsyncWait {0} {
                    group: i64 [] = 0,
                }
                /// A TMA tensor-map descriptor over `src_buf`.
                kernel TmaDescriptor {0} {
                    name: Name [],
                    src_buf: Name [BUFFER_READ],
                    src_shape: Ints [],
                    box_extents: Ints [],
                    swizzle: Name [] = Name::from("NONE"),
                    dtype: Name [] = Name::from("float"),
                }
                /// One TMA load of a box into shared memory, signalled on an mbarrier.
                kernel TmaLoad {0} {
                    smem: Name [],
                    smem_index: Exprs [EXPRESSION],
                    desc: Name [],
                    coords: Exprs [EXPRESSION],
                    mbar: Name [],
                    mbar_slot: Option<Expression> [EXPRESSION] = None,
                }
                /// Initialize an mbarrier.
                kernel MbarrierInit {0} {
                    mbar: Name [],
                    count: i64 [] = 1,
                    slot: Option<Expression> [EXPRESSION] = None,
                }
                /// Arrive on an mbarrier expecting `bytes_` of transactions.
                kernel MbarrierArriveExpectTx {0} {
                    mbar: Name [],
                    bytes_: i64 [],
                    slot: Option<Expression> [EXPRESSION] = None,
                }
                /// Arrive on an mbarrier.
                kernel MbarrierArrive {0} {
                    mbar: Name [],
                    slot: Option<Expression> [EXPRESSION] = None,
                }
                /// Raise or lower the warp group's register budget.
                kernel SetMaxNReg {0} {
                    count: i64 [],
                    direction: Name [],
                }
                /// Wait on an mbarrier phase.
                kernel MbarrierWait {0} {
                    mbar: Name [],
                    phase: Expression [EXPRESSION],
                    slot: Option<Expression> [EXPRESSION] = None,
                }
                /// A block-wide tree reduction over shared memory: `combine_states` folds each state with its `state_b` copy.
                kernel TreeHalve {0} {
                    bufs: Names [],
                    state: Names [READS],
                    state_b: Names [LOCAL],
                    combine_states: Stmts [STMTS],
                    length: i64 [],
                    tid_var: Name [],
                    dtype: DType [] = DType::new("f32"),
                    barrier_id: i64 [] = 0,
                    barrier_count: Option<i64> [] = None,
                    inner: Option<(Name, i64)> [] = None,
                }
                /// A warp shuffle reduction or prefix scan over `length` lanes.
                kernel WarpShuffle {0} {
                    state: Names [READS],
                    state_b: Names [LOCAL],
                    combine_states: Stmts [STMTS],
                    length: i64 [],
                    dtype: DType [] = DType::new("f32"),
                    prefix: bool [] = false,
                }
                /// Broadcast a state from the last lane of a `length`-lane group.
                kernel WarpBroadcast {0} {
                    state: Names [READS],
                    length: i64 [],
                }
                /// The per-element pointwise node over an mma C fragment.
                kernel FragmentApply {0} {
                    out: Name [DEFINES],
                    op: Op [],
                    args: FragmentArgs [RECORD],
                    in_place: bool [] = false,
                    layout: Name [] = Name::from("m16n8k16"),
                    post: Ops [] = Ops::from([]),
                    row_base: Option<Expression> [EXPRESSION] = None,
                    col_base: Option<Expression> [EXPRESSION] = None,
                }
                /// A per-row fold over the C fragments spanning a chunk, into two per-lane row values.
                kernel FragmentRowReduce {0} {
                    top: Name [DEFINES],
                    bot: Name [DEFINES],
                    frags: Names [READS],
                    op: Op [],
                    layout: Name [] = Name::from("m16n8k16"),
                    dtype: DType [] = DType::new("f32"),
                }
                /// A register fragment declaration.
                kernel RegFragment {0} {
                    name: Name [DEFINES | LOCAL_BUFFER],
                    role: Name [],
                    shape: Shape3 [],
                    dtype: DType [],
                    count: i64 [] = 1,
                    nregs: Option<i64> [] = None,
                }
                /// A block-scale fragment load.
                kernel BlockScaleLoad {0} {
                    frag: Name [DEFINES],
                    src_buffer: Name [BUFFER_READ],
                    src_index: Exprs [EXPRESSION],
                    role: Name [],
                    ldm: i64 [] = 0,
                }
                /// An `ldmatrix` fragment load from a slab (or gmem-direct), with its guards, pair and scales.
                kernel LdmatrixLoad {0} {
                    frag: Name [UPDATES],
                    src_buffer: Name [BUFFER_READ],
                    src_index: Exprs [EXPRESSION],
                    role: Name [],
                    ldm: i64 [] = 0,
                    swizzle: Name [] = Name::from("NONE"),
                    staged: bool [] = true,
                    gmem_guard: Option<(Expression, Expression)> [EXPRESSION] = None,
                    k_zero: Option<(Expression, Expression)> [EXPRESSION] = None,
                    b_trans: bool [] = false,
                    pair_frag: Option<Name> [UPDATES] = None,
                    byte_slab: bool [] = false,
                    scale_buffer: Option<Name> [BUFFER_READ] = None,
                    scale_index: Exprs [EXPRESSION] = Exprs::from([]),
                    scale_ldm: i64 [] = 0,
                    fragment_layout: Name [] = Name::from("m16n8k16"),
                    fragment_index: i64 [] = 0,
                }
                /// One `mma.sync` over fragments.
                kernel MmaSyncPtx {0} {
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
                /// Fold a packed f16 accumulator fragment into its f32 shadow and rezero it.
                kernel FragmentPromote {0} {
                    dst: Name [UPDATES],
                    src: Name [UPDATES],
                    fragment_layout: Name [] = Name::from("m16n8k16"),
                }
                /// Repack C fragments into an A or B operand fragment.
                kernel FragmentRepack {0} {
                    frag: Name [DEFINES],
                    srcs: Names [READS],
                    ab_dtype: Name [] = Name::from("f16"),
                    fragment_layout: Name [] = Name::from("m16n8k16"),
                    part: i64 [] = 0,
                    role: Name [] = Name::from("a"),
                }
                /// A `wgmma` shared-memory matrix descriptor.
                kernel WgmmaDescriptor {0} {
                    name: Name [DEFINES],
                    smem: Name [BUFFER_READ],
                    smem_index: Option<Expression> [EXPRESSION],
                    swizzle: Name [],
                    lbo_bytes: i64 [],
                    sbo_bytes: i64 [],
                }
                /// One `wgmma` over descriptors or a register A fragment.
                kernel WgmmaMma {0} {
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
                /// Fence the accumulator registers ahead of a `wgmma`.
                kernel WgmmaFence {0} {}
                /// Commit the pending `wgmma` operations as a group.
                kernel WgmmaCommit {0} {}
                /// Wait until at most `group` `wgmma` groups are pending.
                kernel WgmmaWait {0} {
                    group: i64 [] = 0,
                }
                /// Store an mma accumulator fragment to `dst_buffer` with a per-lane epilogue, its guards, and the cells stored with it.
                kernel RegStore {0} {
                    dst_buffer: Name [BUFFER_WRITE],
                    dst_index: Exprs [EXPRESSION],
                    frag: Name [READS],
                    shape: Shape3 [],
                    ldm: i64 [] = 0,
                    ldn: i64 [] = 0,
                    epilogue: Option<Body> [TERM] = None,
                    epilogue_params: Names [] = Names::from([]),
                    epilogue_result: Option<Name> [] = None,
                    extra_frags: Names [READS] = Names::from([]),
                    m_guard: Option<(Expression, Expression)> [EXPRESSION] = None,
                    n_guard: Option<(Expression, Expression)> [EXPRESSION] = None,
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
                /// One branch of a `Select`: its value when its predicate holds.
                SelectBranch {
                    value: Name [READS],
                    select: Expression [EXPRESSION | READS],
                }
                /// One operand of a `FragmentApply`; what it holds follows its `kind`.
                FragmentArg {
                    kind: Name [],
                    names: Names [READS] = Names::from([]),
                    expressions: Exprs [EXPRESSION] = Exprs::from([]),
                    buffer: Option<Name> [BUFFER_READ] = None,
                }
            }
        }
    };
}

/// Where a kind is exported from: the shared statement module or the Kernel IR.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layer {
    Shared,
    Kernel,
}

/// One field of a declared kind: its name, its roles and whether it has a default.
#[derive(Clone, Copy, Debug)]
pub struct FieldInfo {
    pub name: &'static str,
    pub roles: u32,
    pub has_default: bool,
}

impl FieldInfo {
    /// A block field: spelled as statements under the header, not on it.
    pub fn is_block(&self) -> bool {
        self.roles & (BODY | TERM | STMTS) != 0
    }
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
macro_rules! define_kinds {
    (
        kinds { $( $(#[doc = $doc:literal])* $layer:ident $Kind:ident {$($flag:tt)|*} { $( $field:ident : $ty:ty [$($role:ident)|*] $(= $default:expr)? ,)* } )* }
        records { $( $(#[doc = $rdoc:literal])* $Record:ident { $( $rfield:ident : $rty:ty [$($rrole:ident)|*] $(= $rdefault:expr)? ,)* } )* }
    ) => {
        $(
            $(#[doc = $doc])*
            #[derive(Clone, PartialEq, Eq, Hash, Debug)]
            pub struct $Kind {
                $( pub $field: $ty, )*
            }

            #[allow(non_upper_case_globals, dead_code, unused_variables, unused_mut)]
            impl $Kind {
                pub const NAME: &'static str = stringify!($Kind);
                pub const LAYER: Layer = define_kinds!(@layer $layer);
                pub const FLAGS: u32 = 0 $(| $flag)*;
                pub const FIELDS: &'static [FieldInfo] = &[
                    $( FieldInfo { name: stringify!($field), roles: 0 $(| $role)*, has_default: $crate::__has_default!($($default)?) }, )*
                ];

                /// Visit every role-playing value of this statement, in field order.
                pub fn visit(&self, visitor: &mut dyn $crate::field::Visitor) {
                    $( <$ty as $crate::field::Field>::visit(&self.$field, 0 $(| $role)*, visitor); )*
                }

                /// This statement with every role-playing value passed through `rewriter`.
                pub fn rewrite(&self, rewriter: &mut dyn $crate::field::Rewriter) -> $Kind {
                    $Kind { $( $field: <$ty as $crate::field::Field>::rewrite(&self.$field, 0 $(| $role)*, rewriter), )* }
                }

                /// Whether `other` holds the same values, field by field, by identity where a field is shared.
                pub fn same(&self, other: &$Kind) -> bool {
                    true $( && <$ty as $crate::field::Field>::same(&self.$field, &other.$field) )*
                }

                /// Each field's header text, `None` for a block field or one at its default.
                pub fn header_fields(&self) -> $crate::Result<Vec<(&'static str, String)>> {
                    #[allow(unused_mut)]
                    let mut out = Vec::new();
                    $(
                        if !<$ty as $crate::field::Field>::is_block() {
                            let at_default = define_kinds!(@at_default self $field $ty $(= $default)?);
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
                            let at_default = define_kinds!(@at_default self $field $ty $(= $default)?);
                            if !at_default {
                                out.push((stringify!($field), <$ty as $crate::field::Field>::block(&self.$field)));
                            }
                        }
                    )*
                    out
                }

                /// The type a field is read by.
                pub fn field_ty(name: &str) -> Option<$crate::field::Ty> {
                    match name {
                        $( stringify!($field) => Some(<$ty as $crate::field::Field>::ty()), )*
                        _ => None,
                    }
                }

                /// Build from the values the text reader collected, defaults filling what the text left out.
                pub fn from_values(mut values: $crate::field::Values) -> $crate::Result<$Kind> {
                    #[allow(unused_mut, unused_variables)]
                    let out = $Kind {
                        $( $field: define_kinds!(@take values $Kind $field $ty $(= $default)?), )*
                    };
                    values.finish(Self::NAME)?;
                    Ok(out)
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
                pub const NAME: &'static str = stringify!($Record);
                pub const FIELDS: &'static [FieldInfo] = &[
                    $( FieldInfo { name: stringify!($rfield), roles: 0 $(| $rrole)*, has_default: $crate::__has_default!($($rdefault)?) }, )*
                ];

                pub fn visit(&self, visitor: &mut dyn $crate::field::Visitor) {
                    $( <$rty as $crate::field::Field>::visit(&self.$rfield, 0 $(| $rrole)*, visitor); )*
                }

                pub fn rewrite(&self, rewriter: &mut dyn $crate::field::Rewriter) -> $Record {
                    $Record { $( $rfield: <$rty as $crate::field::Field>::rewrite(&self.$rfield, 0 $(| $rrole)*, rewriter), )* }
                }

                pub fn same(&self, other: &$Record) -> bool {
                    true $( && <$rty as $crate::field::Field>::same(&self.$rfield, &other.$rfield) )*
                }

                pub fn header_fields(&self) -> $crate::Result<Vec<(&'static str, String)>> {
                    #[allow(unused_mut)]
                    let mut out = Vec::new();
                    $(
                        let at_default = define_kinds!(@at_default self $rfield $rty $(= $rdefault)?);
                        if !at_default {
                            let mut text = String::new();
                            <$rty as $crate::field::Field>::spell(&self.$rfield, &mut text)?;
                            out.push((stringify!($rfield), text));
                        }
                    )*
                    Ok(out)
                }

                pub fn field_ty(name: &str) -> Option<$crate::field::Ty> {
                    match name {
                        $( stringify!($rfield) => Some(<$rty as $crate::field::Field>::ty()), )*
                        _ => None,
                    }
                }

                pub fn from_values(mut values: $crate::field::Values) -> $crate::Result<$Record> {
                    #[allow(unused_mut, unused_variables)]
                    let out = $Record {
                        $( $rfield: define_kinds!(@take values $Record $rfield $rty $(= $rdefault)?), )*
                    };
                    values.finish(Self::NAME)?;
                    Ok(out)
                }
            }
        )*

        /// Every statement kind, one variant each.
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        pub enum Statement {
            $( $Kind($Kind), )*
        }

        /// The names of every kind, in declaration order.
        pub const KIND_NAMES: &[&str] = &[ $( stringify!($Kind), )* ];

        impl Statement {
            pub fn kind_name(&self) -> &'static str {
                match self { $( Statement::$Kind(_) => $Kind::NAME, )* }
            }

            pub fn layer(&self) -> Layer {
                match self { $( Statement::$Kind(_) => $Kind::LAYER, )* }
            }

            pub fn flags(&self) -> u32 {
                match self { $( Statement::$Kind(_) => $Kind::FLAGS, )* }
            }

            pub fn fields(&self) -> &'static [FieldInfo] {
                match self { $( Statement::$Kind(_) => $Kind::FIELDS, )* }
            }

            /// The fields of the kind called `name`, or `None` for no kind.
            pub fn fields_of(name: &str) -> Option<&'static [FieldInfo]> {
                match name {
                    $( stringify!($Kind) => Some($Kind::FIELDS), )*
                    _ => None,
                }
            }

            pub fn field_ty(kind: &str, field: &str) -> Option<$crate::field::Ty> {
                match kind {
                    $( stringify!($Kind) => $Kind::field_ty(field), )*
                    _ => None,
                }
            }

            pub fn from_values(kind: &str, values: $crate::field::Values) -> $crate::Result<Statement> {
                match kind {
                    $( stringify!($Kind) => Ok(Statement::$Kind($Kind::from_values(values)?)), )*
                    _ => Err($crate::Error::Value(format!("no statement kind {kind}"))),
                }
            }

            pub fn visit(&self, visitor: &mut dyn $crate::field::Visitor) {
                match self { $( Statement::$Kind(k) => k.visit(visitor), )* }
            }

            pub fn rewrite_fields(&self, rewriter: &mut dyn $crate::field::Rewriter) -> Statement {
                match self { $( Statement::$Kind(k) => Statement::$Kind(k.rewrite(rewriter)), )* }
            }

            pub fn same(&self, other: &Statement) -> bool {
                match (self, other) {
                    $( (Statement::$Kind(a), Statement::$Kind(b)) => a.same(b), )*
                    _ => false,
                }
            }

            pub fn header_fields(&self) -> $crate::Result<Vec<(&'static str, String)>> {
                match self { $( Statement::$Kind(k) => k.header_fields(), )* }
            }

            pub fn block_fields(&self) -> Vec<(&'static str, Vec<std::sync::Arc<Statement>>)> {
                match self { $( Statement::$Kind(k) => k.block_fields(), )* }
            }
        }

        $(
            impl From<$Kind> for Statement {
                fn from(kind: $Kind) -> Statement {
                    Statement::$Kind(kind)
                }
            }
        )*
    };
    (@layer shared) => { Layer::Shared };
    (@layer kernel) => { Layer::Kernel };
    (@at_default $self:ident $field:ident $ty:ty = $default:expr) => {
        <$ty as $crate::field::Field>::equals(&$self.$field, &$default)
    };
    (@at_default $self:ident $field:ident $ty:ty) => { false };
    (@take $values:ident $kind:ident $field:ident $ty:ty = $default:expr) => {
        match $values.take(stringify!($field)) {
            Some(value) => <$ty as $crate::field::Field>::from_value(value)?,
            None => $default,
        }
    };
    (@take $values:ident $kind:ident $field:ident $ty:ty) => {
        match $values.take(stringify!($field)) {
            Some(value) => <$ty as $crate::field::Field>::from_value(value)?,
            None => return Err($crate::Error::Value(format!("{} requires {}", stringify!($kind), stringify!($field)))),
        }
    };
}

pub use crate::field::{Axes, AxisRef, Branches, DType, Exprs, Float, Floats, FragmentArgs, Int2, Ints, Names, NumOr, Ops, Shape3, Stmts, SubroutineRef};
use crate::Name;
use crate::body::Body;
use crate::elementwise::Op;
use crate::expr::Expression;

with_kinds!(define_kinds);
