# Kernel IR operations

Kernel IR is the fifth compiler stage, between Tile IR and CUDA source. A `KernelOp` is one `__global__` GPU kernel:
a graph node whose body is a tree of statements. Tile IR still holds scheduling choices; Kernel IR holds none. Every
decision has already been turned into explicit hardware machinery: the launch geometry, shared-memory arrays, thread
barriers, asynchronous copies, and tensor-core register fragments. The CUDA backend renders the body almost line by
line into kernel source.

The operations fall into five groups:

- **Scalar compute** is shared with Loop IR and Tile IR: loads, pointwise assigns, reduce accumulators, writes, and
  the loop and branch blocks around them.
- **Launch and cooperation** binds threads to cells (`Tile`), declares shared memory (`Smem`), and orders threads with
  barriers (`Sync`).
- **Transport** moves data from global memory into shared memory: `cp.async` copies, TMA box copies, and the
  mbarriers that signal their arrival.
- **Cross-thread combine** folds per-thread partial results into one value, over registers (`WarpShuffle`) or over a
  shared-memory tree (`TreeHalve`).
- **Tensor core** runs matrix multiply-accumulate on register fragments (`mma.sync`) or on warp groups (`wgmma`),
  plus the pointwise, mask, repack and store operations that act on those fragments directly.

The kernel signature is derived from the body. Buffers the body reads, such as `Load` inputs and `CpAsyncCopy`
sources, become input parameters. Buffers it writes become output parameters. `Smem` arrays are never parameters.
Buffer shapes come from the surrounding graph at render time.

Definitions live in `ir/kernel/ir.py` (hardware operations), `ir/stmt/` (shared statements) and `ir/expr.py`
(expressions). Lowering from Tile IR is described in `pipeline/passes/lowering/kernel/ARCHITECTURE.md`.

Operations marked `*` are optional for the Emmy hackathon: no open or hidden evaluation task contains them. A
Kernel IR to PTX compiler that handles only the unmarked operations covers every evaluation task.

## Program

| Operation | Arguments | Description |
| --- | --- | --- |
| `KernelOp` | `name`, `body`, `zero_delegated` | One GPU kernel whose parameters are derived from the body; `zero_delegated` lists outputs whose zero-init a predecessor kernel performs. |

## Scalar compute (shared with Loop IR and Tile IR)

| Operation | Arguments | Description |
| --- | --- | --- |
| `Load` | `names`, `input`, `index`, `dtype` | Read one value, or several consecutive values, from a buffer into SSA names. |
| `Assign` | `name`, `op`, `args`, `dtype` | Bind `name = op(args)` for one elementwise operation. |
| `Accum` | `name`, `value`, `op`, `dtype`, `axes`, `base` | Fold `value` into a reduce accumulator that starts at the operation's identity and is visible after the loop. |
| `Init` | `name`, `identity`, `dtype` | Declare a scope-local seed so a masked tail can select the identity by name. |
| `Const`* | `name`, `value`, `dtype` | Bind one scalar literal to an SSA name. |
| `Select`* | `name`, `branches` (`value`, `select`) | Bind `name` to the value of the branch whose coordinate predicate holds. |
| `Write` | `output`, `index`, `values`, `value_dtype`, `atomic`, `swizzle` | Store one value, or several consecutive values, into a buffer, optionally as an atomic add. |
| `Pack`* | `name`, `low`, `high`, `dtype` | Bundle two scalars into one `__half2` value. |
| `Unpack`* | `low_name`, `high_name`, `value`, `lane_dtype` | Split one `__half2` value back into two `__half` scalars. |
| `ZeroPrologue`* | `dst`, `words` | Zero another kernel's atomic accumulator from an earlier kernel on the same stream, replacing a per-launch memset. |
| `Loop` | `axis`, `body`, `unroll`, `seed` | Run `body` once per value of `axis`, folding any accumulators inside. |
| `StridedLoop` | `axis`, `start`, `step`, `body`, `unroll`, `end`, `seed` | Loop from `start` in steps of `step`, the form cooperative threads use to stride an axis. |
| `Cond` | `cond`, `body`, `else_body` | Run `body` when the predicate holds, otherwise `else_body`. |
| `Mma`* | `c`, `a`, `b`, `atom`, `axes`, `b_trans`, `m_guard`, `n_guard`, `k_zero` | Tile IR form of one tensor-core cell `c += a @ b`; lowering replaces it with `MmaSyncPtx`. |

## Launch, shared memory and barriers

| Operation | Arguments | Description |
| --- | --- | --- |
| `Tile` | `axes`, `body`, `block_threads`, `aux_threads`, `raster_axes`, `raster_group`, `raster_orient` | Map the iteration space onto the thread grid, decode each thread's axis indices, and run `body` once per cell. |
| `Smem` | `name`, `extents`, `dtype`, `align` | Declare one `__shared__` array for the CTA, with optional byte alignment. |
| `IndexDecl`* | `name`, `value` | Bind an integer index once, outside a nested hot loop. |
| `FlatIndexDecl`* | `name`, `buffer`, `index`, `origin` | Bind a buffer coordinate's flattened offset, optionally relative to another coordinate. |
| `Sync` | `barrier_id`, `count`, `warp` | Emit a barrier: CTA-wide `__syncthreads()`, warp-scope `__syncwarp()`, or a named barrier over `count` threads. |
| `Reassign`* | `name`, `value` | Rebind an already declared carried scalar without declaring a new one. |

## Transport into shared memory

| Operation | Arguments | Description |
| --- | --- | --- |
| `CpAsyncCopy` | `smem`, `smem_index`, `src`, `src_index`, `nbytes`, `swizzle`, `lane_index`, `lane_rows` | Copy 4, 8 or 16 bytes from a global buffer straight into shared memory with one `cp.async` instruction. |
| `CpAsyncCommit` | — | Close the thread's preceding `cp.async` copies into one commit group. |
| `CpAsyncWait` | `group` | Block until at most `group` `cp.async` groups remain in flight. |
| `TmaDescriptor`* | `name`, `src_buf`, `src_shape`, `box_extents`, `swizzle`, `dtype` | Declare the host-built tensor-map descriptor a TMA box copy reads; renders nothing in the kernel. |
| `TmaLoad`* | `smem`, `smem_index`, `desc`, `coords`, `mbar`, `mbar_slot` | Copy one box of a global buffer into shared memory with a single-thread `cp.async.bulk.tensor` instruction. |
| `MbarrierInit`* | `mbar`, `count`, `slot` | Initialize one mbarrier, or one slot of an mbarrier array, once in the kernel prologue. |
| `MbarrierArriveExpectTx`* | `mbar`, `bytes_`, `slot` | Announce the byte count of the TMA copy about to arrive on this mbarrier. |
| `MbarrierArrive`* | `mbar`, `slot` | Signal arrival without a byte count, used by consumer warps to mark a slot empty. |
| `MbarrierWait`* | `mbar`, `phase`, `slot` | Block until the mbarrier's parity flips for `phase`, so the copied data is visible. |
| `SetMaxNReg`* | `count`, `direction` | Shrink or grow the calling warp's register budget (sm_90 and newer). |

## Cross-thread combine

| Operation | Arguments | Description |
| --- | --- | --- |
| `WarpShuffle` | `state`, `state_b`, `combine_states`, `length`, `dtype` | Fold per-lane partial states over `length` lanes with a register-only XOR shuffle butterfly. |
| `TreeHalve` | `bufs`, `state`, `state_b`, `combine_states`, `length`, `tid_var`, `dtype`, `barrier_id`, `barrier_count`, `inner` | Reduce a power-of-two shared-memory tree of partial states and broadcast the result to every thread, for combines wider than a warp. |

## Tensor core: register fragments (`mma.sync`)

| Operation | Arguments | Description |
| --- | --- | --- |
| `RegFragment`* | `name`, `role`, `shape`, `dtype`, `count`, `nregs` | Declare a per-thread register array for the A, B or C operand of one atom cell; C starts at zero. |
| `LdmatrixLoad`* | `frag`, `src_buffer`, `src_index`, `role`, `ldm`, `swizzle`, `staged`, `gmem_guard`, `k_zero`, `b_trans`, `pair_frag`, `byte_slab`, `scale_buffer`, `scale_index`, `scale_ldm`, `fragment_layout`, `fragment_index` | Load one operand fragment into registers from shared memory, or straight from global memory, with guards and optional byte-slab dequantization. |
| `BlockScaleLoad`* | `frag`, `src_buffer`, `src_index`, `role`, `ldm` | Load one lane's block scales for the block-scaled fp4 `mma`. |
| `MmaSyncPtx`* | `c_frag`, `a_frag`, `b_frag`, `shape`, `ab_dtype`, `c_dtype`, `b_row_major`, `sfa_frag`, `sfb_frag` | Issue one `mma.sync` instruction computing `c = a · b + c` over the fragments. |
| `FragmentPromote`* | `dst`, `src` | Add a packed f16 accumulator into its f32 shadow fragment and zero the f16 one. |
| `FragmentApply`* | `out`, `op`, `args`, `kinds`, `in_place`, `layout`, `post` | Apply one elementwise operation to every element of a C fragment, with fragment, per-row or uniform arguments. |
| `FragmentRowReduce`* | `top`, `bot`, `frags`, `op`, `layout`, `dtype` | Reduce a warp's C fragments per row, producing the row pair `FragmentApply` broadcasts. |
| `FragmentMask`* | `frag`, `mask_when`, `col_base`, `row_base`, `fill`, `keep`, `keep_op`, `layout` | Replace fragment elements whose absolute coordinates satisfy a predicate with a finite fill value. |
| `FragmentBiasAdd`* | `frag`, `buf`, `index`, `col_base`, `row_base`, `layout` | Add a global-memory bias, such as an attention mask, to each fragment element at its absolute coordinates. |
| `FragmentRepack`* | `frag`, `srcs`, `ab_dtype`, `fragment_layout`, `part` | Convert C fragments into one 16-bit A fragment, so a result can feed the next contraction. |
| `RegStore`* | `dst_buffer`, `dst_index`, `frag`, `shape`, `ldm`, `epilogue`, `m_guard`, `n_guard`, `atomic`, `fragment_layout`, `volta_interleaved`, `fragment_index`, `swizzle`, `row_dim`, `col_dim` | Store a C fragment to the output buffer per lane, after an optional fused pointwise epilogue and a downconvert. |

`RegStore.epilogue` carries a `RegEpilogue`* (`acc`, `loads`, `ops`, `result`, `selects`, `extra_accs`): a pointwise
chain evaluated per fragment element, whose leaves are `EpilogueLoad`* (`name`, `buffer`, `index`, `roles`). Both are
payloads, not body statements. `FragLayout`* (`n_elems`, `elem_row`, `reduce_xors`, `lane_decl`, `lane_names`,
`row_off`, `col_off`) is the per-atom geometry the fragment operations read.

## Tensor core: warp groups (`wgmma`)

| Operation | Arguments | Description |
| --- | --- | --- |
| `WgmmaDescriptor`* | `name`, `smem`, `smem_index`, `swizzle`, `lbo_bytes`, `sbo_bytes` | Build the 64-bit shared-memory matrix descriptor a `wgmma` operand is read through. |
| `WgmmaMma`* | `c_frags`, `b_desc`, `shape`, `ab_dtype`, `a_desc`, `a_frag`, `scale_d`, `trans_a`, `trans_b` | Issue one asynchronous `wgmma.mma_async` cell across the four warps of a group into their C fragments. |
| `WgmmaFence`* | — | Order the group's earlier register and shared-memory accesses before the next cell. |
| `WgmmaCommit`* | — | Close the cells issued since the last commit into one group. |
| `WgmmaWait`* | `group` | Block until at most `group` committed groups remain in flight. |

## Expressions

Indices, predicates and loop bounds are expressions, not statements.

| Expression | Arguments | Description |
| --- | --- | --- |
| `Var` | `name` | Reference an axis variable or an SSA name. |
| `Literal` | `value`, `dtype` | A numeric constant. |
| `BinaryExpr` | `op`, `left`, `right` | An arithmetic, comparison or logical operation on two expressions. |
| `Builtin`* | `name` | A GPU built-in variable such as `threadIdx.x` or `blockIdx.x`. |
| `FuncCallExpr`* | `name`, `args` | A call to a registered elementwise function such as `exp` or `maximum`. |
| `TernaryExpr` | `cond`, `if_true`, `if_false` | Choose between two expressions by a predicate. |
| `CastExpr`* | `dtype`, `expr` | Convert an expression to another scalar type. |

## Audit notes

Findings from the audit that produced this table, as of `eb70ef72`.

- **No producer found for `Reassign`, `Pack`, `Unpack` and `Mma`.** No pass constructs them. `Pack`, `Unpack` and
  `Mma` are still renamed, type-stamped and serialized by the wire codec. `Reassign` is referenced only by its own
  definition. They look like dead code; confirm against stored goldens before removing them.
- **The Kernel IR table in `ir/ARCHITECTURE.md` is incomplete.** It omits the transport group (`CpAsync*`, `Tma*`,
  `Mbarrier*`, `SetMaxNReg`), `WarpShuffle`, `Reassign`, `FragmentBiasAdd`, `BlockScaleLoad` and the shared leaves
  `Init`, `Const`, `Pack`, `Unpack` and `ZeroPrologue`. It also describes `Sync` as only `__syncthreads()`.
- **The package docstrings are stale.** `ir/kernel/__init__.py` lists only `Smem`, `Sync` and `TreeHalve` as hardware
  primitives, and the `ir/kernel/ir.py` module docstring has a truncated sentence about shared leaf compute.
