# CPU Backend

Runs a graph on the host CPU with native code: Emmy's own passes up to the tile cut, then one LLVM-compiled kernel
per cut piece, launched on a thread pool. No GPU, no C++ compiler, no OpenMP.

```
graph ─ LOOP_PASSES ─ tile/lift ─ tile/cut (every seam pinned) ─ TileOp.loop_body → LoopOp
      ─ codegen.generate (Loop IR → LLVM IR) ─ kernel.CpuKernel (llvmlite JIT) ─ CpuBackend.run (topo walk)
```

## Why after the cut

Fusion is maximal by design: a whole layer is often one fused nest, which on a GPU the schedule later splits back
into kernels. Generated as written, that nest recomputes every producer inside its consumer's loops (a Qwen3 decode
layer: ~3e19 statement executions). The cut materializes each seam into its own kernel (~6e7 for the same layer).
`compile` pins `PLACE@<site>=cut` for every seam `cuttable_seams` offers, re-running until a round finds no new
seam, and turns each piece back into a plain `LoopOp` with `LoopOp(body=tile.loop_body, name=tile.name)` — the same
call the cut itself uses.

## Kernel ABI (`codegen.py`)

Every kernel is `part(bufs, lo, hi, partial, sizes)`: `bufs` is the array of buffer pointers in
`(*op.inputs, *op.outputs)` order, `[lo, hi)` is this thread's slice of the split axis, `sizes` holds the runtime
symbolic extents in `Plan.sizes` order. A kernel whose split axis is a reduction writes per-thread partials and also
exports `finish(bufs, partials, nchunks, sizes)`, which combines them and runs the epilogue.

`generate` tries a strategy chosen from the nest's shape, then falls back to the general lowering:

| Strategy               | Shape                                       | Split axis                          |
|------------------------|---------------------------------------------|-------------------------------------|
| `pointwise`            | perfect nest, no accumulators               | first loop of more than one trip    |
| `full reduction`       | one reduce loop, nothing outside            | the reduce loop (partials + finish) |
| `row reduction`        | free loops around a reduce loop             | first free loop of more than one    |
| `contraction`          | reduce loop whose loads walk a free axis    | outer free loop; that axis innermost|
| `contraction, split-K` | the same with every outer loop of one trip  | the reduce loop (partials + finish) |
| `multi-pass rows`      | outer loops around leaf-only passes         | first outer loop of more than one   |
| `serial`               | anything else (`Cond`, `Select`, siblings)  | outermost independent loop, if any  |

A construct neither path covers raises `Unsupported`; `compile` records it in `CpuProgram.fallbacks` and the piece
runs through `LoopOp.forward` (cppyy) instead.

## Numerics

- Buffers keep their storage dtype (f32, f16, bf16 as a `uint16` carrier); every value is widened to f32 on load
  and narrowed on store. An `Assign` with a 16-bit `dtype` rounds its result to that dtype.
- Only the accumulating instruction carries `reassoc nsz`, so the vectorizer can split a sum into lanes while
  every other operation keeps IEEE rounding.
- `exp` is a Cephes polynomial in plain arithmetic (max 0.98 ulp measured) so it vectorizes; `erf`, `tanh`, `sin`,
  `cos`, `log1p` call libm.

## Invariants

- Buffers bind by name. A cut piece's `LoopOp.inputs` order is re-seeded from its body and differs from the graph
  node's `inputs`.
- One target machine per kernel: the MCJIT engine owns and frees the one it is given.
- Kernels below `PARALLEL_MIN_COST` stay on the calling thread; waking the pool costs more than they do.

## Preliminary assumptions

- Cut every seam. No cost model chooses which seams to keep fused on a CPU.
- Kernels JIT in process with llvmlite; the Python thread pool launches them. A CPU executor in the Rust runtime
  (ahead-of-time objects, `dlopen`, persistent pool) is the intended follow-up.
- bf16 is not verified end to end; in a bf16 Qwen3 layer some pieces (loop-carried state) fall back.
