# CPU Backend

Runs a graph on the host CPU with native code: Emmy's own passes up to the tile cut, then one LLVM-compiled kernel
per cut piece, launched on a thread pool. No GPU and no OpenMP; the Rust-runtime path links its kernel library with
the system C compiler driver (`cc`).

```
graph ─ LOOP_PASSES ─ tile/lift ─ tile/cut (every seam pinned) ─ TileOp.loop_body → LoopOp
      ─ codegen.generate (Loop IR → LLVM IR) ─ kernel.CpuKernel (llvmlite JIT) ─ CpuBackend.run (topo walk)
```

## Why after the cut

Fusion is maximal by design: a whole layer is often one fused nest, which on a GPU the schedule later splits back
into kernels. Generated as written, that nest recomputes every producer inside its consumer's loops, many orders of
magnitude more work than the layer needs. The cut materializes each seam into its own kernel, so each value is
computed once.
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
| `serial`               | anything else (`Cond`, `Select`, siblings)  | outermost independent loop, if any  |

A construct neither path covers raises `Unsupported`; `compile` records it in `CpuProgram.fallbacks` and the piece
runs through `LoopOp.forward` (cppyy) instead.

## Numerics

- Buffers keep their storage dtype (f32, f16, bf16 as a `uint16` carrier); every value is widened to f32 on load
  and narrowed on store. An `Assign` with a 16-bit `dtype` rounds its result to that dtype.
- Only the accumulating instruction carries `reassoc nsz`, so the vectorizer can split a sum into lanes while
  every other operation keeps IEEE rounding.
- `exp` is a Cephes polynomial in plain arithmetic so it vectorizes; `erf`, `tanh`, `sin`,
  `cos`, `log1p` call libm.

## Two ways to run (`backend.py`, `runtime.py`)

By default `compile` builds the program for the Rust runtime: every kernel's IR is linked into one module, its
functions renamed `<kernel>_part`/`<kernel>_finish`, emitted as one object and linked into a shared library cached by
content under `~/.cache/emmy/cpu`, and the graph becomes an execution plan with `backend="cpu"` (plan format 5) that
`emmy_runtime.CpuExecutor` runs. A program with any fallback piece or any compute node that is not a kernel, or a
machine without the runtime extension or a C linker, keeps the Python walk instead and says why in
`CpuProgram.native_reason`; `CpuBackend(native=False)` asks for it. The Python walk JIT-compiles each kernel in
process, runs its chunks on a Python thread pool (ctypes releases the interpreter lock), and runs a piece that fell
back through `LoopOp.forward`.

Both paths split a reduction across the same fixed number of chunks and combine them in order, so the result does
not depend on the thread count. The Rust runtime also splits independent work into more chunks than threads, so
efficiency cores help instead of holding a launch up. The default thread count is the performance cores.

## Invariants

- Buffers bind by name. A cut piece's `LoopOp.inputs` order is re-seeded from its body and differs from the graph
  node's `inputs`.
- One target machine per kernel: the MCJIT engine owns and frees the one it is given.
- Kernels below `PARALLEL_MIN_COST` stay on the calling thread; waking the pool costs more than they do.
- A thread's partials start on their own 128-byte line (`Plan.partial_floats` is padded), so no two threads finishing a
  reduction write one cache line.

## Limits

- Every seam is cut: no CPU evidence decides which seams would be cheaper left fused.
- A program with any fallback piece runs on the Python walk, not the Rust runtime.
- Matrix multiplies are generated like any other nest, with no register blocking and no vendor BLAS.
- A bf16 piece that carries loop state across iterations falls back.
