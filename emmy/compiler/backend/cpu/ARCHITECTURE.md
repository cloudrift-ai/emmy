# CPU Backend

Runs a graph on the host CPU with native code: Emmy's own passes through the tile cut and the schedule, then one
LLVM-compiled kernel per cut piece, run by the Rust runtime on its own thread pool. No GPU and no OpenMP; the kernel
library is linked with the system C compiler driver (`cc`).

```
graph ─ LOOP_PASSES ─ tile/lift ─ tile/cut (seams placed, below) ─ tile/schedule (the CPU family)
      ─ TileOp.loop_body + its CPU schedule ─ codegen.generate (Loop IR → LLVM IR)
      ─ one kernel library + a "cpu" plan ─ emmy_runtime.CpuExecutor
```

## The CPU context

`compile` runs every pass under `Context.for_cpu(threads)`: backend `cpu`, the CPU's product name and thread count as
its device (`hardware_id`), LLVM's name for the host CPU as its target (`arch`). The schedule pass offers the CPU
family on it, and measured rows are filed under it in the tune DB, apart from every CUDA card's: a kernel timed on 16
threads says nothing about one deployed on 6. The CUDA priors are not loaded on it, so a fork nothing measured takes
its first leaf.

## Seams

Fusion is maximal by design: a whole layer is often one fused nest, which on a GPU the schedule later splits back
into kernels. Generated as written, that nest recomputes every producer inside its consumer's loops. So `compile`
places every seam `cuttable_seams` offers itself, re-running until a round finds no new seam, without evidence:

- a seam whose cone reads constants alone — a quantized weight's decode — is fused, so the weight streams at its
  stored width and is decoded inside the projection that reads it;
- every other seam is cut, so each value is computed once.

## The CPU schedule (`ir/schedule/cpu.py`)

A schedule family beside the classic and register ones, with its own knobs; no CUDA knob takes a CPU meaning.
`CPU_TILE` (`<rows>x<columns>`) is a projection's register tile: the output cells one pass of its reduce loop
accumulates together, so every weight element loaded feeds that many rows and every input element that many columns.
A kernel with two free loops and a reduction offers each tile whose accumulators fit half the vector registers,
largest and squarest first. Pins, measured rows and `--strict-evidence` work as on CUDA; a hand pin outranks a
measured row, and a published pin a kernel cannot take leaves it its catalog.

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
| `tiled projection`     | two free loops around a reduce loop         | the wider free loop, in `CPU_TILE`s |
| `row reduction`        | free loops around a reduce loop             | first free loop of more than one    |
| `contraction`          | reduce loop whose loads walk a free axis    | outer free loop; that axis innermost|
| `contraction, split-K` | the same with every outer loop of one trip  | the reduce loop (partials + finish) |
| `serial`               | anything else (`Cond`, `Select`, siblings)  | outermost independent loop, if any  |

A symbolic loop (a decode step's rows, which may be 1) gives way to a static loop of at least `SPLIT_MIN` trips as
the split axis, so every thread streams its own slice of a weight at any row count. A construct no strategy covers
raises `Unsupported`, and `compile` fails naming every piece that raised.

A tiled projection over a weight stored as bytes runs its reduce loop 16 lanes at a time: each lane is a copy of the
body with the reduce index spelled `16 * block + lane` and simplified, so a block scale's index stops depending on
the lane and is decoded once a block, and the lanes read consecutive bytes. The block loop is marked for LLVM's
straight-line vectorizer, not its loop vectorizer, which packs the lanes into vectors.

## Numerics

- Buffers keep their storage dtype (f32, f16, bf16 as a `uint16` carrier, and one byte for `f8e4m3`, `f4e2m1x2` and
  `u8`); every value is widened to f32 on load and narrowed on store. A byte loads as its unsigned value, which the
  graph's own decode turns into numbers. An `Assign` with a 16-bit `dtype` rounds its result to that dtype. An
  `Affine` runs as its binary `Assign`s, as in Kernel IR, so each step rounds the same way.
- Only the accumulating instruction carries `reassoc nsz`, so the vectorizer can split a sum into lanes while every
  other operation keeps IEEE rounding. It and a product feeding it carry `contract`, so they fuse into one FMA, as
  nvcc fuses them.
- `exp` is a Cephes polynomial and `from_f8e4m3` integer arithmetic, so they vectorize; `erf`, `tanh`, `sin`,
  `cos`, `log1p` call libm.
- A lookup into the e2m1 pair table an NVFP4 weight decodes through — a constant the loader builds from
  `from_f4e2m1` of each byte's two nibbles, which the backend recognizes by that definition — is emitted as the
  arithmetic decode of the indexed nibble, never a load.

## Running (`runtime.py`)

`compile` builds the program for the Rust runtime: every kernel's IR is linked into one module, its functions renamed
`<kernel>_part`/`<kernel>_finish`, emitted as one object and linked into a shared library cached by content under
`~/.cache/emmy/cpu`, and the graph becomes an execution plan with `backend="cpu"` (plan format 5) that
`emmy_runtime.CpuExecutor` runs on its own thread pool. A compute node that is not a kernel, or a machine without the
runtime extension or a C linker, fails the compile with `Unsupported`.

The default thread count is one per physical performance core the process may run on: a second thread on an SMT
sibling holds every lockstep launch up. A reduction split across threads always uses the same number of chunks,
combined in order, so the result does not depend on the thread count. Independent work splits into more chunks than
threads, so efficiency cores help instead of holding a launch up.

`NativeProgram.alias_buffer` points a buffer at host memory the caller holds (`CpuExecutor.set_region`, as on CUDA),
read in place from the next run: the per-expert call of a serving path whose weights already live in host memory.
`time_launches` times each launch alone, after reading twice the host's last-level cache on every thread when the
cache is to be cold; `run --gpu-arch cpu --bench` reports it and files each kernel's row under the CPU's context.

## Invariants

- Buffers bind by name. A cut piece's `LoopOp.inputs` order is re-seeded from its body and differs from the graph
  node's `inputs`.
- Kernels below `PARALLEL_MIN_COST` stay on the calling thread; waking the pool costs more than they do.
- A thread's partials start on their own 128-byte line (`Plan.partial_floats` is padded), so no two threads finishing a
  reduction write one cache line.

## Limits

- Seams are placed by the rule above, not by CPU evidence, and there is no CPU prior: an unmeasured fused piece
  could otherwise win the first leaf and recompute far more than the layer needs.
- A piece the generator cannot express fails the compile; there is no interpreter fallback.
- Measured rows of a symbolic kernel are pooled across the sizes they ran at, as on CUDA: the fastest row at any size
  decides the kernel.
- A bf16 piece that carries loop state across iterations is not supported.
