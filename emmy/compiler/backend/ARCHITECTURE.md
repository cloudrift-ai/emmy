# Backend Architecture

Four backends with a shared interpreter. Every backend exposes the
same two-step API (`backend/base.py`):

```python
compiled = backend.compile(graph)
result   = backend.run(compiled, input_data={"x": np.ndarray(...)})
# RunResult(outputs: dict[name, ndarray], time_ms: float | None)
bench    = await backend.benchmark_async(compiled, warmup=, num_iters=)  # async-only; callers asyncio.run at the CLI boundary
# BenchmarkResult(time_ms, min_ms, max_ms, num_launches, per_launch)
```

What differs is `compile()` — how far the backend lowers the graph:

| Backend        | `compile(graph)` does                              | `run()` goes through |
|----------------|----------------------------------------------------|----------------------|
| `NumpyBackend` | returns the graph as-is (no-op)                    | default `Backend.run`|
| `LoopBackend`  | runs decomposition → optimization → fusion         | default `Backend.run`|
| `CpuBackend`   | fusion + `tile/cut` + LLVM codegen per cut piece   | the Rust runtime     |
| `CudaBackend`  | fusion + `lowering/kernel` + `lowering/cuda`       | the Rust runtime     |

`numpy` and `loop` backends share the same runtime path — the only
distinction is whether the graph has been fused yet. See
`backend/cuda/ARCHITECTURE.md` for the CUDA-specific dispatch.

## Torch reference (`torch_ref.py`)

Not a `Backend` — a small Graph→torch evaluator that runs a frontend-dialect graph through **real PyTorch**, the eager /
`torch.compile` baseline for `emmy run --ir`. Each frontend / tensor op is mapped to its torch twin
(`RmsNormOp`→`F.rms_norm`, `LayerNormOp`→`F.layer_norm`, `SdpaOp`→`F.scaled_dot_product_attention`,
`LinearOp`→`F.linear`, `ElementwiseOp`/`ReduceOp`/additive `ScanOp`→the torch elementwise/reduce/scan, layout
ops→view/transpose/cat). A two-axis transpose swaps those axes, including for rank-two inputs. Matrix multiplication
promotes its operands to include the declared output dtype, so an FP32 result from FP16 inputs is not rounded to
FP16 before widening. Slices resolve a negative start against the input extent before adding their output extent.
Single-source index maps with unchanged coordinates, broadcasts, permutations, diagonals, or constant-zero coordinates
use strided views. These preserve noncontiguous input storage and avoid unnecessary gather/clamp expressions that can
break Inductor fusion across a later slice. Other maps retain the clipped gather and source-selection semantics.
The legacy stored unary `pad` form is an identity; coordinate-changing padding is an index map.
`Conv1dOp` uses `torch.nn.functional.conv1d`, preserving groups, bias, stride, padding and dilation; `Conv2dOp` and
`ConvTranspose1dOp` use `conv2d` and `conv_transpose1d` with their captured geometry.
FP8 tensors remain exact `uint8` bit carriers; `to_f8*` casts to torch float8 and reinterprets its storage, while
`from_f8*` performs the inverse reinterpretation before widening. Generic integer casts, shifts, masks, and `RangeOp`
also keep their Torch integer dtypes, which lets loader-spelled reconstruction algebra serve as an exact pre-fusion
reference. A source-free `RangeOp` graph defaults to a CPU tensor; otherwise range construction and scalar-only
`IndexMapOp` broadcasts follow the supplied input tensors' device. `is_runnable(graph)` is `True` only when every compute
op and every elementwise operation name has a
mapping. `GatherOp` supports both elementwise selection and table lookup, including the NVFP4 pair table.
`ScatterOp` remains unsupported. Packed FP4 uses a uint8 carrier; FP4 encode rounds to nearest even with saturation
and preserves signed zero, while decode indexes the 16 representable values. This is a decomposed PyTorch reference,
not a native NVFP4 matmul implementation. `build_callable(graph, input_tensors)` returns a pure `fn(*tensors)` (scalar constants read inline)
so `torch.compile` can trace it. A single-output graph returns its tensor directly; a multi-output graph returns an
ordered tuple following `graph.outputs`, matching the backend result boundary. Symbolic
graphs work too: `build_callable` binds every symbolic axis name to its concrete extent read off the supplied tensors
(the CUDA launch convention) and bakes the env into the per-node callables — shape-resolving sites (`ReshapeOp` target
shape, `IndexMapOp` out-shape and coord/select exprs) eval through it, so a dynamic frontend provenance slice
gets the same vs-torch comparison as a static one (benched at the `Dim` hint by
`commands/run.py::bench_lowered_vs_torch`, which sizes its random inputs by hint-resolving symbolic dims). Used to
benchmark decoded golden programs and in-memory provenance slices against torch.

The strict run path evaluates Emmy and eager on the same inputs and returns a direct `rtol=atol=1e-3` proof with
error statistics; an elementwise failure is then weighed against an FP64 evaluation of the frontend graph
(`torch_ref.build_callable(..., compute_dtype=torch.float64)`). The isolated CUDA worker transports that proof, the
eager reference outputs and the FP64 outputs back to the parent, so exact-pinned rows are checked against the same
references rather than against another Emmy realization.

## Backend ABC (`base.py`)

`Backend` is an abstract base class with `compile`, `run`, `benchmark`.

`Backend.run` provides a default implementation: walks the compiled
graph in topological order and calls `node.op.forward(*args)` at each
compute node. `InputOp` and externally bound `ConstantOp` boundaries are seeded from `input_data`. A declared
`ConstantOp.value` is immutable: lowering may consume it, so changing that value requires recompilation.
Values are stored **per buffer** — a multi-output node's
`forward` returns a tuple matched positionally to `node.outputs`, and
each result is reshaped to its own output tensor's shape when statically
known.

Every op implements `forward(*inputs: np.ndarray) → np.ndarray`,
including `LoopOp`: its `forward` delegates to `ir/loop/runner.py`
(`execute_loop_op_cpp`), which renders the body to C++ and JIT-compiles
it in-process via cppyy / Cling. That's why the same default `run` works
for pre-fusion graphs (LoopOp absent) and post-fusion graphs (LoopOp is
just another `Op` subclass).
`NumpyBackend` and `LoopBackend` inherit it verbatim; `CudaBackend`
overrides with the runtime's dispatch.

The default `benchmark` does wall-time iterations around `run`; the
CUDA backend overrides it to populate per-launch CUDA-event timings.

Result dataclasses:

- `RunResult(outputs, time_ms)`
- `BenchmarkResult(time_ms, min_ms?, max_ms?, num_launches, per_launch?, captured, e2e_ms?, e2e_min_ms?)` —
  `time_ms`/`min_ms` sum per-launch solo windows; `e2e_ms`/`e2e_min_ms` (automatic for multi-launch programs
  under capture) time the whole program as replays of one all-launches CUDA graph — the only
  end-to-end-comparable number for multi-kernel programs.
- `LaunchTime(idx, kernel_name, time_ms)` — one per kernel per bench run.

## Numpy backend (`numpy/`)

Thinnest backend. `compile` returns the graph; `run` is inherited from
`Backend`. Used for correctness testing (no GPU required) and as the
ground truth the loop and CUDA backends are triangulated against.

Pointwise operations with a wider floating result widen their floating inputs before evaluation. Casting only the
result would already have rounded or overflowed a half-precision product that the graph declares as full width.

## Loop backend (`loop/`)

Runs the fusion pipeline to turn the graph into `Graph[LoopOp]`, then
executes via the inherited `Backend.run`. Since `LoopOp.forward` works,
this is the numpy backend with fusion in front.

Used as the second axis of triangulation: **loop vs numpy disagreement
implicates fusion; loop vs CUDA disagreement implicates codegen.**

## CPU backend (`cpu/`)

See `cpu/ARCHITECTURE.md`. Cuts the fused graph at every seam, compiles each piece's Loop IR to native code
through LLVM (llvmlite, in process), and runs the pieces on a thread pool. A third triangulation axis that runs
without a GPU: **cpu vs loop disagreement implicates the cut (which the loop backend never runs) or the CPU code
generator, not fusion.**

## CUDA backend (`cuda/`)

See `cuda/ARCHITECTURE.md`. Runs the full lowering chain, compiles every kernel with `nvcc`
into the cubin cache, and executes the program through the Rust runtime.
The `COLD_CACHE` input regime measures one replay per kernel after same-stream L2 eviction outside the event
window. It reports a sum of per-kernel times, with no whole-program timing; worker requests carry the regime
explicitly so a persistent worker cannot reuse a previous request's cache mode.

## Execution plan + pack (`plan.py`, `pack.py`)

`save_executable` bundles an existing pack with resolved constant/input bytes and its cubins for independent execution.
It accepts explicit contiguous storage bytes and publishes only complete new directories. The bytes of a buffer the
runtime's pager holds are kept once for every program that declares it, as its pages one after another beside how many
of its first pages are resident, each written from a function the caller passes so the export never holds every
buffer at once. The Rust runtime
supports static ordinary-pointer plans and rejects unsupported features; its contract and build instructions live in
[`crates/emmy-runtime/ARCHITECTURE.md`](../../../crates/emmy-runtime/ARCHITECTURE.md). `native.py` reuses benchmark
process supervision for the native worker and compares it with the existing Python dispatcher on identical artifacts
through `emmy run --pack DIR --bench --json PATH`. This experimental comparison does not replace the general run/tune
dispatcher. `splice` joins one plan into another under a prefix — the paging programs into a plan, native preparation's
programs into its step — sharing each buffer the target already holds under its mapped name and merging kernels by name
(`merge_kernels`: a different source under a taken name takes a suffix from its own).

`plan.py` defines the **execution plan** — the serializable runtime projection of a lowered `Graph[CudaOp]`:
buffer specs (one `BufferSpec` per BUFFER — a multi-output node mints one per output slot, each with its own
role via `graph.buffer_role`), scalar/runtime constants, the launch list (`LaunchSpec.writes` names every
buffer a launch produces; a scratch planner's first-write test reads it, falling back to `node_id` for plans
stored before the field existed), symbolic-axis plumbing, kernel refs (source and/or a content-addressed
cubin-cache key), and per-weight checkpoint bindings (`source_path` + a pack-own load-op vocabulary applied
with pure numpy). A kernel ref also records `arch_specific` — whether it must compile for the arch-SUFFIXED
target (`sm_120a`, `sm_90a`). Three unrelated instruction families need that suffix: TMA, which announces itself
through the launch's descriptors, and the block-scaled fp4 mma and Hopper's `wgmma`, which a plan can only recognize
by their wrapper names in
the rendered source. Plans stored under the older `uses_tma` key still read, since the meaning is the same and
only the name narrowed. A kernel ref also records `dependent_launch`, read the same way off the source: every kernel
the renderer emits waits on the grid ahead of it (`griddepcontrol.wait`, compiled in on sm_90 and later) before its
first memory access, and that wait is what lets the runtime launch it as a programmatic dependent launch. A plan
stored before the renderer emitted the wait carries no flag, and its kernels launch serialized.
A deterministic source-free bind record is the third binding kind: `plan_from_graph`
evaluates it once and its bytes ride the plan (`WeightSpec.generated`), so no checkpoint can supply it and
`build_from_plan` fills that buffer from the plan itself — a caller-supplied array still wins, but an
unsupplied constant buffer would otherwise allocate as ZEROS and run a silently weightless program.
A bind record computed from ONE checkpoint tensor (the constant fold turns a norm's `1 + weight` into one, at a width
where the broadcast to the token axis does not stop it) binds from that tensor: its `source_path` is the tensor's,
and its first load op, `("record", …)`, carries the record with that leaf renamed to a fixed slot, which
`apply_weight_loads` evaluates. Any other record keeps `load_ops=None`; the serving runner's binding raises on such
a weight, and on one with no source, instead of leaving its buffer zero.
`plan_from_graph` is the seam the whole runtime builds from: after it runs,
nothing reads the graph again — `CompiledProgram.build(graph)` is exactly `build_from_plan(plan_from_graph(g))`,
so a plan loaded from disk and a freshly compiled one share one launch path. The JSON form (`plan_to_dict` /
`plan_from_dict`) carries symbolic shapes and ceil-div grid factors through a tiny self-contained expression
grammar (`int` literal, `"name"` var, `[op, lhs, rhs]`), deliberately not the compiler's `Expr` classes — the
on-disk format survives compiler changes; only runtime-contract changes bump `PLAN_FORMAT_VERSION`.
CUDA-specific launch fields (TMA descriptors) nest under a `"cuda"` key so another backend can add its own
namespace and its own `build_from_plan` equivalent.
The declared output list is also the runtime return order; allocation planning may reorder buffers, but cannot reorder
observable program results.

`LaunchSpec.smem_bytes` records total per-block shared storage. At load time each executor subtracts the cubin's
static allocation and passes the remainder as dynamic shared memory, including requests below 48 KiB. Larger total
allocations opt into the device limit. This avoids omitting small dynamic buffers or reserving static storage twice.

`plan_cache.py` is the process-local reuse seam for repeated compiled structure within one immutable compile session.
It keys the exact graph wire form after loader spelling and ABI hints, erasing only external tensor addresses while
preserving their alias pattern and `source_parts` order. The stored `ExecutionPlan` template carries binding slots;
every lookup returns a fresh plan whose `WeightSpec`s contain that graph instance's real paths. Unknown compiler-minted
paths or unresolved slots fail closed, and only these instantiated plans may reach source loading or pack serialization.
This is deliberately not a persistent cache: `pack.py` owns cross-process environment/model validity.

**Paged buffers** (`ExecutionPlan.paged`, `name -> (axis, page)`): a buffer that is a table of equal-sized pages
rather than one allocation. The declaration enters as a graph hint (`cuda.paged_buffers`, `(name, axis, page)` per
buffer) the host sets before compile and the final kernel lowering reads, so shapes, schedules, goldens and cubin
keys of unpaged programs do not move. The kernel takes `T* const* <name>__pages` in place of the plain pointer, and
every read or write resolves its page from its index along the axis — the index divided by `page` — before its offset
inside one. The index is whatever the program spelled: an affine of the loop axes and symbols, a value loaded from
another buffer, or a mix, and nothing in the lowering knows what the buffer holds. From the same index the lowering
derives the pages the launch touches as a function of its inputs — an affine index gives an interval of pages, a
loaded index the pages its buffer's values name, a mix their union — and emits, per launch, the program that writes
that page list for the runtime's paging program (runtime `ARCHITECTURE.md`, "Paging"). An index computed inside the
kernel is recomputed in that program from the kernel's inputs by its backward slice, which may be the whole producer
run twice: a cost, never a cut or a fusion gate. A launch that touches one page of a buffer takes that page's base
once in the preamble and addresses it flat, so staged copies and fragment stores work over it unchanged. The schedule
search reads the same hint (`ClassicProblem.paged`): a transport that takes an operand's base address cannot feed a
paged one, so TMA, which bakes the address into its descriptor, never stages a paged operand, and cp.async never
stages one of several pages. The plan carries the declaration so the runtime knows the buffer has no slab: it is
never allocated, uploaded, zeroed or read back as one; its page table is bound by address. A weight can be paged
too, and the compiler binds it in a layout it picks — its folded transpose, its source layout — under another
buffer: the declaration names the weight as traced (`ConstantOp.name`), and `dim.paged_declarations` pages each
buffer bound from it as one page, the only paging every layout shares, so such a declaration must be one page. Its
bytes are then the host's to place: serving binds them as host arrays (`gen_runner._bind_plan_constants(paged=True)`)
and writes them into the page the program's table points at.

Paging rests on a scatter the lowering spells as a store into its destination's memory: a `Write` whose index is a value
the kernel loaded or computed, guarded so that an index past the end stores nothing, into an output stored over the
destination (`Tensor.alias`, `BufferSpec.alias`) — the way a step writes the KV cache at its position, and the way the
paging program keeps its tables and fills a slot. The `start` scalar that shifted an output's coordinate was the
scatter's index and is gone, as is the per-launch pointer-table operand the serving MoE dispatch once resolved
in-kernel: a gather by a selector is an index the program spells.

Where a page table entry points — a device slot, a resident page's home in device memory, or a page's home in mapped
host memory — is the pager's, and the compiler never sees it: residency is not part of a kernel's identity, its
schedule, the evidence context or a golden. A resident page enters the paging program only as a table entry: its
slot entry is -1, which the decisions read as a hit and whose age lands on the spare entry, so the programs and their
kernels are the same with or without resident pages.
The rows the evidence picks were measured with resident pages, and the schedules they pick also read host memory
correctly. The paging program copies every miss in before a launch reads it, and only a launch that touches more
pages than its buffer has slots reads host memory, so residency does not join the context. A buffer declared with
no slots (a fourth element of `0`) takes no inspector and no paging program, since nothing can move: its launches
read every page in place, through a page table the pager points at the pages' homes once.

The paging step is three programs — the decisions, the table update, and the copy over the step's list of moved
pages — traced once per page word with every size symbolic and read off the tables (`arange` over a table's length),
so one kernel serves every paged buffer of every geometry. Ranks are counted per element and inverted by a scatter,
never by a scan, so the decision kernels have no serial step. They are compiled at one fixed set of sizes (`paging.HINTS`), and the card's hardware
golden holds their rows at those sizes, so a serving compile picks the recorded kernels whatever its geometry. Each
launch's inspector is a slice of a kernel, with no program a record run could bench. The pager's kernels lower with
strict evidence off: a fork no row decides — an inspector's, or a kernel with nothing to re-bench alone — takes the
prior, under a scope that refuses that for the kernels they serve. A host whose launches run in an order it knows
writes their page list itself, and no inspector derives it: `paging.listed_paging` is the step alone, a plan whose one
input is the list, over buffers declared to the pager with their slots, while the programs that read the pages declare
them with none and read each page through its entry of the buffer's page table (serving `ARCHITECTURE.md`, the vision
tower).

`pack.py` bundles plans on disk: one directory per model × GPU × serving shape holding `manifest.json` (validity
key + environment tags + provenance + program index + each program's evidence) and `plan/<program>.json`. The
validity key is composed by the *runner*, and "model" there must cover everything the compiled programs read off the
CHECKPOINT — not just its architecture config. A compressed checkpoint is the case that makes the difference: two
rungs of one conversion share an architecture config and differ only in the per-tensor rates, which set the coded
extents, so the runner adds the loader's checkpoint digest (`loader.quant.checkpoint_quant_digest`) to the key. The
environment tags carry, beside the toolchain and the precision pins, the card and the tune DB's kernel key version.
Compiler version stays out; a pack keeps serving its frozen snapshot.

The golden rows are validity per program, since a re-recorded golden exists to change what a compile deploys. Each
entry keeps the exact identities of the kernels whose evidence its compile read (`policy.greedy.evidence_reads`;
the plan template cache hands a hit its template's reads) and a digest of the golden rows in scope that those reach
(`golden.rows_digests`). The manifest also keeps the card's scope digest (`golden.scope_digest`, file bytes only):
while it holds, nothing is compared. When it moved, `load_pack` computes the digests again over the files in scope;
an entry whose digest moved comes back with no plan, and so does every other twin of its wrapper (the plans binding
the same checkpoint tensors), since the first twin built settles the layout of each weight for the rest. When no
entry moved, the manifest takes the new scope digest. Cubins are **not** copied — plans
reference the shared `EMMY_CUBIN_CACHE` by content-addressed key, so packs dedupe kernels against each other and
the docker bake ships pack + cubin cache + model snapshot together. `load_pack` returns `None` on *any*
disqualifier (format/environment/key mismatch, unparsable plan, evicted cubin) and the caller falls back to the
full compile — a stale pack costs a recompile, never a wrong result. The serving boot integration
(`EMMY_PACK_DIR`) lives in `emmy/serving/runner.py`; see `emmy/serving/ARCHITECTURE.md`.

## Invariants

- The default `Backend.run` must work on any graph where every op has
  `forward`. It doesn't know about dialects — it just dispatches
  through `Op.forward`.
- A new backend (ROCm, SYCL, Metal) reuses `ir/` and
  `pipeline/passes/lowering/` wholesale; only its own executor
  (the equivalent of the Rust runtime behind `cuda/program.py`) needs to be written.
