# A Kernel IR to PTX compiler in place of nvcc

Written 2026-11-12 against #1386 (`1989254ff`, on main `1000e1991`). Two PRs stacked on #1386:

1. **PR A, refactor.** Kernel IR becomes the whole kernel. What the CUDA text and the nvcc flags say today moves into
   Kernel IR, with the Kernel IR changes that make it a good compiler input. nvcc still builds every kernel, so each
   step is measured against an unchanged translator.
2. **PR B, compiler.** A Rust crate compiles Kernel IR to PTX. ptxas assembles it. The CUDA renderer for Kernel IR, its
   C++ helpers, nvcc and the nvcc flags are deleted.

This plan takes over what `normalization-debt-and-hackathon-optimizations.md` left "for the PTX emitter": address
reuse, division by a constant, unrolling and the precision instructions. Its lane-sharing item stays left out, for the
same reason (below).

## Status (2026-11-12)

PR A is draft #1395. Steps 1 to 10 are done, each checked on the corpus's rendered CUDA. Steps 3 to 10 changed the
text, the flags or the statements and ran the corpus at `correct` on the RTX 5090. Step 11 is done: no test checks a
kernel's structure on the rendered text any more (below). Step 12 has begun: two typing gaps the PTX compiler met are
closed and the finalization audit is applied (below). No step ran the perf lane, the whole suite or an sm_70 kernel
on a V100 yet; they are what is left of the closing step.

| Step | Commit | Corpus CUDA |
| --- | --- | --- |
| 1 | `60144bb83` | identical |
| 2 | `84e21ffa7` | identical |
| 3 | `5d8b1518a` | 58 of 296 kernels: conversions as definitions, literals at their dtype, one integer negation |
| 4 | `b8eaa7f01` | 72 kernels: the `lane` and `warp` declarations, dropped in 42 and moved into the grid in 30 |
| 5 | `c46c902cc` | 3 kernels: a `sigmoid` as four assignments |
| 6 | `f10f4fbdd` | 12 of the 42 fast-math kernels: the reciprocal (7), division (7), a `sigmoid`'s exponential (3) and the logarithm (2) as approximate operations; machine code identical in all 12 |
| 7 | `4aa5a4d73` | identical; the fast-math lane's flags are `--ftz=true` where they were `--use_fast_math`, and the machine code of all 42 fast-math kernels is identical to step 5's |
| 8 | `f3c19f0a9` | 118 kernels, the ones with a flagged loop: each such loop is copies of its body (one keeps rolling, 32 copies per trip), no statement loop carries a `#pragma`; 74 of the 118 build to the same instruction count, 104 to within 3% |
| 9 | `66729a66c` | 218 kernels: 9,320 lines whose index is respelled, 462 whose vector-store temporary is renamed after it; the same elements at 65,400 sampled launch points (3.4 million accesses) |
| 10 | `f68e11911` | identical |
| 12: epilogue | `685250cd1` | 8 kernels: a conversion's name moves its suffix (`v_e0__f32` to `v__f32_e0`), and in 2 a residual load of halves is its own definition ahead of its conversion; machine code identical in all 8 |
| 12: comparison | `95dae5c2b` | 30 kernels, 408 comparisons: `float x = cmp;` is `bool x__bool = cmp; float x = x__bool;`; machine code identical in all 30 |
| 12: audit | `463a197b6` | identical |

What step 6 found, checked with `nvcc -ptx` and `nvdisasm` (CUDA 13.0, `sm_75` to `sm_120`):

- `--use_fast_math` is `-ftz=1 -prec_div=0 -prec_sqrt=0 -fmad=1 -fast-math --gen_div_approx_ftz` to cicc. Each
  instruction it picks has an explicit spelling that gives the same PTX with the flag off: `__expf` (`mul`,
  `ex2.approx`), `__logf` (`lg2.approx`, `mul`), `__sinf`, `__cosf`, `__powf` (`lg2.approx`, `mul`, `ex2.approx`),
  `__fdividef` (`div.approx`), and inline PTX for `rcp.approx`, `sqrt.approx` and `tanh.approx`. The intrinsics take
  `.ftz` from `--ftz`; inline PTX does not, so the two helpers nvcc flushes say `.ftz` from the kernel's decision.
  `tanh.approx` has no flushing form and needs sm_75.
- `rsqrtf` is `rsqrt.approx` in every regime, so the reciprocal square root has no second form.
- `1.0f / x` under approximate division is `rcp.approx`, not `div.approx` with a one: the two differ by one
  instruction in machine code, so the reciprocal is an operation of its own.
- **`--prec-div=false` alone is `div.full`, not `div.approx`.** Only `--use_fast_math` gives `div.approx`. The
  operation is `div.approx` everywhere, so `FAST_DIV` pinned on outside the fast-math lane (with `FAST_MATH` off, or
  with `FAST_EXP` pinned off) changes instruction. No stored row is in either regime.
- A `sigmoid` written out by step 5 kept `expf` under `FAST_EXP` with `FAST_MATH` off; it takes `__expf` now, like
  every other exponential.
- A `sigmoid` that a `FragmentApply`, a combine or a store epilogue holds as one operation cannot take one
  approximate part: it becomes `sigmoid_fast` (both parts) when both knobs are on, and stays exact otherwise.
- The pass cannot stay where `085_fast_exp` was. An approximate instruction is an f32 instruction, and where an
  operation computes is known only after step 3's `120`; the composites are written out by `125`. It is `130`, the
  last kernel pass.
- A kernel whose output buffer is named `log` did not compile: the kernel renderer had no spelling for the logarithm
  and printed `log(`. It prints `logf(` now.

What step 7 found:

- The flag reached one function inside a CUDA header: `htanh` is `tanhf` on the converted value, so a half
  hyperbolic tangent was approximate under `--use_fast_math`. The pass picks it at a half too, as it does the
  exponential. `hexp`, `hlog`, `hsqrt`, `hrsqrt` and the half division are inline PTX and never followed a flag.
- Nothing else follows `-fast-math`: with the schedule held, 190 small traced kernels in 11 regimes build to the same
  machine code under the old flags and the new, the half tanh apart.
- **The fast-math lane's context key moves**, because the flag string is part of it. Three things follow. Its cubins
  build again under new keys. In an existing tune DB its rows stay filed under `--use_fast_math`, which no compile
  asks for: the golden rows import again under the new context on first use, and a row measured there locally is
  not picked from until it is measured again. And a kernel no evidence decides draws its schedule sample with a seed
  that folds the context key, so those picks move. The standard lane is untouched.
- The serving pack's key listed three precision decisions and left the rest to the flag list. It lists them all now:
  `FAST_DIV` and `FAST_SQRT` change the kernel body, not a flag.
- nvcc's defaults are precise division, precise square root and the accurate library, so the flag list says nothing
  for them. The standard lane's flag list stays empty, and its keys with it.

What step 8 found:

- **The flag was not what nvcc did.** cicc unrolled a 64-trip scalar reduce in full (the instruction count of those
  kernels is unchanged), but gave the 128-trip tensor-core K loop of `matmul/f16-warp-splitk` 32 copies per trip.
  Written out in full that kernel has four times the machine code (7,663 to 30,641 instructions) and takes nvcc
  57 s where it took 5.5 s. The pass copies what cicc did: one copy per trip up to 64 trips, and past that a loop
  that runs up to 32 copies per trip (7,299 instructions, 9.2 s). The site that flags K loops at any length is the
  only one past 64 trips. A partial factor as a knob stays left out, below.
- **Body optimization does not drop a decided guard.** It simplifies the condition and leaves `if (1)`; two corpus
  kernels held 33 of them before this step. The unroll drops the guards its own trip value decides.
- **A trip count can depend on the thread.** A lane loop over a row its lanes do not divide runs one trip more on the
  low lanes (8 corpus cases; a 3,840-wide norm at 512 lanes, whose test says the loop must unroll, is another). Such a
  trip keeps the loop's bound as the guard of its copy. No corpus loop is flagged with no static bound on its trips;
  one would stay a loop and lose the flag. The goldens were not surveyed.
- **A loop seeds the folds that stand in its body.** Copies unrolled into an outer loop bring folds it never seeded,
  and it would have declared them a second time: such a loop seeds its own folds as a statement ahead of itself.
- **The printer still decides on what it can prove.** A Volta gmem-direct fragment load takes the four-wide loader
  when its address is provably aligned; a literal K offset is, where the loop variable was not, so
  `matmul/volta-f16acc-direct` and three tests change loader. Nothing here ran it on a V100.
- **cicc unrolls on its own, and differently now.** Of the 14 kernels past 3%, four grow by 41% to 181%: each holds an
  unflagged chunk loop around the written-out inner loop, which cicc unrolled twice before and eight times now
  (`matmul/f16-matvec-reshaped-output-tma`: 64 `HMMA` to 256). Six shrink by 6% to 37% (decided guards, folded
  indices). All 118 are correct on the 5090; the perf lane has to say what moved.
- **`LOOPIFY` is in every stored row.** Its pass stamped `LOOPIFY: 0` on each kernel, so about 21,000 rows of the
  goldens and freezes carry it, and a row key no knob declares is a number to the priors' featurizer. The pass, the
  fragment-family and fused-tail fields and readable mode are gone; the knob stays a registered name nothing reads
  until those rows are rewritten. Eleven corpus cases lost the key or the pin.
- Pipeline time of the ten largest corpus cases (GPU-free lowering, best of three): 1.79 s to 2.33 s in total, the
  worst 0.16 s to 0.33 s. nvcc on the 118 changed sources: 38.1 s to 42.9 s in total.

What step 9 found:

- **The flattening is one function, and the pass is not a kernel pass.** Which buffers are paged is a fact of the
  graph that a kernel from the kernel cache does not know, so the pass is `lowering/cuda/008_flat_indices`, after the
  grid decode, and an access of a paged buffer keeps its dimensions: the page split is untouched.
- Body optimization does not inline a `Let`, so a staged copy's base stays a name and only its K stride, now a
  constant, folds into the loads.
- The Loop IR runner prints through the same statement renderers, so it flattens its body the same way before it
  renders.
- Machine code of the 218 changed kernels: 122 build to the same instruction count, 184 to within 3%, and the total
  falls from 91,469 instructions to 90,329. Two grow past 10% (`matmul/f16-volta-warp-mma-sm70` built for sm_75,
  +27%; `matmul/f16-mma-splitk-unit-output`, +13%). Pipeline time of the ten largest cases: 2.33 s to 2.68 s.

What step 10 found:

- **`LdmatrixLoad` was two statements.** It is the `ldmatrix` instruction now (a staged slab of 16-bit elements: the
  tile base, the row stride, the swizzle, and the role, transpose and pair that pick `x2`, `x4` and `.trans`), and
  `FragmentLoad` is every fragment gathered element by element. The staged drains that are no `ldmatrix` went with
  the gather, not into a third statement: a slab of bytes and every Volta slab are read by the same lane-to-element
  map as a read from global memory, through the same C++ loaders, and differ from it in the address space, the slab's
  stored layout and, for a slab with block scales, a decode. A staged gather carries no clamp and no zero-fill (the
  fill makes the slab exact), and the constructor refuses one that does.
- **No statement and no atom carries a register count.** The count follows from the layout, the role and the element
  dtype, in every case the three copies of the rule (the renderer's formula, the atoms' explicit lists, the paired
  contraction's register budget) handled: bytes pack four a register, and a packed pair of 4-bit values is one stored
  byte, so the fp4 A fragment is 4 registers by the same rule as the fp8 one. `RegFragment` lost `shape` and `nregs`
  and names its layout; a lane's register of block scales has a role of its own (`sfa`, `sfb`) where it was an A
  fragment of zero registers.
- **`FragmentRepack` fills a declared fragment**, as a load does: its `frag` is an update, not a definition. No
  kernel's statement order moved.
- **`setmaxnreg` needs the arch-suffixed target**: ptxas of CUDA 13.0 refuses it on `sm_90`, `sm_100` and `sm_120`.
  The lowering writes it only beside a TMA ring, which already asks for that target, so `needs_arch_isa` names it and
  no kernel changes.
- **The table's operand rows are read by nothing in PR A.** The renderer reads the rows of an accumulator of floats
  (`FragmentApply`, `FragmentRowReduce`, `RegStore`, their lane coordinates). The rows of A and B, of the packed
  accumulators and of the block scales are what the C++ loaders compute, transcribed from them; a crate test holds
  that the 32 lanes hold every element of the tile once (Volta's operands twice), and no card has run them.
  Volta's packed accumulator row is derived from the promote helper's shuffles: one row of eight columns a lane.
- **An index of a fragment statement stays one per dimension.** The swizzle of an `ldmatrix` address is split over
  the tile's row and column, a Volta slab layout is a function of the two, and a flat base would change the text.
  PR B's compiler flattens them with the shapes of the interface.
- **A row stride can be a launch-time value.** A `FragmentLoad` whose operand has a symbolic last extent, or whose
  address the lowering cannot read strides off, has `ldm = None`: the extent of the source's last dimension where the
  statement is compiled. The rule is stated once, on the field. `RegStore` keeps `ldm = 0` and `ldn = 0` for the same
  case, with `row_dim` and `col_dim`; it was not part of the split.

Where step 11 stands:

- **The count.** 31 test files hold 70 lines that read `kernel_source`, down from 181 in 59. None checks structure.
  37 check how the CUDA text spells something, in 17 files carrying 46 markers. 33 are neither: a `CudaOp` built by
  hand for the plan, launch and capture tests, the is-a-kernel filter, source determinism, the kernel cache, the
  rendered-line budget, a removed field's name, a stored table's column.
- **The helpers** are in `tests/compiler/helpers.py`. `compile_with_kernels` runs the compile the test would run and
  keeps, per kernel node, the `KernelOp` the last lowering pass read, recorded as the pipeline rebinds the node to its
  `CudaOp` (the `on_rebind` event). `recording_kernels` records the same for a compile the test does not run itself
  (a backend's, a serving split's). Nothing is stored on the op. `statements` lists a kernel's statements by type,
  `operations` the elementwise operations it computes, `text` its text form.
- **Marked**: every spelling read carries `# CUDA spelling: moves to the PTX compiler's tests`, so one grep lists what
  PR B's crate tests take over. What stays there: helper and intrinsic names, a conversion's spelling, the vector type
  a width spells, a run's 16-byte rows and shuffles, a lane formula, a slot's shift and mask, the paged buffers' page
  tables, the unroll pragma, a C declaration per brace scope.
- **Text checks that could not fail.** A search for a helper's name matches its definition in the prelude, which a
  kernel carries whether or not it calls the helper; the Volta prelude goes into every `m8n8k4` kernel whole. Found:
  the matmul coverage's "operands load gmem-direct", the attention coverage's three ("both the score and the
  expectation reach mma.sync", "the weight is a register repack", "the pivot is a per-row fold"), the knob pinning's
  "operands loaded gmem-direct", the FP8 staged "byte-gather drain", the Volta "one wide load" and its crosswise and
  lane-swap formulas, a Volta line no renderer writes any more, the twisted rewrite's f16 conversion (it accepted
  "acc" anywhere), the f32 accumulator's ("float v" passed), and the SwiGLU chunk's "each name is declared once" (its
  window held one declaration). Each is a statement check now, or dropped where the statement check already said it.
- **Found on the way**: `test_deferred_reduction_cuda_preserves_empty_states_and_the_uneven_tail[r4-]` gives a wrong
  answer on the RTX 5090 (5 of 8 values off). It reads no source; not chased.

What step 12 found and did so far:

- **Two typing gaps, both met by the PTX compiler.** A `RegStore` epilogue was typed by the renderer while it printed
  it, so the compiler re-implemented its rules (a float load widens to f32, an untyped value takes the promoted dtype,
  a float read converts). `120` types it now like any body over the f32 accumulator elements
  (`ir.types.epilogue_conversions`): a load that declares no dtype reads its buffer's type and a narrower float is
  widened by a conversion of its own, a select is f32. `check_types` reads the epilogue instead of exempting it, the
  renderer prints it, and `130` picks approximate forms on it as on the body. A comparison assigned at f32 left the
  truth value's conversion to the code generator; it is a `bool` definition now, converted once, and `check_types`
  refuses a comparison that defines anything else. Only `FragmentApply`'s operand conversion and the combine programs
  are still exempt.
- **The audit removed** what nothing reached: the cubin cache's clear helper, `address.reads_declared_rows`,
  `VOLTA_SMEM_LAYOUTS`, the renderer's self-recursive `_read_counts`, the fragment table binding's `supplies` field
  (no Python reader; the crate keeps it for PR B) and three attention-test helpers. Comments that still described the
  renderer promoting operands or re-rolled stores now say what the code does, and "register statement" is "fragment
  statement". No durable doc named `EMMY_READABLE`, `--no-readable`, `literal_constants` or a retired pass; a code
  comment and a crate doc named `literal_constants`, and are fixed.
- **Line balance** (`origin/main...`, after the audit): Python under `emmy/` +2,043 / −2,050, so 7 lines fewer; Rust
  +1,446 / −116. The growth is the crate's new checks and table, with their tests: `fragment.rs` +408 (the
  lane-to-element table), `types.rs` +336 (the type check), `input.rs` +201, the composite definitions and
  approximate operations in `elementwise.rs` +104. In Python it is `ir/types.py` +488 (the rules the renderer and
  `030` held, now one module), the unroll (`_unroll.py`, `117`), `130`, `125`, the interface and `008`;
  `100_loopify` −410, `render.py` −282, `030` −162 and `kernel/ir.py` −136 pay for it.
- **Plans**: `plans/` holds 11 files, two of them the owner's untracked ones. No tracked plan's work has clearly
  landed on main, so none was deleted.

## End state

- `emmy_ptx::compile` returns PTX text for one kernel. It reads the crate's own statements, so nothing is serialized on
  the way.
- `ptxas -arch=sm_NN -m64 k.ptx -o k.cubin` is the only external tool. It keeps register allocation and instruction
  scheduling. The runtime, the plan and the pack still hold cubins; what they are keyed on changes from CUDA text and
  nvcc's version to PTX text and ptxas's.
- The compiler makes no decision. A kernel's two whole-kernel precision decisions (multiply-add fusion, flush to zero)
  reach it as input from their knobs. Every other choice is in Kernel IR before it runs.
- Kernel builds get about ten times faster (measured below), and `make test` has one compile regime.

## What was measured

Dev box, CUDA 13.0, `sm_120`, 2026-11-12. Hand-written PTX of an 8-trip sum-of-squares loop, assembled by ptxas alone,
against nvcc's build of the same kernel in C. Instructions counted in the machine code ahead of the final divide.

| PTX given to ptxas | Instructions | What ptxas did |
| --- | --- | --- |
| nvcc's own PTX (unrolled, `[base+imm]`, `fma.rn`) | 19 | nothing left to do |
| Rolled loop, index in wrapping signed 32-bit, address rebuilt each trip | 49 | unrolled, merged the repeated load, fused multiply-add; **kept four address instructions per load** |
| Rolled loop, index widened to 64 bits first | 23 | unrolled and folded every address into `[base+imm]` |
| The same with `mul.rn` / `add.rn` | 29 | unrolled; **no fusion** (8 `FMUL`) |
| The same loop with 64 trips | 137 | fully unrolled (64 loads) |

Build time of two corpus kernels: `nvcc --cubin` 0.66 s and 0.16 s; ptxas alone on the same PTX 0.07 s and under
0.01 s. ptxas on nvcc's PTX gives a byte-identical cubin, so `nvcc -ptx` shows exactly what cicc does for a kernel.

So ptxas already unrolls small loops, merges repeated pure work in straight-line code and fuses un-suffixed float math.
It does not see through a wrapping 32-bit index, and an explicit rounding mode stops fusion. The hackathon entries
report the same with machine code (row 35), plus:

- ptxas unrolls a loop fully only when it counts from zero to a constant. With a thread-dependent start it unrolls
  four times (rows 35, 41, 8). A 64-trip loop with a 70-instruction body stayed rolled (row 20).
- ptxas decides its unrolling from the size of the PTX it is given, before it merges anything. Eight more PTX lines
  moved one kernel from 255 registers to 128 and made it 2.6x slower (row 35). Compact PTX is not cosmetic.
- In a rolled loop ptxas rebuilds every address from scratch, three instructions per load against one (row 35).
- Division or remainder by a constant is left as the general sequence, a power of two included (row 41; 262 `IABS` in
  one matmul, row 26). Repeated `rcp`, `div` and `sqrt` are not merged: each is a call (row 41).
- No vector access without an alignment proof. Invariant loads are hoisted by ptxas when registers allow; every entry
  that hoisted loads itself measured slower (rows 35, 26).

## What the hackathon compilers teach

57 entries compiled this same Kernel IR to PTX in Python. 17 were read in full, by sheet row; links under Sources.
Timings are the authors' own, on different runners, so they are not compared across entries.

- **The entries level with nvcc got there at lowering time, not with an optimizer.** Affine addresses with the constant
  in the address, the unrolled copy's axis bound to a constant, unsigned division from a non-negative fact, alignment
  facts for vector access: rows 26 and 20 did this in 1.8k and 2.5k lines. Row 54 spent 3.6k of its 4.6k lines on a
  control-flow graph, 7 analyses and 18 passes for the same gains; its authors note ptxas already does most of them.
- **Typed instruction objects in a flat list are enough.** Row 20's shape: instruction, label, structured `if` and
  `for` in the builder, one register per IR name, states updated in place, so no phi nodes. Row 26 kept text lines and
  had to run its two later passes with regular expressions.
- **Facts ride on the value.** Range, non-negative, known multiple and alignment are set when a register is made (rows
  26, 38, 41) and drive every later choice. Emmy's crate already computes ranges; the compiler reads them.
- **LLVM is not worth it** (row 11). Tensor-core, async-copy, barrier and shuffle instructions became opaque inline
  assembly, one loop was hand-written PTX anyway, three text rewrites were still needed, and it cost a pinned LLVM, a
  global lock and 5 to 270 ms per kernel.
- **The entries far ahead of nvcc changed the schedule, not the code.** Rows 38, 52 and 39 share work across lanes,
  re-tile attention and fuse sibling loops. Row 38 with every such rewrite off is level with nvcc. That work belongs
  where Emmy picks schedules, above Kernel IR. It is out of this plan.
- **Things that looked right and measured slower**: pointer-bump addressing (row 26), fusing address math into one wide
  multiply-add (row 54, 3.8x), wide shared-memory loads (rows 38, 56), merging global loads (row 26; row 38 saw a
  gain), full unrolling of a large attention loop (rows 20, 26: spills, and 5.5 s of ptxas against 54 ms). None is
  built here without a perf-lane number.
- **ptxas details that cost a day each**: without `.maxntid` it reserves registers for 1,024 threads (rows 56, 38); on
  `sm_120` an external shared symbol has a nonzero base that `cp.async` must add (row 52); `div.rn.f32` is a subroutine
  call; `tanh.approx` is 125 ulp off (row 20); a register read undefined gave a right answer at `-O3` and a wrong one
  at `-O0` (row 56); non-ASCII bytes are rejected even in comments (rows 45, 8).
- **What caught real bugs**: the differential against nvcc's build of the same kernel (row 37: five cache bugs),
  assembling every instruction form through ptxas (rows 34, 37), random kernels against an independent IR interpreter
  (row 38), a thread interpreter that steps lanes one at a time (row 54: one race), a float64 torch reference (row 18,
  which found Emmy bugs). Checkers over PTX text caught nothing on 100 problems (row 37).

## Where each optimization lives

| Level | Rule | Holds |
| --- | --- | --- |
| The `emmy-ir` crate: body normalization for Loop IR, and body optimization, its optimizing half, for Kernel IR (every body, every target) | Exact, always removes work, decides nothing. No knob. | Simplification under ranges, each value once at the shallowest scope, congruent states merged, unread values dropped, a grid's launch ranges. Unchanged by this plan. |
| Kernel IR passes (most GPUs) | Structure, and every choice a knob owns. | Grid decode, staging, vector access with its alignment proof, load order. New in PR A: unrolling, the pick of each approximate math operation, composite functions spelled out, flat indices. |
| Kernel IR to PTX (NVIDIA only) | A translation. Exact, no knob of its own, no reordering of float math, no moved load. | Instruction selection, register banks, integer width and signedness, address forming, division by a constant, rounding modifiers and fusion as the kernel's decisions say, the math sequences, tensor-core fragment maps, cache modifiers, the module's directives and shared declarations. |

Work sharing across lanes and threads sits above all three, with the schedule.

## The Kernel IR the compiler reads

Today a kernel is Kernel IR plus what only the CUDA text says. Read in main:

- **Types.** `030_stamp_types` stamps most values; the renderer decides the rest (promotion, which half-precision
  operations run natively, the destination type of a write, literal kinds) and C++ templates convert inside helpers.
- **The interface.** Parameter order and kinds, the shared-memory layout, the thread bound and the lane variable are
  worked out while printing. The plan learns that a kernel needs the arch-specific ISA by searching the source text for
  a helper's name.
- **Choices.** `#pragma unroll`, `--use_fast_math`, `--fmad`, `--prec-div`, `--prec-sqrt` and `-Xcicc -O1` are
  requests to nvcc. cicc at `-O3` also decides inlining, fusion and motion with no request at all.
- **Logic.** About 1,200 lines of C++ helpers hold the fragment loads and stores, packing, format decoders, swizzles
  and descriptors. `LdmatrixLoad` renders as a call to one of some thirty helpers, picked from seven of its fields.
- **Addresses.** The renderer flattens a multi-dimension index with the buffer's shape and simplifies it with no
  ranges in scope, after the body was optimized. Rendered today: `(128 * a0_b + 64 * a0_u) * 132 + a2`.

The recommended Kernel IR leaves the compiler nothing to work out. A crate function checks five properties on every
kernel in `lowering/cuda`, after the grid is decoded and before anything is rendered or compiled:

1. **Typed.** Every value carries its dtype, every conversion is a cast in the body, every literal has its kind.
2. **Flat.** A `Load`, `Write` or `CpAsyncCopy` holds one element index into its buffer.
3. **No requests.** No statement carries a flag that asks for a transform. A loop in the body is a loop in the code.
4. **Three kinds of statement.** Target-neutral (load, store, assign, loop, condition, barrier); or exactly one
   machine instruction with its operands (`mma.sync`, `ldmatrix`, `cp.async`, `wgmma`, a shuffle); or a fragment
   statement, whose instructions follow from its fields and the one lane-to-element table in the crate.
5. **Nothing implicit.** The lane index, shared-memory offsets and the parameter list are in the kernel, not in the
   printer.

## PR A: Kernel IR becomes the whole kernel

Branch `feature/kernel-ir-complete` on `feature/kernel-ir-decode`. nvcc still builds everything. Each step is checked
three ways: the corpus's rendered CUDA before and after (identical, or changed only as the step says), the corpus at
`correct` on a card, and the perf lane. No step moves a kernel identity, since the kernel key is computed from Loop IR:
no restamp is expected. A row whose time moves is measured again in the closing step.

1. **The interface is computed once.** One function gives a kernel's parameters (name, kind, dtype, shape), shared
   regions with offsets, thread bound, and whether it needs the arch-specific ISA, read from its statements. The
   lowering to `CudaOp`, the renderer and the plan read it. A computed fact, not a stored one. Two dead paths go: the
   plan's search for the grid-ahead wait (every kernel has it), and literal inputs (`literal_constants`, which no
   production caller passes). *Check: CUDA text identical on the corpus.*
2. **A target description.** The vector widths and the native half-precision operation set leave `CudaRenderTarget`
   for a small description the vector passes (`050`, `080`), the type pass and the Loop IR runner's target read. PR B
   deletes the class; its facts must not go with it. *Check: text identical.*
3. **Types complete** (property 1). The renderer's promotion and conversion rules move into one module,
   `ir/types.py`. `030_stamp_types` stays early, because the passes after it read the stamps and match one-operation
   values. A last pass, `120_explicit_conversions`, writes the conversions: one conversion is one definition, since
   body optimization keeps one operation per definition. Half-precision math that runs in f32 says so, a write knows
   both types. Add the crate's type check. The renderer stops inferring. The check exempts two kinds of statement.
   The cross-thread combines (`TreeHalve`, `WarpShuffle`, `WarpBroadcast`): PR B's compiler types their combine
   programs itself, from the stamped states. The register fragment statements: step 10 covers them, and step 12
   types the `RegStore` epilogue.
   *Check: CUDA text changes only by explicit casts; `test_dtype_cuda`.*
4. **Lane and warp in the grid** (property 5). `007_decode_grid` binds each one the body reads, a `Let` over the
   thread index at the top of the `Grid`, so they take ranges and are shared like any value. `TreeHalve` gets a
   `lane` field for the register fold of a per-warp slab, which read `lane` from rendered text only. The fragment
   statements and the prefix scan keep their own lane math in rendered text until step 10. *Check: text.*
5. **Composite functions spelled out.** The definitions of `sigmoid`, `relu`, `square` and `reciprocal` in primitive
   functions move into the crate, beside the operations' declarations. Only these four reach a kernel: the frontend
   decomposes `silu`, `softplus`, `gelu` and `gelu_tanh` before Loop IR. A kernel pass, `125_spell_composites`,
   applies the table to every `Assign`, so the optimizer shares an exponential between two of them. It runs after
   `120`: where a composite computes is decided on the whole function (an f16 `sigmoid` computes in f32). A
   `FragmentApply`, a combine program and a `RegStore` epilogue hold one operation and keep it; the renderer prints
   it through the table, and PR B's compiler applies the same table there. *Check: text; cicc already shares these,
   so no time should move.*
6. **Approximate math as operations.** `085_fast_exp` grows into the one pass that picks, under the precision knobs,
   the approximate form of each instruction class: exponential, division and reciprocal, square root, and the
   functions `--use_fast_math` covers today (log, sine, cosine, hyperbolic tangent, power). Each form is its own
   elementwise operation, as `exp_fast` and `divide_rn_f32` already are. It runs last, as `130_approximate_math`.
   *Check: perf lane level.*
7. **nvcc flags shrink to the two whole-kernel decisions**, as its own commit after step 6: `--use_fast_math`,
   `--prec-div` and `--prec-sqrt` go; `--fmad` and `--ftz` stay until PR B hands them to the compiler. The machine
   code of a kernel with a recorded row does not change, so no timing should move; the perf-lane run of the closing
   step checks that. What does move is the fast-math lane's context key (Status).
8. **Unrolling in Kernel IR** (property 3). A kernel pass, `117_unroll_loops`, copies the body of every loop flagged
   `unroll` whose trips a static number bounds, binding the axis to its value in each copy and dropping the guards
   that value decides; body optimization then folds each copy's index. It runs before `120`: a copy reads its axis as
   a literal, and a literal is typed by the operation that reads it. Up to 64 trips a loop is written out in full,
   past that it keeps rolling over 32 copies per trip, as cicc did (Status). `047_reuse_lane_loads` unrolls through
   the same function. The atom lowering and the lane loops of `_factor` set the flag, so no Loop IR statement
   changes. The `#pragma` leaves statement loops; `FragmentApply` keeps the one inside its rendered text until step
   10. What existed only to re-roll code into pragma loops is gone: `100_loopify`, `RegFragment.count`,
   `FragmentApply.post`, `EMMY_READABLE` and `--no-readable`; the `LOOPIFY` name stays registered until the stored
   rows that carry it are rewritten (Status). The `UNROLL` budget caps which loops a site flags. Unflagged loops stay
   rolled. *Check: perf lane, pipeline time per kernel, nvcc time per kernel.*
9. **Flat indices** (property 2). `lowering/cuda/008_flat_indices`, after the grid decode, turns each `Load`, `Write`
   and `CpAsyncCopy` index into one flat index (`address.flat_index`), carrying the 64-bit widening for buffers past
   the 32-bit range. Body optimization then spells it affine under the ranges in scope, the launch's included. The
   accesses their dimensions define keep them: one of a paged buffer (a graph fact the cached kernel does not know,
   so the pass stands in `lowering/cuda` and the page and offset split stays as it was), a `Write` under a Volta slab
   layout, the shared-memory side of a `CpAsyncCopy` with a lane offset, the fragment statements (`LdmatrixLoad`,
   `BlockScaleLoad`, a `FragmentApply`'s gathered operand, `RegStore` and the loads of its epilogue) and `TmaLoad`.
   The crate's `check_input` holds properties 1 to 3. *Check: text shows simplified indices; perf lane.*
10. **Fragment statements say what they are** (property 4). No expansion moves; the statements get the shape the
    compiler needs. `LdmatrixLoad` is the instruction on a staged slab of 16-bit elements, and `FragmentLoad` a
    fragment gathered element by element (from global memory, a slab of bytes, a Volta slab), whose role, transpose,
    clamp, zero-fill, row stride and two element dtypes are plain fields; `120_explicit_conversions` stamps the
    dtypes. The lane-to-element table is the crate's `fragment` module: per layout, role and element dtype, the lane
    coordinates and each element's row and column offset as expressions over them, the row-reduce masks, and the
    lanes that supply a register of block scales under the product's two selectors. It counts a fragment's registers,
    so `RegFragment` names its layout and carries no count. The Python renderers read the accumulator rows through
    the binding until PR B deletes them, and `check_input` holds that every fragment statement names a row.
    *Check: text identical.* What stays in the renderer for PR B's fragment expansion to take over:
    - the lane coordinates each fragment statement declares in its own text (`_g`, `_t`, `_vl`, …), now printed from
      the table's coordinates over `threadIdx.x & 31`, which the compiler binds once per kernel over the lane of
      step 4; and the prefix `WarpShuffle`'s `threadIdx.x % length`;
    - `FragmentApply`'s own `#pragma unroll` loop over the elements, and the conversion of its operands to f32;
    - the thirty C++ loaders a `FragmentLoad` picks among, with the choice of the four-wide Volta loader on an
      address the printer proves aligned, and `BlockScaleLoad`'s, `FragmentPromote`'s and `FragmentRepack`'s helpers;
    - `RegStore`: the packed pair stores of the m16n8 layout spelled as text, its typed epilogue printed once per
      element, and `ldm = 0` / `ldn = 0` for a stride the buffer's shape gives;
    - one index per dimension on every fragment statement.
11. **Tests read Kernel IR.** 59 test files hold 181 reads of the rendered source. A check of structure becomes a
    check of Kernel IR statements now (`compile_with_kernels`). A check of an instruction's spelling stays on the CUDA
    text, marked, until PR B replaces it with a crate test. Status has the count and what is converted.
12. **Closing.** The realization corpus and the perf lane on every held card, `make test` on a rental, then the
    `refresh-golden` skill for rows whose time moved, and both priors refit if a hardware golden changed.

Steps 1, 2, 4, 10 and 11 are independent. 3 needs 2. 5 and 6 need 3. 7 needs 6. 8 and 9 come last, in that order.

## PR B: the compiler

Branch `feature/ptx-compiler` on PR A. A new crate `emmy-ptx` beside `emmy-ir`, linked into the existing binding so a
body crosses as the crate's own value.

### Design

- **Input.** The body (one `Grid`), the interface of step A1, a target (`sm` version, arch-specific or not) and the
  kernel's fusion and flush-to-zero decisions. A body that fails the input check, or a statement with no rule for the
  target, is an error that names the statement. Nothing is skipped silently.
- **Instructions.** A flat list of typed instructions and labels. An instruction is an opcode, typed modifiers, results,
  operands and an optional predicate guard. An operand is a register (bank and number), an immediate, an address (base
  and offset) or a register tuple. Banks follow PTX: predicate, 16, 32 and 64-bit, f32, f64; signedness is on the
  instruction. No strings until printing.
- **Names.** One register per Kernel IR name. An accumulator or carried state is a register updated in place. Scopes
  are the body's own nesting, so there are no phi nodes and no dominance analysis.
- **The builder.** Loops and conditions are opened and closed through it; only it writes labels and branches. It keeps
  a value table per open scope, so a value the compiler makes twice is made once. The table is needed even where ptxas
  would merge the copies: the size of the PTX steers ptxas's own unrolling. It never moves a Kernel IR value or a load.
- **Loops.** A loop whose trip count is the same for every thread is written as a counter from zero to a constant, with
  the loop variable computed from it. That is the shape ptxas unrolls.
- **Facts.** Ranges come from the crate's range analysis under the grid's launch ranges, the same call body
  optimization makes. The builder adds the known multiple and alignment of each register.
- **Integers.** An index is computed in 32 bits where its range fits and widened once. Division and remainder by a
  constant become a shift, a mask or a multiply-high where the range proves the operand non-negative (`reduce_strength`
  moves here from the renderer), else the floor-correct signed form.
- **Addresses.** A flat index is split into its variable part and its constant. The variable part is scaled, widened
  and added to the base; the constant goes into the address. That is the form the measurement shows ptxas folds.
  Shared-memory addresses stay 32-bit. No pointer bumping and no fused wide multiply-add: both measured slower.
- **Floats.** Every float instruction carries its rounding mode, and `.ftz` when the kernel's decision says. ptxas then
  fuses nothing on its own. When the kernel's fusion decision is on, a multiply read once, by an add, a subtract or an
  accumulate in its own scope, becomes `fma.rn`. The rule is checked against cicc's PTX on the corpus.
- **Fragments.** One expansion per fragment statement, driven by the crate's lane table: per register and element, the
  address, the guard, the load or store, the cast and the pack. It replaces the thirty helpers. Lane math goes through
  the value table, so it is computed once per kernel. Loads stay where their statement stands.
- **Math.** Sequences for the primitive functions the corpus and the goldens use, counted first; any other function is
  an error until it is written. Target for each: no worse than the bound CUDA's documentation states for the same
  function. The approximate operations of step A6 map to the hardware instructions, with the subnormal fix of the
  earlier plan for the exponential.
- **One-instruction statements.** A table from statement to instruction form per target: `mma.sync` shapes, `ldmatrix`,
  `cp.async`, `wgmma`, the bulk tensor copy, `mbarrier`, shuffles, barriers, packed conversions, atomics. The PTX each
  needs is already in the C++ helpers' inline assembly; the table takes it over.
- **Module.** `.version` and `.target` from the target, `.maxntid` from the thread bound, the parameter list, shared
  arrays or the external pool with its base added, and the wait on the grid ahead. Each Kernel IR statement is kept as
  an ASCII comment above its instructions, so a kernel can still be read.
- **Verifier**, on every compile, for the two things ptxas does not check: every register is defined on every path to
  its use (a guarded definition counts as partial), and an alignment fact stands behind every vector access.

### Milestones

Until the last milestone a kernel holding a statement with no rule yet is rendered and built the old way, and a report
lists which corpus cases still do. That fallback exists only inside the PR.

1. **Scalar kernels.** The instruction list, builder, printer, verifier, addresses and integers; rules for `Grid`,
   loops, conditions, loads, stores, assigns, selects, states and casts; the ptxas wrapper beside the nvcc one; the
   differential harness. *Done when the pointwise and scalar reduction families pass on the 5090 and the perf lane is
   level.*
2. **Cooperative kernels.** Shared memory, barriers, shuffles, tree reductions, atomics, vector access, `cp.async`.
   *Softmax, cooperative reductions and scalar attention.*
3. **Math.** The sequences, each with a sweep of all f32 inputs on a card against a float64 host reference.
4. **Tensor cores.** The fragment expansion and `mma.sync` with `ldmatrix` on `sm_80` and later, then fp8, then fp4 on
   `sm_120a`, then Volta.
5. **Hopper.** `wgmma`, the bulk tensor copy, `mbarrier`, the register limit statement.
6. **Speed.** The perf lane on every held card against PR A. A slower case is compared with cicc's PTX for the same
   kernel (`nvcc -ptx`), then explained or fixed. First thing to try for a rolled loop: placing a compiler-made
   address part in the outermost scope that defines its operands, which the builder does not do until a case needs it.
   *The unrolling nvcc did on its own* (`118_unroll_as_nvcc`, `119_vectorize_copies`, the loop field `unrolled`).
   Most of what stayed slower after the first speed commits was loops cicc unrolled without being asked. Its rule,
   read off `nvcc -ptx -arch=sm_120` (CUDA 13.0) across trip counts and body sizes, with the body sized in Kernel IR
   as the work of one trip (one per access, fold, selection and branch, one more for a changing address, one per
   operation whose value changes, a math function by the instructions cicc writes: `exp` 14, `log` 28, `tanh` 17,
   `sin` 88, `pow` 113, the fast forms 1 to 3):
   - A static counted loop of `N` trips is written out when `N·(size + 2) ≤ 480` (96 trips of a two-statement body
     are, 97 are not), else steps the largest power of two `C` dividing `N` with `C·(size + 2) ≤ 199`. It carries no
     `.pragma`. An odd `N` gets four copies with the exit test between the third and the fourth (not modelled).
   - A loop over a runtime bound takes 16, 8, 4 or 2 copies by `C·(size + 7) ≤ 228`, none past a size of 89. The
     remainder is blocks of `C/2`, `C/4`, … copies, each under its bit of the count left, and with 16 copies the
     last three trips a loop of one copy; the stepping loop and that loop carry `.pragma "nounroll"`, nothing else
     does. A thread-start loop over a symbolic extent gets the same; one over a static extent cicc left to ptxas
     below 24 trips (and copied 16 times from 24 on, not modelled).
   - A loop with a barrier is not unrolled at run time; the tensor-core loops over a symbolic K stayed rolled.
   - Checked on the corpus: the symbolic reductions (16 copies, blocks 8 and 4, loop), their ILP forms (4, then 2
     and 1), softmax's normalize (8; the online pass 4), the f32 matmuls' 128-trip `_ki` loop over shared slabs (4,
     the four A reads one `ld.shared.v4`), `qwen3emb/sdpa-s32`'s inner loops (16) all come out as cicc's.

   The rule is a fixed copy of nvcc, like step A8's two limits; the number of copies moves under the `UNROLL` knob
   with them. `119` runs the vector widenings again over the copies (cicc merged them after unrolling); a stepping
   loop states its step, so the alignment proof holds. Every loop the unroll passes leave stepping several trips says
   `unrolled`, and the PTX compiler marks it `.pragma "nounroll"`: unmarked, ptxas unrolled the compact static loops
   in full where it had left cicc's alone (`qwen3emb/sdpa-s32`: 11,344 machine instructions and 0.53 s of ptxas,
   2,216 and 0.06 s marked, 180 to 82 us). The CUDA renderer writes `#pragma unroll 1` on the runtime-bound ones
   only, which is what cicc marked: marking the static ones too cost nvcc's build up to 1.6x where ptxas used to
   finish cicc's unrolling.

   Measured on the rented RTX 5090, two rounds per generator taking turns, against the lanes before it: through
   nvcc, median 1.001, 8 of 101 cases past 5 us more than 5% slower and 28 faster, total 20.8 to 13.4 ms; through
   PTX, median 1.000, 10 slower and 30 faster, total 18.5 to 13.5 ms (the symbolic reductions 0.59, the f32
   projections 0.40 to 0.46, `qwen3emb/sdpa-s32` 0.52). Between the two generators: median 1.015, 32 slower and 16
   faster. What is left:
   - The f32 matmuls over shared slabs got slower through nvcc too (1.10 to 1.32x, `qwen3emb/matmul-*`): cicc's
     four copies were software-pipelined by ptxas when the loop counted `ne` to a constant, and it pipelines neither
     cicc's stepping loop nor this compiler's. Through PTX they run at nvcc's new speed, `matmul-qproj-s512` 1.33x.
   - A static loop kept rolled runs slower than cicc's kept loop of the same copies (the serial reductions 1.5 to
     1.6x, `norm-weight-linear-f16-computed-b` 1.6x, `computed-a-nested-contraction-scalar` 1.4x): ptxas issues the
     last load of a trip into the registers of the first, still being read. Unmarked, ptxas unrolls those in full
     and they run at nvcc's speed, but the attention kernels then grow 5x (`sdpa-s32` 2.2x, `sdpa-s512` 1.36x slower).
   - The softmax kernels (1.1 to 1.2x) take cicc's copies now; the rest of the gap is the exponential's sequence and
     the division's subroutine.

7. **The switch.** The fallback goes, and with it:
   - the Kernel IR renderers, the C++ helpers, `CudaRenderTarget`, the nvcc wrapper. The Loop IR runner keeps its C
     renderer for the shared statements.
   - **Source text as a key.** `KernelSpec.source`, the cubin cache key (text, toolkit tag from `nvcc --version`,
     flags), `CudaOp.identity_key`, the kernel merge that renames by text replacement, the pack's `toolkit` and
     `nvcc_flags` fields, `test_source_determinism`.
   - **The opt level.** `-Xcicc -O1` is gone, so `Context.compile_flags`, `split_opt_level`, the `H_opt` prior feature,
     the tune DB's `opt` and `flags` context columns and the freeze's deployable-opt gate go too. That is a featurizer
     version bump, a tune DB version bump and a refit of both priors.
   - **The nvcc surface.** `EMMY_NVCC_FLAGS`, `EMMY_NO_NVCC`, `--nvcc-flags` on `compile` and `run`, the `nvcc_flags`
     option of the dynamo backend and the bench worker, the `Makefile` test targets, the corpus timings workflow, the
     image bake, warm and verify scripts, three skills, the measuring tutorial.
   - **Stages and dumps.** `--ir cuda`, the default stage of `emmy compile`, prints PTX; the dump writes `.ptx`.
   - **The corpus.** `built` no longer needs the card: ptxas assembles any target anywhere. The realization
     `ARCHITECTURE.md` and the corpus rules in `AGENTS.md` change with it.
   - **Docs.** The CUDA backend, IR and passes `ARCHITECTURE.md` files, `AGENTS.md` (prerequisites, running tests),
     `README.md`, `GLOSSARY.md` for PTX.
   - Then record runs on each card and the refit.

### Checks

Without a GPU: crate tests per rule; the verifier; every corpus case compiled to PTX for its declared capability, and
assembled at `-O0` and `-O3` wherever ptxas is installed.

On a card, in this order: the differential against nvcc's build of the same kernel while both exist (within the corpus
tolerance; bit-equal for a kernel with no math function once the fusion rule matches cicc's); the same kernels
assembled at `-O0`; compute-sanitizer with its race and barrier checks over the corpus on one card; the corpus at
`correct` and `make test`; the math sweeps; the interleaved perf lane last.

Accepted when every corpus case on every held card is correct through PTX, `make test` is green, no perf-lane case is
more than 5% slower than PR A without a stated cause, and the build-time table is in the PR body.

## Cards

Held now: RTX 5090 (rental and dev box), A100 and V100. An H100 is requested on GCP and pending. No RTX 4090: `sm_89`
gets PTX assembled but not run, and the PR says so. PR B's tensor-core and Hopper milestones each need their card:
Volta on the V100, fp4 on a 5090, `wgmma` on the H100.

## Risks

- **Instruction order.** ptxas schedules from the order it is given. #1386 showed one kernel 1.4x slower from the same
  loads issued later. Any change of translator moves that order, so scattered moves both ways are expected in PR B's
  speed milestone. The fix for a kernel that stalls on memory is issuing the next step's loads early in Kernel IR.
- **Tensor-core kernels change translator and fragment expansion in one PR.** The differential against nvcc's build is
  the guard, family by family, before the perf lane is read.
- **Math accuracy and edge values.** NaN, infinities, subnormals and the maximum's NaN rule must match what kernels
  give today. The sweeps cover values; the corpus covers kernels.
- **Half precision.** Which operations run natively in f16 today, and their PTX forms per target, must be confirmed
  against the PTX reference in step A3.
- **PTX versions.** Each instruction family needs a minimum `.version`, and each toolkit's ptxas has a maximum (9.0 on
  CUDA 13.0). The table is checked against the PTX reference and the toolkits on the held cards, not assumed.
- **PR B is big.** If it has to land in parts, the fallback of its milestones is the seam.

## Open decisions

1. **Fragments expand in the compiler** (recommended). The other way, expanding them into plain Kernel IR statements
   in PR A, would let nvcc measure the expansion before the compiler exists. It needs a statement for an in-place
   register update, which plain statements cannot express today, and body optimization would hoist the expanded loads
   in the fastest kernels with no knob. The lane maps are NVIDIA-only facts in any case.
2. **Delete `LOOPIFY` and readable mode** (step A8). Done, except the knob's name: stored rows carry `LOOPIFY: 0`
   (Status). Rewriting them (every golden, the freezes, a tune DB version) removes it.
3. **ptxas as a binary** (recommended first) **or the PTX compiler library linked in.** The toolkit ships a static
   library; linking it would drop the subprocess and the toolkit from serving images. Its redistribution terms need a
   look first.
4. **`plans/` holds 11 files with this one**, over the cap of 10. One executed plan should go.

## Left out

- **Work sharing across lanes, re-tiling, sibling loop fusion.** Schedule questions; the fastest entries show the size
  of the prize (row 38: half of nvcc's time), so it deserves its own plan.
- **A partial unroll factor and early loads as knobs.** The earlier plan's milestone 3. They build on step A8, whose
  pass holds the factor as two constants copied from cicc (64 trips in full, 32 copies per trip past that). Row 41
  issues the next trip's loads a trip ahead and reports 24 attention kernels at 0.36x to 0.97x of their time, and every
  unroll threshold in the entries was tuned on one card, so both belong to the search.
- **A host interpreter of the instruction list.** It would give CI, which has no GPU, a check of kernel code, and row
  54's found a race. It is also a second statement of what every opcode means, to be kept right. The sanitizer on a
  card covers races and barriers; revisit if one escapes it.
- **Other targets.** The three levels are drawn so that an AMD compiler would replace only the third.

## Sources

Sheet rows of "Emmy Hackathon 2026 Submissions", each at its cutoff commit.

| Row | Repository | Read for |
| --- | --- | --- |
| 20 | [Tirondzo/emmy-ptxc-hackathon](https://github.com/Tirondzo/emmy-ptxc-hackathon/tree/d89a724f8b) | instruction list and builder (`ptxc/ptx/`), unroll policy, measured cost of each fast form (`tests/FINDINGS.md`) |
| 26 | [currybab/emmy-hackathon-2026](https://github.com/currybab/emmy-hackathon-2026/tree/22255bf035) | facts per register (`ptxc/builder.py`), affine offsets (`ptxc/expr.py`), rejected experiments (`REPORT.md`) |
| 54 | [Dino-ArchE/GPU-Compiler-Hackathon](https://github.com/Dino-ArchE/GPU-Compiler-Hackathon/tree/9ae10c8372) | verifier (`src/core/opt/verifier.py`), thread interpreter (`tests/interp.py`), the cost of a full optimizer |
| 11 | [stashuk-olek/deuteron](https://github.com/stashuk-olek/deuteron/tree/8590ce82d2) | the LLVM route and its costs (`compiler/build.py`, `compiler/ptx.py`) |
| 38 | [shubham-jadhav-FiatLux/emmy-ptx-compiler](https://github.com/shubham-jadhav-FiatLux/emmy-ptx-compiler/tree/36a6138009) | value with range and lane uniformity (`emmy_ptx/builder.py`), math (`emmy_ptx/mathlib.py`), IR interpreter (`tests/oracle.py`), ablation (`docs/DESIGN.md`) |
| 52 | [N00bcak/emmy-hackathon-2026](https://github.com/N00bcak/emmy-hackathon-2026/tree/26da042232) | "must survive to machine code" rule, the `sm_120` shared base (`docs/shared-window-fix.md`), failed experiments |
| 39 | [mrbahrani/emmy-hackathon-2026](https://github.com/mrbahrani/emmy-hackathon-2026/tree/0f5c13e9f2) | PTX emulator with race and barrier checks (`tools/ptxemu.py`), trip counts from tile ranges |
| 37 | [RatanKokal/emmy-hackathon-2026](https://github.com/RatanKokal/emmy-hackathon-2026/tree/2cb042f4d1) | differential against nvcc's build (`tools/harness/differential.py`), instruction-form sweep through ptxas |
| 34 | [bledden/emmy-hackathon](https://github.com/bledden/emmy-hackathon/tree/fe469c5b40) | participation and barrier proofs (`ptxgen/participation.py`), load-merging proof (`ptxgen/shared_loads.py`) |
| 31 | [Skytuhua/emmy-ptx](https://github.com/Skytuhua/emmy-ptx/tree/6d06581947) | uniformity analysis for barriers (`emmy_ptx/control_facts.py`), instruction verifier (`emmy_ptx/verify.py`) |
| 56 | [titoatwork/emmy-ptx-compiler-submission](https://github.com/titoatwork/emmy-ptx-compiler-submission/tree/68bc29e0aa) | launch contract checks (`ptxc/lower.py`), `.maxntid`, changes that removed instructions and ran slower |
| 35 | [alexpejovic/emmy-hackathon-submission](https://github.com/alexpejovic/emmy-hackathon-submission/tree/cdc57e31a5) | what ptxas does, with machine code (`examples/cse/`, `examples/loop_unroll/`, `examples/op_fusion/`), trial unrolling with rollback |
| 45 | [KunnnnnQ/emmy-ptx-compiler](https://github.com/KunnnnnQ/emmy-ptx-compiler/tree/3ce613d0e8) | the quotient and remainder split that makes a grid decode free (`ptxc/index.py`), unroll rule around barriers |
| 41 | [huixu11/emmy-hackathon-2026](https://github.com/huixu11/emmy-hackathon-2026/tree/5e09eb8e08) | range and divisor per register (`ptxc/facts.py`), multiply-high division (`ptxc/index.py`), next-trip loads (`ptxc/loops.py`), tensor cores (`ptxc/tensorcore.py`) |
| 8 | [nileshpatil6/emmy-ptx-compiler](https://github.com/nileshpatil6/emmy-ptx-compiler/tree/ed6378c50b) | `cp.async` alignment proved from divisibility (`emmyptx/memory.py`), A/B table (`RESULTS.md`) |
| 30 | [LadyJ101/emmy-hackathon-2026](https://github.com/LadyJ101/emmy-hackathon-2026/tree/2a41e045a1) | zero-based loop counter, non-affine index pieces with ranges (`compile.py`) |
| 18 | [ayu-ch/emmy-ptx-compiler](https://github.com/ayu-ch/emmy-ptx-compiler/tree/2fe525c5df) | float64 torch reference (`tools/check_torch.py`), shared address register across unrolled copies |
