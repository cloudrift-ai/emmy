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

PR A is draft #1393. Steps 1 to 5 are done, each checked on the corpus's rendered CUDA. Steps 3 to 5 changed the
text and ran the corpus at `correct` on the RTX 5090. No step ran the perf lane or the whole suite yet; both wait for
the closing step.

| Step | Commit | Corpus CUDA |
| --- | --- | --- |
| 1 | `7c5c6932c` | identical |
| 2 | `ff201cee1` | identical |
| 3 | `8e44947fb` | 58 of 296 kernels: conversions as definitions, literals at their dtype, one integer negation |
| 4 | `42c46cc44` | 72 kernels: the `lane` and `warp` declarations, dropped in 42 and moved into the grid in 30 |
| 5 | `0936d9d9d` | 3 kernels: a `sigmoid` as four assignments |

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
   programs itself, from the stamped states. The register fragment statements and `RegStore`: step 10 covers them.
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
   the approximate form of each instruction class: exponential, division and reciprocal, square root and its
   reciprocal, and the functions `--use_fast_math` covers today (log, sine, cosine, power). Each form is its own
   elementwise operation, as `exp_fast` and `divide_rn_f32` already are. *Check: perf lane level.*
7. **nvcc flags shrink to the two whole-kernel decisions**, as its own commit after step 6: `--use_fast_math`,
   `--prec-div` and `--prec-sqrt` go; `--fmad` and `--ftz` stay until PR B hands them to the compiler. This commit
   re-baselines timings by itself, so it carries its own perf-lane run.
8. **Unrolling in Kernel IR** (property 3). A last kernel pass copies the body of every loop flagged `unroll` with a
   static trip count, binding the axis to its value in each copy; body optimization then folds each copy's index and
   drops guards the value decides. `047_reuse_lane_loads` already unrolls this way; its unroll becomes the shared one.
   Only the atom lowering sets the flag, so no Loop IR statement changes. The `#pragma` leaves statement loops, and
   with it what exists only to re-roll code into pragma loops: `100_loopify`, `RegFragment.count`, `EMMY_READABLE`
   and `--no-readable` (open decision 2). Cost, measured: body optimization takes 0.17 s on 4,608 statements and
   0.46 s on 9,216, and runs again at each later construction; the `UNROLL` budget is the cap. Unflagged loops stay
   rolled. *Check: perf lane, pipeline time per kernel, nvcc time per kernel.*
9. **Flat indices** (property 2). The last kernel pass turns each `Load`, `Write` and `CpAsyncCopy` index into one
   flat index, carrying the 64-bit widening the renderer applies today for buffers past the 32-bit range. Body
   optimization then spells it affine under the ranges in scope. Statements that are defined by their dimensions keep
   them: `RegStore`, the Volta shared layouts, `TmaLoad` coordinates. A paged buffer is a graph fact the cached kernel
   does not know, so its page and offset split stays in `lowering/cuda`. *Check: text shows simplified indices; perf
   lane.*
10. **Fragment statements say what they are** (property 4). No expansion moves; the statements get the shape the
    compiler needs. `LdmatrixLoad` splits into the instruction on staged shared memory and a fragment load from global
    memory, whose role, transpose, clamp and zero-fill are plain fields. `FragLayout`, the lane-to-element table, moves
    into the crate; the Python renderers read it through the binding until PR B deletes them. The lane coordinates
    these statements and the prefix scan compute in rendered text (`_g`, `_t`, `threadIdx.x % length`) follow from
    that table and the lane of step 4. *Check: text identical.*
11. **Tests read Kernel IR.** 59 test files hold 186 reads of the rendered source. A check of structure becomes a
    check of Kernel IR statements now. A check of an instruction's spelling stays on the CUDA text until PR B replaces
    it with a crate test.
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
2. **Delete `LOOPIFY` and readable mode** (step A8). They only re-roll code into `#pragma unroll` loops. Dropping a
   feature is the owner's call; the plan assumes yes.
3. **ptxas as a binary** (recommended first) **or the PTX compiler library linked in.** The toolkit ships a static
   library; linking it would drop the subprocess and the toolkit from serving images. Its redistribution terms need a
   look first.
4. **`plans/` holds 11 files with this one**, over the cap of 10. One executed plan should go.

## Left out

- **Work sharing across lanes, re-tiling, sibling loop fusion.** Schedule questions; the fastest entries show the size
  of the prize (row 38: half of nvcc's time), so it deserves its own plan.
- **A partial unroll factor and early loads as knobs.** The earlier plan's milestone 3. They build on step A8. Row 41
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
