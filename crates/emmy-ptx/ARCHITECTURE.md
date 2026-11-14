# The Kernel IR to PTX compiler

`emmy-ptx` translates one kernel, a Kernel IR body with its interface, into PTX text that ptxas assembles. It is the
only code generator: a statement, an operation or a target instruction it has no rule for is an error. It reads the
statements of the `emmy-ir` crate as they are, so nothing is serialized on the way.

**It decides nothing.** Every choice is in the Kernel IR before it runs: the schedule, the unrolled copies, the
approximate operations, the staging. Two whole-kernel precision decisions, multiply-add fusion and flush to zero, and
whether the launch is one wave reach it as input. ptxas keeps register allocation, instruction scheduling and the
unrolling of small loops.

The crate covers scalar, cooperative and tensor-core kernels: the grid, loops, conditions, loads, stores, casts, the
elementwise operations and the math functions kernels hold; shared memory, barriers, shuffles, the tree reduction,
vector access, atomic adds, async copies and slab layouts; fragments, `ldmatrix`, `mma.sync`, the fragment statements,
block scales, and Hopper's `wgmma`, tensor copy, `mbarrier` and register limit; paged buffers.

## Layering

```
lowering/cuda/010_lower_kernelop ─▶ emmy/compiler/ir/kernel/ptx.py ─▶ emmy.emmy_ir.compile_ptx ─▶ emmy-ptx
                                                                                                   │
            emmy-ir: statements, range analysis, lane-to-element tables ◀──────────────────────────┘
backend/cuda/ptxas.py: assembles the text, keys the cubin on it
```

The crate depends on `emmy-ir` alone: its statements, the range analysis body optimization uses, and the
lane-to-element table of the fragments (`fragment::fragment_table`). Python reaches it through the extension; nothing in
it knows the pipeline, the search or the backend.

## Entry points

- `compile(kernel, target, precision)` returns the PTX text. `lower` returns the instruction list it prints, which is
  what the crate's tests read.
- An error is one of two kinds. `Error::NoRule` says the compiler has no rule for a statement, an operation or a
  target's instruction. Any other error refuses the kernel, with the statement's text: an input the compiler does not
  take, or the verifier's finding, which is a bug of the compiler or of the pass that wrote the kernel.
- `Kernel` is the entry name, the body and the `Interface`: the parameters in the launch's argument order, each buffer
  with the alignment the launch gives it, the per-block thread bound, the shared-memory pool, and whether the launch is
  one wave. `Interface` and `Param` mirror the Python `ir/kernel/interface.py`, which computes them.
- The `emmy.emmy_ir` extension exports `compile_ptx` and `NoRule` (a `NotImplementedError`); every other error is a
  `ValueError`. `emmy/compiler/ir/kernel/ptx.py` is the Python entry the lowering calls.

## What the compiler expects of a body

The body is typed and flat. Every value-defining statement carries its dtype, every conversion is a cast in the body,
and a `Load` or a `Write` holds one element index into its buffer. No statement carries a request: a loop flagged
`unroll` is refused, since unrolling is a kernel pass. A loop that says `unrolled` holds copies that pass already
wrote, and stays a loop in the code. The compiler checks what it reads and refuses with the statement's text: a value
with no dtype, an operand of another type than its reader, an index of several coordinates, a store into an input.
Integers are the one exception to "every conversion is a cast": their width and signedness are the compiler's. Two
statements still reach it with no dtype, the cooperative combine's `maximum` and an index `let` the chunk bounds
write; each computes at its operands' type. A literal takes its reader's type.

A condition the body's optimization decided and left standing (`if 1:`) is no branch. A comparison defines a `bool`. A
byte (`u8`) is a buffer's element type only; an 8-bit float and a byte of two e2m1 codes stand in the low byte of a
32-bit register, and no arithmetic reads them. A fragment is a local buffer of registers: `RegFragment` declares it,
the tensor-core statements read and define its registers in place, and a `Load` or a `Write` that names it reaches one
register by a constant index. A name some statement updates in place (a shuffle's state, a staged copy's registers) is
found before lowering starts and is a state from where it is defined.

## The instruction list

A kernel between the rules and the printer is a `Module`: the directives, the parameter and shared declarations, the
register count of each bank, and one flat list of instructions, labels and comments. An instruction is an opcode, the
type it computes at, a rounding mode, the flush-to-zero flag, the registers it defines, its operands and an optional
predicate guard. Opcodes, types, rounding modes, state spaces and modifiers are enums; no instruction is a string
before `print`. Register banks follow PTX: predicate, 16-bit, 32-bit, 64-bit and f32. The printer names the module's
symbols by position (`<entry>_param_<n>`, `<entry>_shared_<n>`), so no Kernel IR name has to be a PTX identifier, and
writes each statement's first line as a comment above its instructions.

## The builder

Every instruction reaches the list through the `Builder`.

- **Registers.** One per Kernel IR name; ptxas allocates the real ones. A state is a register defined again where it
  steps, so there are no phi nodes.
- **Structure.** The builder is the only writer of labels and branches. A loop's test stands at the bottom; a loop
  whose first trip is not known to run has a second test ahead of it.
- **The value table.** Each open scope keeps the pure instructions it wrote, so a value the compiler makes twice is
  made once. A scope's table is dropped when the scope closes. The table matters even where ptxas would merge the
  copies, because ptxas decides its unrolling from the size of the PTX it is given. It shares only what the compiler
  itself makes (address parts, widenings, constants, a lane's offset in a tile); it never moves a Kernel IR value or a
  load.
- **Values a loop does not change.** A pure integer value the compiler makes whose operands are all defined outside
  the loops around it is computed ahead of the outermost of them. A condition is never left and a float value never
  moves: the body's optimization placed every value the kernel names.
- **Facts.** The builder keeps, per register, a number its value is a multiple of and, for an address, the bytes it is
  aligned to, starting at what a `Param` or a `Smem` states. Ranges are not kept per register: they come from the
  `emmy-ir` range analysis on the expression being lowered, under the same context body optimization uses.

## The verifier

`verify` runs at the end of every `lower`, for what ptxas does not check. It requires every register an instruction
reads to be defined on every path from the entry (a definition under a guard counts for no path), every label a branch
names to be bound once, and an alignment fact behind every access of several elements: a vector load or store, both
addresses of an async copy, each row `ldmatrix` reads and the destination of a tensor copy. A register read undefined
assembles without complaint and can give a right answer at one optimization level and a wrong one at another.

## Rules

- **Loops.** A loop whose trip count is the same for every thread is a counter from zero to that count, the shape ptxas
  unrolls. A strided loop that starts at a thread's own value steps its axis itself; one that runs at most one trip is
  the condition that its start is below its end. The accumulators a loop folds are seeded ahead of it. A loop that
  says `unrolled` carries `.pragma "nounroll"`, so ptxas keeps it as it is; a static loop of at most four trips stays
  unmarked, and ptxas writes it out.
- **Integers.** An expression is computed at the narrowest of `i32`, `u32` and `i64` that holds its range. A sum, a
  product or a bitwise operation runs at the type its own range needs; a division, a shift right, a minimum or a
  comparison at the first type that holds both operands unchanged.
- **Division.** Integer `/` and `%` floor. By a power of two they are a shift and a mask; by another constant under a
  32-bit dividend, a multiply-high and a shift; else the divide instruction, unsigned where neither side can be
  negative.
- **Addresses.** A flat index is split into its terms and its constant. Each term is scaled to bytes, widened and added
  to the base in 64 bits; the constant is the address's offset. A term of a loop's axis reads a 64-bit copy of the
  counter stepped beside it; in a loop marked for ptxas to keep, a multiple of the counter stepped by the term's bytes.
  Addresses into shared memory are 32-bit.
- **Floats.** Every instruction that rounds carries `.rn`, so ptxas fuses nothing on its own, and every f32 instruction
  that takes it carries `.ftz` when the flush-to-zero decision is on. `fma` is written where the body says `fma`, and,
  with the fusion decision on, for an f32 product read once by an add, a subtract or an accumulate in its own body. A
  half product is never fused: each half operation rounds, as eager's do.
- **Half precision.** `f16` add, subtract, multiply, fused multiply-add, negation, absolute value, minimum and maximum
  run natively; any other `f16` operation computes in f32 and rounds once. `bf16` is converted, never computed in.
- **The module.** A kernel waits for the grid launched ahead of it (`griddepcontrol.wait`) before anything else, on the
  targets that have the instruction. A load of a buffer the kernel never writes goes past the coherent cache
  (`ld.global.nc`).
- **Shared memory.** A `Smem` is an array of bytes aligned to its elements or to what the statement asks for; where the
  interface gives a pool, one external array and a buffer is its offset in it. A buffer's address is read from its
  symbol and never taken as zero.
- **Memory access.** A `Load` of several names or a `Write` of several values is one access of 4, 8 or 16 bytes. An
  atomic `Write` is one `atom.add`. A `CpAsyncCopy` is one `cp.async`, writing zeros where `valid` fails. A shared slab
  can be stored under a swizzle (`B128`, `B64`, `B32`: 16-byte chunks exchanged over the flat index) or one of Volta's
  two layouts (`fragment::volta_slab_index` in `emmy-ir`).
- **Carried states.** A cell of a carried state at a literal index is two registers, seeded ahead of the loop and moved
  at the end of each trip. A cell at an index known only where the kernel runs has no rule.

## Math

A math function is a sequence of f32 instructions written through the builder. Every instruction that rounds carries
`.rn`, a fused multiply-add stands only where the sequence writes one, and no sequence reads the fusion decision; the
flush-to-zero decision reaches every instruction that takes the modifier. A half computes its function in f32 and
converts once. Only the functions kernels hold are written: `exp`, `log`, `tanh`, `erf`, `rsqrt` (the hardware
instruction), `sin`, `cos` and `pow`; `sqrt`, `divide` and `reciprocal` are one correctly rounded instruction each. A
composite such as `sigmoid` is compiled as its definition in the operations table. A math function outside this list
is an error that names it.

Each sequence is held to the bound NVIDIA's CUDA math documentation states for the same single-precision function,
with the special values the CUDA function gives; `test_math.py` measures that on a card. The polynomial coefficients
are minimax fits made for these sequences; the crate's tests check each table against the function on the host.

An approximate operation is one hardware instruction, on its operand as it stands or scaled. One table in the math
module holds them, a row for each `_fast` operation the operations table of `emmy-ir` declares:

| Operation | Instructions |
| --- | --- |
| `exp_fast` | `ex2.approx(x * log2(e))` |
| `log_fast` | `lg2.approx(x) * ln(2)` |
| `pow_fast` | `ex2.approx(y * lg2.approx(x))` |
| `sin_fast`, `cos_fast` | `sin.approx`, `cos.approx` |
| `tanh_fast` | `tanh.approx`, from `sm_75` |
| `divide_fast`, `reciprocal_fast` | `div.approx`, `rcp.approx` |
| `sqrt_fast` | `sqrt.approx` |

Each flushes to zero when the kernel's decision says (`tanh.approx` has no such form).

## Cooperation between threads

- **The combine program.** `WarpShuffle` and `TreeHalve` carry a program of statements over their state names; it goes
  through the same statement rules.
- **Shuffles.** A shuffle moves 32 bits. A butterfly halves its stride; a prefix doubles it and steps the state under a
  predicate guard, so the warp does not split.
- **The tree reduction.** Where each thread has a slot of its own, a halving step is a condition over the low half and
  a barrier. Where the lanes of a warp share a slot, the block's first lanes take one slot each and fold them by a
  butterfly under the mask of exactly those lanes, so no slot is read by one lane and written by another.
- **What is decided of a block's threads.** An integer or bool register whose value is an expression over the thread
  index alone is evaluated for every thread of a block. From that the lowering knows which threads may reach a
  statement, which have left the kernel, and whether a loop runs a trip count that differs between threads.
- **Barriers.** `Sync` is `bar.sync`, or `bar.warp.sync`. A barrier of the whole block is refused where a thread still
  in the kernel is decided not to reach it, or in a loop of differing trips. A named barrier is refused when its count
  is not the threads decided to reach it, or not whole warps.

## Tensor cores

- **Fragments.** A fragment of n registers is n registers of the instruction list, each a state. An accumulator starts
  at zero; an operand holds nothing until a statement loads it, so the verifier refuses a product that reads one no
  path loaded.
- **`ldmatrix`.** Each lane gives the address of one row of eight halves in a shared slab; the lane's offset is an
  integer expression over the thread index, computed once a kernel.
- **`mma.sync`.** One instruction a statement:

  | Tile | Operands | Accumulator | Registers A, B, C | Target, PTX |
  | --- | --- | --- | --- | --- |
  | m8n8k4, B by columns or by rows | `f16` | `f32` or `f16` | 2, 2, 8 or 4 | `sm_70`, 6.4 |
  | m16n8k16 | `f16` | `f32` or `f16` | 4, 2, 4 or 2 | `sm_80`, 7.0 |
  | m16n8k16 | `bf16` | `f32` | 4, 2, 4 | `sm_80`, 7.0 |
  | m16n8k32 | `e4m3` or `e5m2` | `f32` | 4, 2, 4 | `sm_89`, 8.4 |
  | m16n8k64, block-scaled | `e2m1` | `f32` | 4, 2, 4, a scale register a side | `sm_120a`, 8.7 |

- **The fragment statements.** `FragmentLoad`, `RegStore`, `FragmentApply` and `FragmentRowReduce` are each one
  expansion over their row of the lane-to-element table: per register and element the table gives the element's row
  and column in the tile, and the rule adds them to the statement's own index. No rule holds a lane map of its own.
  Elements a register holds as they stand in memory, proven aligned, are one access; a clamp or a guard is a predicate
  on each load, never a branch. A slab of codes beside a slab of block scales decodes each byte, multiplies it by its
  block's scale in f32 and rounds once. `BlockScaleLoad` gives a lane the four scales of its row or column as one
  register.
- **The promote and the repacks.** An accumulator of halves is folded into its accumulator of floats and zeroed. An
  accumulator of floats becomes an operand of the next product without leaving registers.

## Hopper

`WgmmaDescriptor` is the 64-bit shared-memory matrix descriptor; the rule refuses a slab with no 16-byte alignment
proven. `WgmmaMma` is one `wgmma.mma_async`; the fence, the commit and the wait are one instruction each. A tensor
descriptor is built by the host and passed as a pointer parameter; `TmaDescriptor` declares it and writes nothing, and
`TmaLoad` is one `cp.async.bulk.tensor` against a barrier. An `mbarrier` is 64 bits of a shared array; an arrival
stands behind `fence.proxy.async`. `SetMaxNReg` is one `setmaxnreg`.

## Targets

The module's `.maxntid` is the interface's thread bound: without it ptxas keeps registers for 1,024 threads. A launch
of one wave adds `.minnctapersm 1`, so ptxas does not size the kernel's registers for more blocks on a multiprocessor
than the launch places there.

The module's `.target` is the card's compute capability, with the `a` suffix when the interface says the kernel needs
the arch-specific instruction set. Its `.version` is the target's own (7.8 below `sm_100`, 8.7 from there), raised by
each instruction that needs a later one. Every rule that writes an instruction not every card has states what it
needs, and refuses a target that lacks it with a message naming both. The cards are what ptxas of CUDA 13.0 takes:

| Instruction | Cards | PTX |
| --- | --- | --- |
| `ldmatrix` | `sm_75` and later | 6.5 |
| `cvt` to `f16x2` or `bf16x2` | `sm_80` and later | 7.0 |
| `mma.sync`, by form | the table above | |
| `mbarrier.init` | `sm_80` and later | 7.0 |
| `mbarrier.try_wait` | `sm_90` and later | 7.8 |
| `fence.proxy.async`, `mbarrier.arrive.expect_tx` | `sm_90` and later | 8.0 |
| `cp.async.bulk.tensor` into `.shared::cta` | `sm_90` and later | 8.6 |
| `setmaxnreg` | `sm_90a` and later, arch-specific only | 8.0 |
| `wgmma` | `sm_90a` alone: refused on `sm_100a` and `sm_120a` | 8.0 |

## In the pipeline

`lowering/cuda/010_lower_kernelop` generates each kernel's code here, and any refusal is an error. The CUDA backend
`ARCHITECTURE.md` says how the PTX is assembled and what is keyed on it.

A paged buffer is bound as a table of its pages (`Param.page`). One page that holds the whole buffer is read from the
table once and is the buffer's base; otherwise every access reads its page's address from the table and addresses the
remainder inside that page, one element at a time. The e2m1 pair table an NVFP4 weight decodes through is named by the
interface (`Param.pair_table`), and a load of it computes its value from the index and reads nothing.

## Tests

`cargo test -p emmy-ptx` runs without a GPU. The rule tests build bodies from the text form, compare the instruction
list, and assemble the printed kernel with ptxas at `-O0` and `-O3` wherever ptxas is on the path. `tests/compiler/ptx/`
launches compiled kernels on a card, each assembled at ptxas's default level and at `-O0`, and compares them with
NumPy bit for bit, except an atomic sum of arbitrary floats. The tensor-core tests write each fragment layout by hand,
as the PTX reference documents it, with small integers as operands so a product is exact. `test_corpus.py` compiles
every realization corpus case without a card and assembles it for the case's declared capability; `python -m
tests.compiler.ptx.corpus` prints what became of each kernel. The math functions are held to their documented bounds
over a sample by default, and over every f32 bit pattern under `pytest -m sweep`.
