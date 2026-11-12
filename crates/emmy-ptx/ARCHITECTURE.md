# The Kernel IR to PTX compiler

`emmy-ptx` translates one kernel, a Kernel IR body with its interface, into PTX text that ptxas assembles. It reads
the statements of the `emmy-ir` crate as they are, so nothing is serialized on the way. It is a translation: it makes
no decision. A kernel's two whole-kernel precision decisions, multiply-add fusion and flush to zero, reach it as
input, and every other choice is in the Kernel IR before it runs. ptxas keeps register allocation, instruction
scheduling and the unrolling of small loops.

The crate covers scalar and cooperative kernels today. Scalar: the grid, loops, conditions, loads, stores, assigns,
selects, accumulators and seeded states, carried states, casts, the elementwise operations that are one instruction
each, and the math functions kernels hold (below). Cooperative: shared memory, barriers, warp shuffles, the tree
reduction, vector access, atomic adds, async copies and the zeroing one kernel does for another. The tensor-core
statements, the bulk tensor copy and `wgmma` have no rule yet, and a kernel that holds one is an error that names it.
So is any math function outside the table below.

## Entry points

- `compile(kernel, target, precision)` returns the PTX text. `lower` returns the instruction list it prints, which is
  what the crate's tests read.
- `Kernel` is the entry name, the body and the `Interface`: the parameters in the launch's argument order, each
  buffer with the alignment the launch gives it, the per-block thread bound, the shared-memory pool. `Interface` and
  `Param` mirror the Python `ir/kernel/interface.py`, which computes them.
- The `emmy.emmy_ir` extension exports `compile_ptx`, so a body crosses from Python as the crate's own value.

## What the compiler expects of a body

The body is typed and flat. Every value-defining statement carries its dtype, every conversion is a cast in the body,
and a `Load` or a `Write` holds one element index into its buffer. No statement carries a request: a loop flagged
`unroll` is refused, since unrolling is a kernel pass. The compiler checks what it reads, statement by statement, and
refuses with the statement's text: a value with no dtype, an operand of another type than its reader, an index of
several coordinates, a store into an input. Integers are the one exception to "every conversion is a cast": their
width and signedness are the compiler's, so `i32`, `u32` and `i64` values mix freely in an index.

A literal has no type of its own. It takes its reader's: an integer literal in a float expression is that float.

A byte (`u8`) is a buffer's element type only: a load gives an integer with its range, a store takes the low byte of
one. The fp8 and fp4 element types have no rule until their decode has one.

A name some statement updates in place, the state of a warp shuffle or a tree reduction, the registers of a staged
copy, is found before lowering starts and is a state from where it is defined.

## The instruction list

A kernel between the rules and the printer is a `Module`: the directives, the parameter and shared declarations, the
register count of each bank, and one flat list of instructions, labels and comments. An instruction is an opcode, the
type it computes at, a rounding mode, the flush-to-zero flag, the registers it defines, its operands and an optional
predicate guard. Opcodes, types, comparisons, rounding modes, state spaces, shuffle modes and cache modifiers are
enums; no instruction is a string before `print`. The hardware's approximate form of an instruction (`ex2.approx`,
`rcp.approx`) is spelled where its rounding mode would stand. An operand is a register (bank and number), an integer
or float constant, an address (a base and a constant byte offset), a register tuple, the address of a shared
declaration, a special register or a label.

Register banks follow PTX: predicate, 16-bit, 32-bit, 64-bit and f32. Signedness is on the instruction. A half lives
in a 16-bit register, a bool in a predicate register (a byte in memory), and a byte is held zero-extended in 32 bits.

The printer names the module's symbols by position (`<entry>_param_<n>`, `<entry>_shared_<n>`), so no Kernel IR name
has to be a PTX identifier. A float constant is printed by its bits. Each statement's first line stands as a comment
above its instructions; every non-ASCII character in one is replaced, since ptxas refuses the byte.

## The builder

Every instruction reaches the list through the `Builder`.

- **Registers.** One per Kernel IR name; ptxas allocates the real ones. A state (an accumulator, a value an `Init`
  seeds) is a register that is defined again where it steps, so there are no phi nodes. A name bound to a state is a
  copy of it.
- **Structure.** A loop or a condition is opened and closed through the builder, which is the only writer of labels
  and branches. A loop's test stands at the bottom; a loop whose first trip is not known to run has a second test
  ahead of it.
- **The value table.** Each open scope keeps the pure instructions it wrote, by opcode, modifiers and operands, so a
  value the compiler makes twice is made once. A scope's table is dropped when the scope closes: not every thread ran
  a condition's side, and a loop's body runs again with other values. Nothing computed from a state is kept. The table
  matters even where ptxas would merge the copies, because ptxas decides its unrolling from the size of the PTX it is
  given. The table shares what the compiler itself makes (address parts, widenings, constants in registers); it never
  moves a Kernel IR value or a load, and a load is never shared.
- **Facts.** The builder keeps, per register, a number its value is a multiple of and, for an address, the bytes it is
  aligned to. A parameter's pointer starts at the alignment its `Param` states, a shared buffer's address at its
  declaration's, and each term added to an address keeps what the term's multiple allows. Ranges are not kept per
  register: they come from the `emmy-ir` range analysis on the expression being lowered, under the same context body
  optimization uses (`launch_ctx` for a grid's built-ins, `inner_ctx` for the axes a statement binds). An integer
  name's range joins that context where the name is bound.

## Rules

- **Loops.** A loop whose trip count is the same for every thread is a counter from zero to that count, which is the
  shape ptxas unrolls. A plain `Loop`'s axis is the counter. A `StridedLoop` with a literal start and step over a
  static extent counts its trips and computes the axis from the counter; one that starts at a thread's own value, or
  stops early, steps the axis itself. The accumulators a loop folds are seeded ahead of it with the combine's neutral
  element, as the `seed` flag says.
- **Integers.** An expression is computed at the narrowest of `i32`, `u32` and `i64` that holds its range, and at its
  operands' type where it has no range. A sum, a product or a bitwise operation depends on the low bits of its
  operands alone, so it runs at the type its own range needs. A division, a shift right, a minimum or a comparison
  reads every bit, so it runs at the first type that holds both operands unchanged. A product of two 32-bit values
  that needs 64 bits is the one widening multiply.
- **Division.** `/` and `%` over integers floor. By a constant power of two they are an arithmetic shift and a mask,
  which floor whatever the sign. By another constant, under a 32-bit dividend, the quotient is a multiply-high and a
  shift, exact below 2^31; a dividend the ranges do not prove non-negative is folded into that range by its sign and
  the quotient folded back. Anything else is the divide instruction, lowered by one where the remainder is nonzero
  and its sign is not the divisor's, unless the operands are unsigned or the ranges prove both sides non-negative.
- **Addresses.** A flat index is split into its terms with their integer coefficients and its constant. Each term is
  scaled to bytes and widened by one multiply, then added to the base in 64 bits, the terms bound in outer bodies
  first; the constant is the address's offset. Measured on `sm_120`: ptxas folds an address built this way into
  `[base + constant]` across the trips of a loop it unrolls, and does not see through a sum made in 32 bits. There is
  no pointer bumping and no fused wide multiply-add; both measured slower in the hackathon entries.
- **Floats.** Every instruction that rounds carries `.rn`, so ptxas fuses nothing on its own, and every f32
  instruction that takes it carries `.ftz` when the flush-to-zero decision is on. `fma` is written where the body says
  `fma`. With the fusion decision on it is also written for a product read once, by an add, a subtract or an
  accumulate in the product's own body: the product is held back at its definition and becomes one instruction with
  its reader. A difference is the fused sum over an exact negation. A product whose reader does not take it is
  computed where it is first read.
- **Half precision.** `f16` add, subtract, multiply, fused multiply-add, negation, absolute value, minimum and maximum
  run natively (minimum and maximum from `sm_80`). `bf16` is converted, never computed in. Half instructions take no
  immediate, so a half constant is moved into a register by its bits.
- **The module.** A kernel waits for the grid launched ahead of it (`griddepcontrol.wait`) before anything else, on
  the targets that have the instruction (`sm_90` and later). A load of a buffer the kernel never writes, a parameter
  of kind input, goes past the coherent cache (`ld.global.nc`), as nvcc loads through the `const` pointer the CUDA
  renderer declared.
- **Shared memory.** A `Smem` is an array of bytes of the module, aligned to its elements or to what the statement
  asks for. Where the interface gives a pool, the module declares one external array aligned to the strictest
  buffer, and a buffer is its offset in it. Either way the buffer's address is read from the symbol where the buffer
  is declared and is never taken as zero: on `sm_120` neither an array nor the pool starts where the block's shared
  memory does (measured: `0x400`). Addresses into shared memory are 32-bit arithmetic.
- **Vector access.** A `Load` of several names or a `Write` of several values is one access of 4, 8 or 16 bytes: of
  that many registers where PTX has the form (two or four f32, two or four halves, two or four bytes, two 64-bit
  integers), else of 32-bit words that hold the elements (eight halves, split and joined by one move a word; eight
  or sixteen bytes, by shifts and masks). A load marked as a staged copy defines no name: it loads again into the
  registers its first load made.
- **Atomic adds and the delegated zeroing.** An atomic `Write` is one `atom.add` at the element's type (`f32`,
  `i32`, `u32`, `i64`). A `ZeroPrologue` is a loop of 32-bit stores from the thread's index in the grid by the
  grid's thread count, both held to the word count so the index stays in 32 bits.
- **Async copies.** `CpAsyncCopy` is one `cp.async` of 4, 8 or 16 bytes between a 32-bit shared address and a 64-bit
  global one; under `valid` a 16-byte copy takes its source size from the condition, so it writes zeros where the
  condition fails. A swizzled or lane-indexed destination has no rule yet.
- **Carried states.** A cell of a carried state at a literal index is two registers: the one a `Pre` copies and the
  one a `Carry` moves into. Both are seeded ahead of the loop that carries the state, and the end of each trip moves
  the second into the first, so a cell a trip skips keeps its value. A cell at an index known only where the kernel
  runs needs storage and has no rule.

## Math

A math function is a sequence of f32 instructions written through the builder, so the value table shares what two
calls compute alike. Every instruction that rounds carries `.rn`, a fused multiply-add stands only where the sequence
writes one, and no sequence reads the kernel's fusion decision: its accuracy is measured with the instructions it
has. The flush-to-zero decision reaches every instruction of a sequence that takes the modifier. A half value has no
sequence: its function computes in f32 and converts once, which the body says with its casts, and a math function
handed a half operand is refused.

Only the functions kernels hold are written. Counted over the realization corpus and the repository goldens (5,930
kernel bodies): `rsqrt` 8,739 calls, `exp` 7,510, `log` 551, `tanh` 72, `pow` 28 (one golden), `erf` 8. No kernel
holds a sine or a cosine. `sqrt`, `divide` and `reciprocal` are one correctly rounded instruction each. `sigmoid`
(1,186 calls) is a composite: a kernel pass writes it as its primitives.

The target of each sequence is the bound NVIDIA's CUDA math documentation states for the same single-precision
function, with the special values the CUDA function gives. The polynomial coefficients are minimax fits made for
these sequences, each the function it is fit to within a sixth of an f32's step; the reductions are the textbook
ones.

- **`exp`.** The exponent `k` is the floor of `x * log2(e)`, held to the exponents a normal float has. The rest of
  the product, taken with the rest of the constant by two fused multiply-adds, lies in `[0, 1)`, where the
  hardware's power of two (`ex2.approx`) is most accurate, and `2^k` is written through its bits. Past either end
  `k` stays at its bound and the hardware instruction carries the result to infinity, through the subnormals, to
  zero.
- **`log`.** A subnormal is scaled into the normals. The float's bits less those of `2/3` hold the exponent `k`, and
  what is left is a mantissa `m` in `[2/3, 4/3)`, so `ln(m)` is near zero only where `k` is zero. The result is
  `k * ln(2) + ln(1 + f)` with `f = m - 1` exact and a polynomial of degree 10. Zero, a negative, an infinity and a
  NaN take the hardware logarithm's value, which is the right one for each; one unsigned comparison finds them.
- **`tanh`.** Below 1 an odd polynomial of degree 15. From 1 on, `1 - 2 / (2^(2|x| * log2(e)) + 1)` with the sign of
  `x`, through the hardware's power of two and reciprocal. The hardware's own `tanh.approx` measures 134 ulp off
  and is not used.
- **`erf`.** Below 1 an odd polynomial of degree 13. From 1 on, `1 - 2^G(|x|)` with the sign of `x`: `G` is a
  polynomial of degree 7 fit to `log2(erfc)`, with the error weighted by what it moves of the result.
- **`rsqrt`** is the hardware instruction (`rsqrt.approx`), as CUDA's `rsqrtf` is.
- **`pow`** is `2^(y * log2|x|)`, with the logarithm and the product each held as a pair of floats whose sum is the
  value: the product reaches 150 and its error is the result's relative error. With a mantissa `m` from the square
  root of a half to the square root of two and `f = m - 1`, `log2(m)` is a series in `u = 2f / (2 + f)`. The quotient
  goes through the hardware's reciprocal, and what that leaves of `u` is computed back from `2f - u * (2 + f)`, so
  the reciprocal's own error drops out. The product less its nearest integer goes through a polynomial of degree 6,
  and the power of two is applied in two factors, since it runs past both ends of a normal float's exponent. Selects
  then give what the operands' kinds decide, as C's `pow` has it: a zero or an infinite base, a negative power for a
  negative base under an odd exponent, a NaN for one under a fractional exponent, and one for a zero exponent or a
  base of one whatever the other operand is.

Measured on an RTX 5090 over every f32 bit pattern, against a float64 reference on the host; `pow` over 1.3 billion
sampled pairs (below). The error is the
distance from the exact value in steps of the f32 grid there; the documentation counts steps from the correctly
rounded result, which is never the larger number. The instruction counts are machine instructions of the sequence
(`nvdisasm`, the kernel less the same kernel without the function).

| Function | CUDA's bound | Maximum error | Inputs over 1 ulp | Furthest from nvcc's build | Instructions, nvcc's |
| --- | --- | --- | --- | --- | --- |
| `exp` | 2 ulp | 1.931 ulp at -13.899315 | 14,299,457 | 2 steps | 11, 10 |
| `log` | 1 ulp | 0.931 ulp at 2.6688592 | 0 | 1 step | 24, 26 |
| `tanh` | 2 ulp | 1.367 ulp at 1.0126345 | 5,604 | 2 steps | 18, 17 |
| `erf` | 2 ulp | 1.166 ulp at 1.0339911 | 788 | 2 steps | 21, 28 |
| `rsqrt` | 2 ulp | 1.515 ulp at 8.640916e37 | 10,159,988 | the same bits | 4, 4 |
| `pow` | 4 ulp | 2.613 ulp at (1.3825005, 269.64386) | sampled | 3 steps | 90, 94 |

The pairs sampled for `pow`, 268 million of each kind: a mantissa at the fold of the logarithm's range under a
product near the ends of the exponent's range, where the error is largest (2.613 ulp, 0.9% of the pairs over 1
ulp); bases of every magnitude under products from one end to the other (2.215); random bit patterns (1.567); bases
next to one under large exponents (1.744); signed bases under small integer exponents (1.001). Over every base of one
exponent the maximum is 0.90 to 0.96 ulp for 0.5, 2, 3, -1, 1/3 and 2.5, 1.291 for 100 and 2.253 for -227.

nvcc's own builds measure 1.931, 0.864, 1.815 and 1.037 ulp for the first four, and 2.366 for `pow` on the same pairs.
Every special value is the CUDA function's: zeros keep their signs through `tanh` and `erf`, `exp` goes through the
subnormals to zero, `log` of zero is negative infinity and of a negative a NaN, and a NaN operand gives a NaN. `pow`
gives what C's `pow` gives on every pair of a grid of 67 special values, as nvcc's build does.

### The approximate operations

An approximate operation is one hardware instruction, on its operand as it stands or scaled: what `--use_fast_math`
and the fast intrinsics give a CUDA kernel today. One table in the math module holds them, a row each. Only
`exp_fast` is an operation the IR declares; the other rows stand under provisional names, matched by spelling, until
the pass that picks the approximate forms declares them.

| Operation | Instructions |
| --- | --- |
| `exp_fast` | `ex2.approx(x * log2(e))` |
| `log_fast` | `lg2.approx(x) * ln(2)` |
| `pow_fast` | `ex2.approx(y * lg2.approx(x))` |
| `sin_fast`, `cos_fast` | `sin.approx`, `cos.approx` |
| `tanh_fast` | `tanh.approx`, from `sm_75` |
| `divide_fast`, `reciprocal_fast` | `div.approx`, `rcp.approx` |
| `sqrt_fast`, `rsqrt_fast` | `sqrt.approx`, `rsqrt.approx` |

Each flushes to zero when the kernel's decision says (`tanh.approx` has no such form). With the decision off, the
instruction is written in the form that keeps subnormals, and ptxas itself writes the shift that needs: it halves
the operand of `ex2.approx` below the normal range and squares the result, three more machine instructions.
`exp_fast` gives the bits nvcc's `__expf` gives on every f32 input, in both forms. Its error grows with the operand
(the product is rounded before the power): the documentation states `2 + floor(|1.173 * x|)` ulp.

An accurate sequence uses the flushing form of a hardware instruction inside it: none hands one a subnormal whose
value matters or reads a subnormal from one.

## Cooperation between threads

- **The combine program.** `WarpShuffle` and `TreeHalve` carry a program of statements over their state names. It
  goes through the same statement rules: a statement that defines a state steps that state's register in place (the
  instruction that computes the value defines the state itself where no other reader keeps that value, else a move
  does), any other is a temporary.
- **Shuffles.** A shuffle moves 32 bits, so an `f32` goes through its bits and a half through a 32-bit register. The
  mask is read once per statement (`activemask`) and every shuffle of the statement waits for those lanes. A
  butterfly halves its stride. A prefix doubles it, reads the lane below within its group, and steps the state under
  a predicate guard and not a branch, so the warp does not split between steps. `WarpBroadcast` reads each group's
  last lane.
- **The tree reduction.** `TreeHalve` reduces a shared slab into its first slot. The rule first finds who writes
  each slot, from the slot index of every thread of a block (below). Where each thread has a slot of its own, a
  halving step is a condition over the low half, which reads its slot and the one a stride above, combines and
  writes its own, then a barrier. Where the lanes of a warp share a slot, that form would have every lane of a warp
  read, combine and write one slot, which is a race. There the block's first lanes, as many as there are slots, take
  one slot each into registers, fold them by a butterfly under the mask of exactly those lanes, and the first writes
  the result. The CUDA renderer folds in the whole first warp, so several lanes read the slot the first lane then
  writes; compute-sanitizer reports that as a write-after-read hazard. Any other sharing, and a slot the thread
  index does not decide, is refused.
- **What is decided of a block's threads.** The lowering remembers each integer or bool register whose value is an
  expression over the thread index alone, through the names that define it. Such an expression is evaluated for
  every thread of a block. From that it keeps, where a statement stands, which threads may reach it (every condition
  around it that is such an expression), which have left the kernel (the last condition of the grid's body, its
  mask, turns threads away for good), and whether a loop around it runs a trip count that differs between threads.
  A condition that reads anything else, the block's index, a size, a loop's axis, a loaded value, a state, is not
  decided, and decides nothing.
- **Barriers.** `Sync` is `bar.sync`, or `bar.warp.sync` over every lane. A barrier of the whole block is refused
  when a thread that is still in the kernel is decided not to reach it, or when it stands in a loop of differing
  trips. A named barrier is refused when its count is not the threads decided to reach it (checked only when every
  condition around it is decided) or not whole warps, which ptxas requires. A barrier under undecided control flow
  is written as it stands: compute-sanitizer's barrier check on a card is the check there.

## The verifier

`verify` runs at the end of every `lower`, for what ptxas does not check. It splits the list into blocks at labels and
branches and requires every register an instruction reads to be defined on every path from the entry; a definition
under a guard counts for no path. It requires every label a branch names to be bound, once. It requires an alignment
fact behind every access of several elements: a load or store of several registers, and both addresses of an async
copy, need the base register proven aligned to the whole access and the constant offset a multiple of it. The fact
starts at what the interface states of a buffer and a `Smem` of a shared array, so a vector access of a shared array
aligned only to its elements is refused. A register read undefined assembles without complaint and can give a right
answer at one optimization level and a wrong one at another, which is why the first check exists.

## Tests

`cargo test -p emmy-ptx` runs without a GPU. The rule tests build bodies from the text form, compare the instruction
list, and assemble the printed kernel with ptxas at `-O0` and `-O3` wherever ptxas is on the path, so every
instruction form a rule writes has been through the assembler. `tests/compiler/ptx/` launches compiled kernels on a
card, each assembled at ptxas's default level and at `-O0`, and compares them with NumPy: bit for bit, with the fold
replayed in the kernel's order, except an atomic sum of arbitrary floats. The cooperative kernels are also the ones
to run under compute-sanitizer's race and barrier checks.

The math functions are measured there too (`test_math.py`). The default tests hold each function to its documented
bound over a sample: every special value, the neighbourhoods of the points where a sequence changes form, and two
million random bit patterns. The tests marked `sweep` (`pytest -m sweep`) put every f32 bit pattern through the kernel
and hold the maximum of the table above, for `pow` as every base of each of eight exponents; one function takes half a
minute. The crate's own tests check each table of coefficients against the function on the host.
