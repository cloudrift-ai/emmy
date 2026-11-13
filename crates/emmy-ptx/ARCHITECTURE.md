# The Kernel IR to PTX compiler

`emmy-ptx` translates one kernel, a Kernel IR body with its interface, into PTX text that ptxas assembles. It reads
the statements of the `emmy-ir` crate as they are, so nothing is serialized on the way. It is a translation: it makes
no decision. A kernel's two whole-kernel precision decisions, multiply-add fusion and flush to zero, reach it as
input, and every other choice is in the Kernel IR before it runs. ptxas keeps register allocation, instruction
scheduling and the unrolling of small loops.

The crate covers scalar and cooperative kernels, and of tensor-core kernels every statement that is one instruction
with its operands. Scalar: the grid, loops, conditions, loads, stores, assigns, selects, accumulators and seeded
states, carried states, casts, the elementwise operations that are one instruction each, and the math functions
kernels hold (below). Cooperative: shared memory, barriers, warp shuffles, the tree reduction, vector access, atomic
adds, async copies and the zeroing one kernel does for another. Tensor cores: fragments as registers, `ldmatrix`,
`mma.sync` in every form, the promote and the repacks that stay in registers, block scales, and Hopper's `wgmma`,
tensor copy, `mbarrier` and register limit.

A fragment read from or written to memory element by element has no rule yet: a fragment load from global memory
(`LdmatrixLoad` with `staged` off), the staged drains that are no `ldmatrix` (Volta's layout, a slab of bytes),
`RegStore`, `FragmentApply` and `FragmentRowReduce`. Their instructions follow from the lane-to-element table, which
the crate does not hold yet, and a kernel that holds one is an error that names it and says so. So is any math
function outside the table below.

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
one. A buffer of fp8 elements, of packed fp4 pairs or of 64-bit barriers is declared and addressed, and read by the
statements that take its bits as they are (a block-scale load, an `mbarrier` instruction); a plain load or store of
one is refused, since no value rule decodes it.

A fragment is a local buffer of registers. `RegFragment` declares it, the tensor-core statements read and define its
registers in place, and a `Load` or a `Write` that names it reaches one register by a constant index: `f32 x = load
c[2]` copies register 2 of the accumulator `c`, and `u32 a[0] = w` moves a packed word into register 0 of the operand
`a`. That is how a body moves a value between a fragment and plain code until the fragment statements are compiled,
and what their expansion will write.

A name some statement updates in place, the state of a warp shuffle or a tree reduction, the registers of a staged
copy, is found before lowering starts and is a state from where it is defined.

## The instruction list

A kernel between the rules and the printer is a `Module`: the directives, the parameter and shared declarations, the
register count of each bank, and one flat list of instructions, labels and comments. An instruction is an opcode, the
type it computes at, a rounding mode, the flush-to-zero flag, the registers it defines, its operands and an optional
predicate guard. Opcodes, types, comparisons, rounding modes, state spaces, shuffle modes and cache modifiers are
enums; no instruction is a string before `print`. The hardware's approximate form of an instruction (`ex2.approx`,
`rcp.approx`) is spelled where its rounding mode would stand. An operand is a register (bank and number), an integer
or float constant, an address (a base and a constant byte offset), a register tuple, a tuple of integer constants,
a box of a tensor (the register of its descriptor's address and its coordinates), the address of a shared
declaration, a special register or a label.

An instruction with several type and shape modifiers holds them in its opcode: `mma.sync` is its tile, B's layout, the
operand type, the accumulator type and whether blocks are scaled; `wgmma` its width and operand type; `ldmatrix`
whether it transposes, with the matrix count read from the registers it defines. The types an instruction names but
no register holds alone are types too: the packed pairs `f16x2` and `bf16x2` a conversion writes into 32 bits, and
`e4m3`, `e5m2`, `e2m1` and `ue4m3`.

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
  given. The table shares what the compiler itself makes (address parts, widenings, constants in registers, a lane's
  offset in a tile); it never moves a Kernel IR value or a load, and a load is never shared.
- **Waiting.** `repeat_until` runs one instruction again until the predicate it defines holds: the one loop with no
  body, a wait on a barrier.
- **Facts.** The builder keeps, per register, a number its value is a multiple of and, for an address, the bytes it is
  aligned to. A parameter's pointer starts at the alignment its `Param` states, a shared buffer's address at its
  declaration's, and each term added to an address keeps what the term's multiple allows. A bitwise operation keeps
  the low zero bits its operands share, so an index stays a multiple of eight through a swizzle. Ranges are not kept per
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

## Tensor cores

- **Fragments.** A fragment of n registers is n registers of the instruction list: 32 bits of packed elements each
  for an operand or an accumulator of halves, a float each for an accumulator of floats. Each is a state, since every
  product defines the accumulator again and every load an operand. An accumulator starts at zero; an operand holds
  nothing until a statement loads it, so the verifier refuses a product that reads one no path loaded. The register
  count is the statement's, or follows from the tile and the element size where it names none. A declaration of no
  registers is one register by the fragment's own name: a lane's block scales. A declaration of several fragments
  (`count`) asks for a loop over them and is refused.
- **`ldmatrix`.** Each lane gives the address of one row of eight halves in a shared slab. A is four matrices: lane
  `l` reads row `l % 16` at the column block `l / 16`. B from a slab with K along its rows is two transposed matrices
  of 16 rows; from a slab with N along its rows (a transposed B), two plain ones of eight rows at the column block
  `l / 8 % 2`. Two B fragments side by side in their slab are one load of four matrices. The lane's offset is an
  integer expression over the thread index like any other, so the value table computes it once a kernel.
- **The swizzled address.** A swizzled slab holds its 16-byte chunks exchanged: element `e` stands at
  `e ^ (((e >> shift) & mask) << 3)`, the mask 7, 3 or 1 by the mode and the shift 6 unless the mode states one
  (`B128@7`). The exchange is linear over parts that share no bit. Where the tile's row and column are proven
  multiples of what a lane's offset spans, the lane's part is swizzled alone, once for every load, the column's part
  is a constant, and the row is added after; any other tile swizzles its whole index. Volta's two layouts are no
  exchange of chunks and have no rule.
- **`mma.sync`.** One instruction a statement: the accumulator's registers defined again from A's, B's and their own.

  | Tile | Operands | Accumulator | Registers A, B, C | Target, PTX |
  | --- | --- | --- | --- | --- |
  | m8n8k4, B by columns or by rows | `f16` | `f32` or `f16` | 2, 2, 8 or 4 | `sm_70`, 6.4 |
  | m16n8k16 | `f16` | `f32` or `f16` | 4, 2, 4 or 2 | `sm_80`, 7.0 |
  | m16n8k16 | `bf16` | `f32` | 4, 2, 4 | `sm_80`, 7.0 |
  | m16n8k32 | `e4m3` or `e5m2` | `f32` | 4, 2, 4 | `sm_89`, 8.4 |
  | m16n8k64, block-scaled | `e2m1` | `f32` | 4, 2, 4, a scale register a side | `sm_120a`, 8.7 |

  The block-scaled form takes, after the accumulator, each side's register of scales with the byte it starts at and
  the selector of the lanes that supply it: 0 for A, 1 for B. ptxas takes 0 or 1 there, not the 0 to 3 the PTX
  reference states.
- **Block scales.** `BlockScaleLoad` gives a lane the four scales of its row of A or its column of B, a byte each, as
  one register. Only the lanes the selector names supply one (for A those whose place in their group of four is 0 or
  1, the second eight rows on; for B those whose place is 1); every other lane holds zero and reads nothing. The four
  bytes are loaded one by one, as the CUDA helper does: one 32-bit load would need an alignment no fact gives.
- **The promote.** An accumulator of halves is folded into its accumulator of floats and zeroed: each register split
  into its halves, each widened and added. On Volta a lane's halves are one row of the tile and its floats two, so
  each lane first takes the pair from the lane that holds it, by a shuffle of every lane of the warp.
- **The repacks.** An accumulator of floats becomes an operand of the next product without leaving registers. In the
  m16n8k16 layout a lane's floats stand where A's halves do, so two accumulators side by side along K are one A
  fragment by a packed conversion a register (`cvt.rn.f16x2.f32`, `sm_80`). B wants two rows of a column where a lane
  holds two columns of a row, so each lane takes the packed pairs of the two lanes that hold its column and keeps
  its own half of each. On Volta one accumulator covers 16 by 16, and `part` picks four of its columns as an A
  fragment or four of its rows as a B fragment by the same kind of exchange; `sm_70` has no packed conversion, so a
  pair there is two conversions joined by a move.

## Hopper

- **`wgmma`.** A descriptor is 64 bits: the slab's address in shared memory and the two byte offsets between its 8x8
  matrices, each in units of 16 bytes at bits 0, 16 and 32, and the slab's swizzle in the top two. The rule refuses a
  slab with no 16-byte alignment proven, since the low bits would be dropped. `WgmmaMma` is one
  `wgmma.mma_async`: the accumulator is the N / 8 fragments of four floats a warp holds side by side, read and
  defined; A is a descriptor or the four registers of an A fragment; a predicate says whether the product keeps what
  the accumulator held; the transpose bit of each descriptor operand follows the two scales, which are one. The
  fence, the commit and the wait are one instruction each.
- **The tensor copy.** A descriptor of a tensor is built by the host and passed as a pointer parameter, aligned to 64
  bytes; `TmaDescriptor` declares it and writes nothing. `TmaLoad` is one `cp.async.bulk.tensor` of one to five
  coordinates, which the instruction takes innermost first, into a shared slab and against a barrier.
- **`mbarrier`.** A barrier is 64 bits of a shared array. `MbarrierInit` starts it. An arrival, with or without the
  bytes its phase waits for, stands behind `fence.proxy.async`, which orders the block's earlier shared-memory
  accesses before the copy engine's. `MbarrierWait` asks `mbarrier.try_wait.parity` again until the phase is over.
- **The register limit.** `SetMaxNReg` is one `setmaxnreg`, up or down.

## Targets

The module's `.target` is the card's compute capability, with the `a` suffix when the interface says the kernel needs
the arch-specific instruction set. Its `.version` is the target's own (7.8 below `sm_100`, 8.7 from there), raised by
each instruction that needs a later one. Every rule that writes an instruction not every card has states what it
needs, and refuses a target that lacks it with a message naming both. The cards of each row are what ptxas of CUDA
13.0 takes and refuses, and so is every version from 7.8 on; an earlier version is the PTX reference's and never
decides, since no module states less than 7.8:

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

PTX 8.0 names the tensor copy's destination `.shared::cluster` only; the kernels spell `.shared::cta`, so a kernel
with a tensor copy states 8.6.

## The verifier

`verify` runs at the end of every `lower`, for what ptxas does not check. It splits the list into blocks at labels and
branches and requires every register an instruction reads to be defined on every path from the entry; a definition
under a guard counts for no path. It requires every label a branch names to be bound, once. It requires an alignment
fact behind every access of several elements: a load or store of several registers, and both addresses of an async
copy, need the base register proven aligned to the whole access and the constant offset a multiple of it. So do the
16 bytes of each row `ldmatrix` reads and the destination of a tensor copy. The fact
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

The tensor-core tests (`test_tensor_cores.py`, `test_hopper.py`) write each fragment layout by hand in the test body,
as the PTX reference documents it, with `Load` and `Write` on the fragments, and use small integers as operands so a
product is exact and the comparison is bit for bit. Volta's m8n8k4 forms run on later cards too, so they are launched
wherever the suite runs. `wgmma` runs on `sm_90` alone; the tensor copy, `mbarrier` and `setmaxnreg` from there on.

What was assembled and launched, 2026-11-12. Crate tests assemble every form at `-O0` and `-O3` with ptxas of CUDA
13.0: the m16n8k16 forms and `ldmatrix` for `sm_80`, `sm_89`, `sm_90` and `sm_120`; the fp8 forms for `sm_89`,
`sm_90` and `sm_120`; the block-scaled form for `sm_120a`; `wgmma` for `sm_90a`; the tensor copy and the barrier's
arrivals and waits for `sm_90` and `sm_120`, its start for `sm_80` too; `setmaxnreg` for `sm_90a`, `sm_100a` and
`sm_120a`. That ptxas no longer takes `sm_70`: the
Volta forms are listed for `sm_70` and assembled for `sm_80` there, and the eleven Volta kernels of the launch tests
were assembled for `sm_70` at both levels with ptxas of CUDA 12.9 on a V100 host, where they were not launched. Every
launch test passes at both levels on an RTX 5090 (`sm_120`: all but `wgmma`) and on an H100 (`sm_90`, ptxas of CUDA
12.9: all but the block-scaled form). No `sm_80` or `sm_89` card ran them.

One m16n8k16 tile over three steps of K (two `ldmatrix` and one `mma.sync` a step, then four stores) is 38 machine
instructions through this compiler and 40 through nvcc with the CUDA helpers, on `sm_120`: the same three products and
six matrix loads, the loop unrolled and every address folded by ptxas in both.

The math functions are measured there too (`test_math.py`). The default tests hold each function to its documented
bound over a sample: every special value, the neighbourhoods of the points where a sequence changes form, and two
million random bit patterns. The tests marked `sweep` (`pytest -m sweep`) put every f32 bit pattern through the kernel
and hold the maximum of the table above, for `pow` as every base of each of eight exponents; one function takes half a
minute. The crate's own tests check each table of coefficients against the function on the host.
