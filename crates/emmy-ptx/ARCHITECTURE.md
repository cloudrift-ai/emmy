# The Kernel IR to PTX compiler

`emmy-ptx` translates one kernel, a Kernel IR body with its interface, into PTX text that ptxas assembles. It reads
the statements of the `emmy-ir` crate as they are, so nothing is serialized on the way. It is a translation: it makes
no decision. A kernel's two whole-kernel precision decisions, multiply-add fusion and flush to zero, reach it as
input, and every other choice is in the Kernel IR before it runs. ptxas keeps register allocation, instruction
scheduling and the unrolling of small loops.

The crate covers scalar kernels today: the grid, loops, conditions, loads, stores, assigns, selects, accumulators and
seeded states, casts, and the elementwise operations that are one instruction each. Shared memory, barriers, vector
access, the math functions and the tensor-core statements have no rule yet, and a kernel that holds one is an error
that names it.

## Entry points

- `compile(kernel, target, precision)` returns the PTX text. `lower` returns the instruction list it prints, which is
  what the crate's tests read.
- `Kernel` is the entry name, the body and the `Interface`: the parameters in the launch's argument order, the
  per-block thread bound, the shared-memory pool. `Interface` and `Param` mirror the Python
  `ir/kernel/interface.py`, which computes them.
- The `emmy.emmy_ir` extension exports `compile_ptx`, so a body crosses from Python as the crate's own value.

## What the compiler expects of a body

The body is typed and flat. Every value-defining statement carries its dtype, every conversion is a cast in the body,
and a `Load` or a `Write` holds one element index into its buffer. No statement carries a request: a loop flagged
`unroll` is refused, since unrolling is a kernel pass. The compiler checks what it reads, statement by statement, and
refuses with the statement's text: a value with no dtype, an operand of another type than its reader, an index of
several coordinates, a store into an input. Integers are the one exception to "every conversion is a cast": their
width and signedness are the compiler's, so `i32`, `u32` and `i64` values mix freely in an index.

A literal has no type of its own. It takes its reader's: an integer literal in a float expression is that float.

## The instruction list

A kernel between the rules and the printer is a `Module`: the directives, the parameter and shared declarations, the
register count of each bank, and one flat list of instructions, labels and comments. An instruction is an opcode, the
type it computes at, a rounding mode, the flush-to-zero flag, the registers it defines, its operands and an optional
predicate guard. Opcodes, types, comparisons, rounding modes and state spaces are enums; no instruction is a string
before `print`. An operand is a register (bank and number), an integer or float constant, an address (a base and a
constant byte offset), a register tuple, a special register or a label.

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
  aligned to. Ranges are not kept per register: they come from the `emmy-ir` range analysis on the expression being
  lowered, under the same context body optimization uses (`launch_ctx` for a grid's built-ins, `inner_ctx` for the
  axes a statement binds). An integer name's range joins that context where the name is bound.

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
  which floor whatever the sign. By another constant, over a dividend the ranges prove non-negative and below 2^31,
  the quotient is a multiply-high and a shift. Anything else is the divide instruction, lowered by one where the
  remainder is nonzero and its sign is not the divisor's, unless the ranges prove both sides non-negative.
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

## The verifier

`verify` runs at the end of every `lower`, for what ptxas does not check. It splits the list into blocks at labels and
branches and requires every register an instruction reads to be defined on every path from the entry; a definition
under a guard counts for no path. It requires every label a branch names to be bound, once. It requires an alignment
fact behind every vector access: a load or store of several registers needs its base register proven aligned to the
whole access, and its constant offset a multiple of it. A register read undefined assembles without complaint and can
give a right answer at one optimization level and a wrong one at another, which is why the first check exists.

## Tests

`cargo test -p emmy-ptx` runs without a GPU. The rule tests build bodies from the text form, compare the instruction
list, and assemble the printed kernel with ptxas at `-O0` and `-O3` wherever ptxas is on the path, so every
instruction form a rule writes has been through the assembler. `tests/compiler/ptx/` launches compiled kernels on a
card and compares them with NumPy.
