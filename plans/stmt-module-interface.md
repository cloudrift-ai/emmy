# The statement module: what is left before a Rust port

Goal: `emmy/compiler/ir/stmt` replaceable by a Rust implementation behind its interface. The interface work is done:
one package with a pinned export list, rendering and the body-carrying op outside, one statement protocol, a copy
method, guards and a protocol test (`emmy/compiler/ir/ARCHITECTURE.md`, "The statement module's interface"). This
note holds what that work left open and what a port has to know. Written 2026-11-05.

## Why a port

Body normalization and exact identity are 68% of the placement walk that dominates `make test` (the prior
reproduction gate is 62% of the suite's CPU time, the RTX 5090 placement half alone 15,850 s). Normalization is
linear, about 0.6 ms per statement from 10 to 800 statements, and no function holds more than 9% of the self time.
That is a constant factor, which a Rust port removes and no Python fix does.

## Left open

1. **Tests through the interface.** The tests under `tests/compiler/ir/stmt/implementation/` call inner steps of
   normalization (`place_values`, `prepare_body`, `rename_ssa_sequential`, value numbering and others). Where a
   step's effect shows in the normal form or the identity, rewrite the test to state a body and its normal form
   through `normalize_body`, and move it up one directory. What stays in that directory leaves with the
   implementation. Until then the conformance suite is thinner than it could be.
2. **Body assembly is still inside.** Fusion's assembly (the splicer, the body builder and the body analysis)
   could not move beside its caller: normalization expands subroutine calls with the same engine. Six exported
   names exist for it. Separating subroutine expansion from fusion's assembly would take them off the
   interface and out of the port.
3. **Field listing on statements.** The loopify pass anti-unifies whole statements by listing their dataclass
   fields, and `canonicalize_identity` lists fields to map elementwise ops. Both need what a port can offer instead:
   the wire form, or a protocol member.
4. **`Body` as a tuple.** Callers no longer rely on it, bar two generic walkers: the structural form and the wire
   rewrite both reach a `Body` through `isinstance(..., tuple)`. With `Body` made a plain sequence, those two were
   all that `tests/compiler/ir` and `tests/compiler/passes` needed changed.
5. **Kernel IR's kinds and expressions.** Five Kernel IR kinds report expressions and cannot be rebuilt from them;
   the base refuses by name. Six more hold expressions `exprs` does not report (`CpAsyncCopy`, `TmaLoad` and the
   four mbarrier kinds), so a map would skip them. Nothing maps expressions over a Kernel IR body today.
6. **Statement copies no test reaches.** The guard against `dataclasses.replace` on a statement watches a lowering.
   A call site that no corpus case and no pass or IR test executes is not found by it.

## Notes for the port

- **Field types.** Names, dtypes, floats and booleans are plain values. Three need care: an `Axis` carries a window
  whose parent chain is walked by object identity, and equality deliberately ignores what the structural form
  includes; `Call.target` is a `Subroutine` compared by object identity, with its own pickling; `Assign.op` is an
  `ElementwiseImpl` holding a NumPy callable, portable as its name.
- **The expression boundary.** What the module needs from an expression is its free variables, substitution,
  renaming, simplification, the structural form and the wire form. That list is the expression surface a port
  must provide. The layering guard lists every module the statement module imports.
- **Caches.** Analysis results are cached on the `Body` instance, and the Tile layer stores memos on objects through
  `structural.instance_memo`. A Rust-backed `Body` must hold its own caches and accept that slot.
- **Object identity.** About 53 call sites key side tables on `id(statement)`. A statement read from a body twice
  must be the same object; a test pins it.
- **Kernel IR's kinds** stay Python classes implementing the protocol. To a Rust core they are leaves it calls back
  into through the protocol's methods. Two of them, `RegStore` and `Tile`, still keep part of their rendering as
  methods the render functions call.
- **Conversion cost** across the Python/Rust boundary is unmeasured. Measure it on the heaviest RTX 5090 placement
  pool before committing to a design.

## Not in this plan

- Any change to what normalization does or to the normal form.
- The expression module's own interface, beyond the list above.
