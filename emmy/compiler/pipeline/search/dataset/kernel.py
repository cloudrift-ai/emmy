"""``KernelDef`` — one kernel's definition: the value the ``kernel`` table stores per row and a dataset carries per
golden pool, so a pool can be enumerated from it on any machine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from emmy.compiler.wire import Wire


@dataclass(frozen=True)
class KernelDef(Wire):
    """The exact identity, the clustered deploy identity, the Loop IR wire, the C name, the ``S_*`` stamps, and
    whether the wire is the body the kernel was formed from (``formed``: the lowering passes take it back to the
    kernel, identity and stamps alike) or the derived body of a kernel formed from no loop op, which only its
    parent's program reaches."""

    exact_identity: str
    structural_identity: str
    loop_ir: dict
    name: str
    stamps: dict
    formed: bool

    def program(self, bindings: Mapping[str, int]):
        """The kernel's definition as a program, its symbolic dims hinted at ``bindings`` — the sizes a measurement
        of it ran at — so the program stays symbolic; binding them would make the dims static, another kernel."""
        import emmy.compiler.ir.loop.ir  # noqa: F401, PLC0415 — registers the Loop IR wire classes the body decodes through
        from emmy.compiler.graph import Graph  # noqa: PLC0415 — the graph package imports this module's neighbours
        from emmy.compiler.specialize import rehint_program  # noqa: PLC0415

        return rehint_program(Graph.from_wire(self.loop_ir), bindings)

    def op(self, bindings: Mapping[str, int]):
        """The kernel's Loop op at ``bindings``, carrying its stored stamps — the stamps a kernel formed from no loop
        op keeps are its tile's, which its derived wire body does not reproduce."""
        from emmy.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415

        [op] = [node.op for node in self.program(bindings).nodes.values() if isinstance(node.op, LoopOp)]
        return replace(op, knobs={**op.knobs, **self.stamps})
