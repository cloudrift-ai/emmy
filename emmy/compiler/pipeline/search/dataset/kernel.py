"""``KernelDef`` — one kernel's definition: the value the ``kernel`` table stores per row and a dataset carries per
golden pool, so a pool can be enumerated from it on any machine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property, lru_cache
from types import MappingProxyType

from emmy.compiler.wire import Wire


@lru_cache(maxsize=1)
def _lift():
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415 — the pipeline imports this module's neighbours

    return Pipeline.build(["tile/lift"])


@dataclass(frozen=True)
class KernelDef(Wire):
    """The Loop IR wire, the C name, and whether the wire is the body the kernel was formed from (``formed``: the
    tile lift takes it back to the kernel) or the derived body of a kernel formed from no loop op, which only its
    parent's program reaches. Inputs only: the kernel's identity and its stamps are computed from the wire
    (:attr:`exact_identity`, ``features.stamps`` of :meth:`op`), never stored beside it."""

    loop_ir: dict
    name: str
    formed: bool

    def program(self, bindings: Mapping[str, int]):
        """The kernel's definition as a program, its symbolic dims hinted at ``bindings`` — the sizes a measurement
        of it ran at — so the program stays symbolic; binding them would make the dims static, another kernel."""
        import emmy.compiler.ir.loop.ir  # noqa: F401, PLC0415 — registers the Loop IR wire classes the body decodes through
        from emmy.compiler.graph import Graph  # noqa: PLC0415 — the graph package imports this module's neighbours
        from emmy.compiler.specialize import rehint_program  # noqa: PLC0415

        return rehint_program(Graph.from_wire(self.loop_ir), bindings)

    def op(self, bindings: Mapping[str, int] = MappingProxyType({})):
        """The kernel at ``bindings``, bound to its buffers: the tile the lift takes a formed body to, or — for a
        kernel formed from no loop op — the loop op over its derived body, which has the kernel's identity and
        stamps as it stands."""
        from emmy.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
        from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415
        from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415

        program = self.program(bindings)
        if self.formed:
            program, _ = Run(pipeline=_lift(), ctx=None).resolve(program, lambda fork: next(fork.leaves()))
        [node] = [node for node in program.nodes.values() if isinstance(node.op, TileOp if self.formed else LoopOp)]
        return node.op.with_io(program, node)

    @cached_property
    def exact_identity(self) -> str:
        """The kernel's exact identity (``identity_key(structural=False, with_io=True)``), computed from the wire;
        the sizes are no part of it."""
        return self.op().identity_key(structural=False, with_io=True)

    def keyed(self, identity: str) -> KernelDef:
        """This definition with its identity already known — the key the DB row holding it is stored under, or the
        identity of the live kernel it was written from — so nothing lifts the wire to learn it."""
        self.__dict__["exact_identity"] = identity
        return self
