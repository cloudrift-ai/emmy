"""The statements of one scope, read by their ordering constraints rather than their positions.

A kernel body is in normal form, so the order of sibling statements is the canonical one and tells a pass nothing
about which statements it may bring together: the two halves of a load pair can stand a whole fragment chain apart.
A pass that fuses statements asks the scope instead. Two statements can become one when no path of ordering
constraints (:func:`~emmy.compiler.ir.stmt.ordering_constraints`) runs between them; the fused scope is written back
in an order that keeps every constraint, and the normal form the next ``KernelOp`` takes orders it from there.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable

from emmy.compiler.ir.stmt import Body, Statement, ordering_constraints


class Siblings:
    """One scope's statements and the constraints between them. A statement is named by the position it had when
    the scope was read; a fused statement keeps its first member's."""

    def __init__(self, stmts: Iterable[Statement]) -> None:
        self._stmts: dict[int, Statement] = dict(enumerate(stmts))
        self._before: dict[int, set[int]] = dict(enumerate(map(set, ordering_constraints(Body(tuple(self._stmts.values())), effects=True))))
        self._reach: dict[int, int] | None = None

    def __getitem__(self, position: int) -> Statement:
        return self._stmts[position]

    def positions(self) -> list[int]:
        return sorted(self._stmts)

    def independent(self, first: int, second: int) -> bool:
        """No constraint, direct or through other statements, orders the two."""
        reach = self._reaching()
        return not (reach[first] >> second & 1 or reach[second] >> first & 1)

    def fuse(self, members: list[int], fused: Statement) -> None:
        """Replace ``members`` (pairwise :meth:`independent` when taken one after another) with ``fused``."""
        kept, gone = members[0], set(members[1:])
        self._stmts[kept] = fused
        for position in gone:
            self._before[kept] |= self._before.pop(position)
            del self._stmts[position]
        self._before[kept] -= gone | {kept}
        for sources in self._before.values():
            if sources & gone:
                sources -= gone
                sources.add(kept)
        self._before[kept].discard(kept)
        self._reach = None

    def ordered(self) -> list[Statement]:
        """The statements in the order closest to the one they were read in that keeps every constraint."""
        waiting = {position: len(sources) for position, sources in self._before.items()}
        after: dict[int, list[int]] = {}
        for position, sources in self._before.items():
            for source in sources:
                after.setdefault(source, []).append(position)
        ready = [position for position, count in waiting.items() if not count]
        heapq.heapify(ready)
        out: list[Statement] = []
        while ready:
            position = heapq.heappop(ready)
            out.append(self._stmts[position])
            for follower in after.get(position, ()):
                waiting[follower] -= 1
                if not waiting[follower]:
                    heapq.heappush(ready, follower)
        assert len(out) == len(self._stmts), "fused statements were ordered by a constraint"
        return out

    def _reaching(self) -> dict[int, int]:
        """Per statement, the bit set of every statement a path of constraints leads from."""
        if self._reach is None:
            reach: dict[int, int] = {}

            def visit(position: int) -> int:
                if position not in reach:
                    reach[position] = 0  # constraints are acyclic; the entry only guards re-entry
                    bits = 0
                    for source in self._before[position]:
                        bits |= 1 << source | visit(source)
                    reach[position] = bits
                return reach[position]

            for position in self._stmts:
                visit(position)
            self._reach = reach
        return self._reach
