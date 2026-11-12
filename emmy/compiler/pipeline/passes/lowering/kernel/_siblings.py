"""The statements of one scope, read by their ordering constraints rather than their positions.

Where two statements stand tells a pass little about whether it may bring them together: an unrolled body interleaves
each element's load with its arithmetic, and the optimization every kernel body takes can move a value between them.
A pass that fuses statements asks the scope instead. Statements can become one when no other statement must stand between
them under the scope's ordering constraints (:func:`~emmy.compiler.ir.stmt.ordering_constraints`); the fused scope is written back
in an order that keeps every constraint: each statement as early as its constraints allow, in its old order.
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

    def fusable(self, members: Iterable[int]) -> bool:
        """Whether ``members`` can become one statement: no statement outside them must follow one member and precede
        another. Members a constraint orders directly (the writes of one buffer) stay fusable."""
        reach = self._reaching()
        inside = 0
        before = 0
        for position in members:
            inside |= 1 << position
            before |= reach[position]
        between = before & ~inside
        while between:
            low = between & -between
            if reach[low.bit_length() - 1] & inside:
                return False
            between ^= low
        return True

    def fuse(self, members: list[int], fused: Statement) -> None:
        """Replace ``members`` (:meth:`fusable`) with ``fused``, which stands where the earliest of them stood unless
        a constraint puts it later."""
        kept = min(members)
        gone = set(members) - {kept}
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
