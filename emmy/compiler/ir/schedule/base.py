"""Generic schedule assignments and compatible enumeration.

Two terms make a schedule enumeration, and the interface names both:

* A :class:`ScheduleProblem` is the problem and the target, factored into :class:`Site`\\ s: one
  per node in composition order and the kernel site last. A site answers ``options`` — the values
  it may take on its own, with no other site in view. That is the SOURCE of every candidate. A
  problem may carry a knob row; a site the row names offers the row's value alone, parsed and
  checked, so nothing is generated to be filtered away later. A site the row leaves free offers
  its catalog.
* A :class:`ScheduleContext` is the immutable prefix ``c``: what earlier sites decided. It owns the
  compatibility between sites and nothing else — no catalog, no restriction. ``extensions`` yields
  the next site's options that compose with the prefix; ``extend`` composes one, or refuses.

The interface deliberately exposes no domain catalog, site order, or schedule family. A context
chooses the smallest useful frontier, and :func:`schedule` lazily composes that frontier. For
example, the classic context groups one node with its incident edges so it can reject mixed
transport and fragment-seam combinations before they create subtrees.

Three invariants make those different granularities one enumeration:

* ``assignment`` is an immutable kernel × node × edge :class:`Schedule`; a non-``None`` kernel
  marks a complete leaf.
* ``extensions`` yields a lazy, context-aware frontier. It may omit picks already proved
  incompatible, but must retain a route to every accepted complete assignment.
* ``extend`` is the authority. It accepts a frontier pick or a complete assignment supplied by a
  caller, returns a new context, and raises :class:`ScheduleRefused` without mutating the prefix.

The generic driver knows only those operations. Repeatedly calling it on the returned contexts is
the lazy enumeration; no schedule-family visitor or product materialization exists beside it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Self

from frozendict import frozendict

from .views import EdgeSite, NodeId


@dataclass(frozen=True)
class Schedule[KernelT, NodeT, EdgeT]:
    """One immutable kernel × node × edge assignment, possibly still incomplete."""

    kernel: KernelT | None
    nodes: Mapping[NodeId, NodeT]
    edges: Mapping[EdgeSite, EdgeT]

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, Mapping) or not isinstance(self.edges, Mapping):
            raise TypeError("schedule node and edge assignments must be mappings")
        if any(type(site) is not int or site < 0 for site in self.nodes):
            raise TypeError("schedule node assignments must use non-negative integer sites")
        if any(
            not isinstance(edge, tuple)
            or len(edge) != 2
            or type(edge[0]) is not int
            or edge[0] < 0
            or type(edge[1]) is not int
            or edge[1] < 0
            for edge in self.edges
        ):
            raise TypeError("schedule edge assignments must use (consumer, operand) sites")
        object.__setattr__(self, "nodes", frozendict(self.nodes))
        object.__setattr__(self, "edges", frozendict(self.edges))


class ScheduleRefused(ValueError):
    """A pick cannot compose with the immutable schedule context."""


class Site[Pick](ABC):
    """One independent factor of a schedule problem: what this site may take on its own.

    ``keys`` are the row keys the site spells. ``options`` is the site's whole candidate set — a
    memoized tuple, so its identity keys the caches a context keeps over it. A site built from a
    problem whose row names its keys offers the row's value alone (parsed, then checked exactly as
    a catalog value would be); otherwise it offers its catalog. Either way the site is the source:
    nothing downstream generates a candidate.
    """

    @property
    @abstractmethod
    def keys(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def options(self) -> tuple[Pick, ...]: ...


class ScheduleProblem[Pick](ABC):
    """``p + t`` and the row, factored into sites.

    ``sites`` lists the node sites in composition order and the kernel site last. ``row`` is the
    knob row installed, empty for a catalog enumeration; ``with_row`` is the same problem with a
    row installed, whose sites offer the row's values where it names them. ``bounds`` are the
    pool's size bound and one descent's work bound, which a search reads before deciding whether
    to walk the pool at all.
    """

    row: Mapping[str, str]

    @property
    @abstractmethod
    def sites(self) -> Sequence[Site[Pick]]: ...

    @property
    @abstractmethod
    def bounds(self) -> tuple[int, int]: ...

    @abstractmethod
    def with_row(self, row: Mapping[str, str]) -> Self: ...


class ScheduleContext[KernelT, NodeT, EdgeT](ABC):
    """One immutable prefix of a compatible enumeration.

    Implementations own frontier granularity, compatibility and validation. They may therefore
    emit one site at a time, a node together with related edges, or one complete schedule when the
    problem's row already identifies it.
    """

    @property
    def problem(self) -> ScheduleProblem | None:
        """The sites this prefix composes over; ``None`` for a family that sources its frontier
        another way, or a context that only validates."""
        return None

    @property
    @abstractmethod
    def assignment(self) -> Schedule[KernelT, NodeT, EdgeT]:
        """The immutable kernel × node × edge assignment prefix decided so far."""

    @abstractmethod
    def extensions(self) -> Iterator[Schedule[KernelT, NodeT, EdgeT]]:
        """Yield the next site's options that compose with this prefix."""

    @abstractmethod
    def extend(self, pick: Schedule[KernelT, NodeT, EdgeT]) -> Self:
        """Compose a partial or complete pick, or raise when it is incompatible."""

    def narrowed(self, row: Mapping[str, str]) -> Self:
        """This prefix over the problem with ``row`` installed. Only an empty prefix can be
        narrowed: the sites change, and a decided site cannot be re-sourced under it."""
        if self.problem is None or self.assignment.nodes or self.assignment.kernel is not None:
            raise ValueError("only an empty schedule prefix can be narrowed to a row")
        return self._with_problem(self.problem.with_row(row))

    def _with_problem(self, problem: ScheduleProblem) -> Self:
        raise NotImplementedError


def schedule[KernelT, NodeT, EdgeT](
    context: ScheduleContext[KernelT, NodeT, EdgeT],
    *,
    recursive: bool = True,
) -> Iterator[ScheduleContext[KernelT, NodeT, EdgeT] | Schedule[KernelT, NodeT, EdgeT]]:
    """Lazily enumerate complete assignments, or one frontier for a generic tree adapter."""
    for pick in context.extensions():
        try:
            child = context.extend(pick)
        except ScheduleRefused:
            continue
        if child.assignment.kernel is not None:
            yield child.assignment
        elif recursive:
            yield from schedule(child)
        else:
            yield child


__all__ = ["Schedule", "ScheduleContext", "ScheduleProblem", "ScheduleRefused", "Site", "schedule"]
