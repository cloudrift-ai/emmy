"""The CPU backend's schedule: the decisions its code generator takes on one kernel.

A third family beside the classic and register ones, on the same schedule base, offered by the same scheduling fork
when the compile targets the CPU. So a CPU kernel's choices are pinned, measured and picked the way a CUDA kernel's
are, from rows filed under the CPU's context.

The family has its own knobs: no CUDA knob takes a CPU meaning, so CUDA evidence and priors never see a CPU decision.
It has no choices yet: one leaf, on which the generator's shape rules decide. A choice joins only once a measurement
shows each of its values winning somewhere; a value that always wins is a rule of the generator instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

from frozendict import frozendict

from emmy.compiler.ir.schedule.base import Schedule, ScheduleContext, ScheduleProblem, ScheduleRefused, Site

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.ir.tile import TileOp


@dataclass(frozen=True, slots=True)
class CpuSchedule:
    """The CPU code generator's decisions for one kernel."""


@dataclass(frozen=True)
class _CpuSite(Site):
    problem: CpuProblem
    keys = ()

    @cached_property
    def options(self) -> tuple[Schedule, ...]:
        if set(self.problem.row) - set(self.keys):
            return ()
        return (Schedule(CpuSchedule(), {}, {}),)


@dataclass(frozen=True)
class CpuProblem(ScheduleProblem):
    tile: TileOp
    target: Context | None
    row: frozendict[str, str] = field(default_factory=frozendict)

    def __post_init__(self):
        object.__setattr__(self, "row", frozendict(self.row))

    @cached_property
    def sites(self):
        return (_CpuSite(self),)

    @property
    def bound(self):
        return len(self.sites[0].options)

    def with_row(self, row, *, strict=False):
        if strict and set(row) - set(_CpuSite.keys):
            raise ValueError(f"a CPU schedule takes no knobs but {sorted(_CpuSite.keys)}")
        return replace(self, row=frozendict(row))


@dataclass(frozen=True, slots=True)
class CpuContext(ScheduleContext):
    _problem: CpuProblem
    _schedule: Schedule = field(default_factory=lambda: Schedule(None, {}, {}))

    @property
    def problem(self):
        return self._problem

    @property
    def schedule(self):
        return self._schedule

    def _with_problem(self, problem):
        return CpuContext(problem)

    def extensions(self):
        if self.schedule.kernel is None:
            yield from self.problem.sites[0].options

    def extend(self, pick):
        if self.schedule.kernel is not None or pick.nodes or pick.edges or not isinstance(pick.kernel, CpuSchedule):
            raise ScheduleRefused("not a CPU schedule of this kernel")
        if pick not in self.problem.sites[0].options:
            raise ScheduleRefused("the row leaves this CPU schedule out")
        return replace(self, _schedule=pick)


class CpuCodec:
    """A CPU schedule as its knob row, and back."""

    def __init__(self, context: CpuContext):
        self.context = context

    @staticmethod
    def keys() -> tuple[str, ...]:
        return _CpuSite.keys

    def _encode(self, schedule: Schedule) -> dict[str, str]:
        return {}

    def encode(self, schedule: Schedule) -> dict[str, str]:
        return self._encode(self.context.extend(schedule).schedule)

    def decode(self, row) -> Schedule:
        if set(row) != set(self.keys()):
            raise ValueError(f"a CPU schedule row names exactly {sorted(self.keys())}")
        context = self.context.narrowed(row, strict=True)
        (choice,) = context.extensions()
        return context.extend(choice).schedule

    def delta(self, before, after) -> dict[str, str]:
        return self._encode(after.schedule)


@dataclass(frozen=True, slots=True)
class CpuMaterialization:
    """A CPU kernel keeps its loop body; the generator reads the schedule beside it."""

    def validate(self, schedule, source, *, place, workers):
        if workers is not None or not isinstance(schedule.kernel, CpuSchedule):
            raise ValueError("a CPU kernel takes a CPU schedule and no warp split")


def materialize_cpu(tile: TileOp, schedule: Schedule, knobs: dict) -> TileOp:
    """``tile`` with its CPU schedule: every free axis is the generator's to split over threads."""
    return replace(tile, place=tile.place.on_grid(), schedule=schedule, materialization=CpuMaterialization(), knobs=knobs)
