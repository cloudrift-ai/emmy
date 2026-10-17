"""The CPU backend's schedule: the decisions its code generator takes on one kernel.

A third family beside the classic and register ones, on the same schedule base, offered by the same scheduling fork
when the compile targets the CPU. So a CPU kernel's choices are pinned, measured and picked the way a CUDA kernel's
are, from rows filed under the CPU's context.

The family has its own knobs (``CPU_*``): no CUDA knob takes a CPU meaning, so CUDA evidence and priors never see a
CPU decision. A choice joins only once a measurement shows each of its values winning somewhere; a value that always
wins is a rule of the generator instead.

``CPU_TILE`` is the register tile of a projection: its rows and columns accumulated in one pass of the reduce loop. A
kernel with two free loops and a reduction offers every tile whose accumulators fit half the vector registers (its
accumulators times the tile's cells at most :data:`ACCUMULATOR_BUDGET`); the best differs by kernel — on a Ryzen 9
9950X3D an OLMoE expert's fused gate and up projection was fastest at 2x4 and its down projection at 4x4. The catalog
starts with the largest, squarest tile, the pick where nothing was measured.
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

#: Tile sides on offer.
TILE_SIDES = (1, 2, 4, 8)
#: The most accumulators a tile may keep live: half of AVX-512's 32 vector registers, the other half holding the
#: loaded operands. Past it the accumulators spill.
ACCUMULATOR_BUDGET = 16


@dataclass(frozen=True, slots=True)
class CpuSchedule:
    """The CPU code generator's decisions for one kernel: ``tile`` is (rows, columns) of its register tile."""

    tile: tuple[int, int] = (1, 1)

    def spell(self) -> dict[str, str]:
        rows, columns = self.tile
        return {"CPU_TILE": f"{rows}x{columns}"}

    @staticmethod
    def parse(row) -> CpuSchedule:
        rows, _, columns = row["CPU_TILE"].partition("x")
        return CpuSchedule((int(rows), int(columns)))


def accumulators(tile: TileOp) -> int:
    """The running sums one output of ``tile`` keeps (an expert's fused gate and up projection keeps two), or 0 for a
    kernel that does not reduce."""

    def count(body) -> int:
        return sum(type(s).__name__ == "Accum" or count(getattr(s, "body", None) or ()) for s in body)

    return count(tile.loop_body or ())


def tiles(tile: TileOp) -> tuple[tuple[int, int], ...]:
    """The register tiles ``tile`` offers, largest and squarest first: every tile within the accumulator budget for a
    kernel with two free loops and a reduction, the single cell otherwise."""
    n = accumulators(tile)
    if n == 0 or len(tile.place.free) < 2:
        return ((1, 1),)
    fits = [(r, c) for r in TILE_SIDES for c in TILE_SIDES if r * c * n <= ACCUMULATOR_BUDGET]
    return tuple(sorted(fits, key=lambda rc: (-rc[0] * rc[1], abs(rc[0] - rc[1]), -rc[1])))


@dataclass(frozen=True)
class _CpuSite(Site):
    problem: CpuProblem
    keys = ("CPU_TILE",)

    @cached_property
    def options(self) -> tuple[Schedule, ...]:
        p = self.problem
        if set(p.row) - set(self.keys):
            return ()
        offered = [CpuSchedule(rc) for rc in tiles(p.tile)]
        if "CPU_TILE" in p.row:
            try:
                pinned = CpuSchedule.parse(p.row)
            except ValueError:
                return ()
            offered = [s for s in offered if s == pinned]
        return tuple(Schedule(s, {}, {}) for s in offered)


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
        """The problem narrowed to ``row``. ``self.row`` holds the hand pins the fork was built under: a measured row
        narrows within them and never overwrites them, as in the classic family."""
        if strict and set(row) - set(_CpuSite.keys):
            raise ValueError(f"a CPU schedule takes no knobs but {sorted(_CpuSite.keys)}")
        return replace(self, row=frozendict({**{str(k): str(v) for k, v in row.items()}, **self.row}))


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
        return schedule.kernel.spell()

    def encode(self, schedule: Schedule) -> dict[str, str]:
        return self._encode(self.context.extend(schedule).schedule)

    def decode(self, row) -> Schedule:
        if set(row) != set(self.keys()):
            raise ValueError(f"a CPU schedule row names exactly {sorted(self.keys())}")
        choices = tuple(self.context.narrowed(row, strict=True).extensions())
        if len(choices) != 1:
            raise ValueError("the row does not select one CPU schedule of this kernel")
        return self.context.extend(choices[0]).schedule

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
