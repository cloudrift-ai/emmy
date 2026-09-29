"""Register storage for an ordered loop of matrix products and elementwise operations.

One warp owns sixteen rows and all columns of each stored value. The row is an independent
state coordinate; contractions may mix columns but never communicate between row owners.
The term remains the ordinary Fold tree, its state a carried one whose reads are carrier reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

from frozendict import frozendict

from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.pure import Fold
from emmy.compiler.ir.schedule.base import Schedule, ScheduleContext, ScheduleProblem, ScheduleRefused, Site
from emmy.compiler.ir.schedule.choices import Tile, Work
from emmy.compiler.ir.stmt import Assign, Body, Let, Load, Pre, Select

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.tile import OutputSpec, TileOp


_ATOMS = tuple(atom for atom in ATOM_REGISTRY.values() if atom.c_to_a_repack and atom.c_to_b_repack)


@dataclass(frozen=True, slots=True)
class RegisterProgram:
    """The rectangular outputs and one state of a loop, derived before scheduling.

    Read off the fold that carries the state (:attr:`Fold.carries`): ``time`` is its axis, the
    state's cells end in the column and the independently owned row (the physical state is
    transposed), and every per-step output is a matrix under the same batch and time coordinates
    whose last coordinate is that row. ``roots`` are zero-axis terms over the carrying fold's own
    operands, one per output and the state's next value last, which the emitter evaluates in
    fragments; a read of the state is a carrier read, resident in the warp's own rows."""

    time: str
    state: str
    seed: float | str
    outputs: tuple[OutputSpec, ...]
    roots: tuple[Fold, ...]
    batch: tuple[Axis, ...]
    rows: int
    columns: int

    @classmethod
    def from_tile(cls, tile: TileOp) -> RegisterProgram | None:
        if not tile.carries or any(not a.extent.is_static for a in tile.axes):
            return None
        (carrying,) = (site.node for site in tile.sites if site.node.carries)
        if len(carrying.base.results) != 1 or len(carrying.cells) < 2:
            return None
        time, (state,) = carrying.axis, carrying.base.results
        extents = {a.name: a.extent.as_static() for a in tile.axes}
        outputs = tile.output_specs
        if not outputs:
            return None
        col, row = carrying.cells[-2:]
        batch_names = carrying.cells[:-2]
        rows, columns = extents[row], extents[col]
        if rows < 1 or columns < 1:
            return None
        for spec in outputs:
            # Under the same batch and time coordinates, its last coordinate the state's row — by
            # extent: a sweep beside the cells spells the row under a name of its own.
            idx = spec.write.index
            if len(idx) < 3 or idx[0] != Var(time) or len(spec.write.values) != 1:
                return None
            if not all(isinstance(e, Var) and e.name in extents for e in idx[-2:]) or extents[idx[-1].name] != rows:
                return None
            if idx[1:-2] != tuple(Var(name) for name in batch_names):
                return None
        lift = carrying.lift
        results = (*(spec.write.values[0] for spec in outputs), lift.results[0])
        roots = tuple(
            Fold(operands=carrying.operands, lift=replace(lift, params=lift.params[1:], body=Body(lift.body.backward_cone((value,)).members), results=(value,)))
            for value in results
        )
        for site in tile.sites:
            node = site.node
            if node.twist is not None or node.observe is not None:
                return None
            if node.axis is not None and not node.carries:
                view = node.as_contraction()
                if view is None or len(node.operands) != 2 or len(node.exposes) != 1 or node.init != (0.0,):
                    return None
                if (view.product.name, view.plus.name) != ("multiply", "add"):
                    return None
            if any(not isinstance(s, (Assign, Load, Let, Pre, Select)) for s in node.lift.body):
                return None
        batch_axes = tuple(tile.axis_of(name) for name in batch_names)
        batch = set(batch_names)

        def owns(node, row, col, resident=True):
            if row == col:
                return False
            if node.axis is not None:
                left, right = node.operands
                if row not in left.free_axes:
                    left, right = right, left
                return (
                    row in left.free_axes
                    and row not in right.free_axes
                    and col not in left.free_axes
                    and owns(left, row, node.axis, resident)
                    and owns(right, node.axis, col, False)
                )
            needed = node.lift.body.ssa_uses | set(node.lift.results)
            if any(not owns(edge, row, col, resident) for param, edge, _ in node.bindings if param in needed):
                return False
            for stmt in node.lift.body:
                if isinstance(stmt, Let) and not isinstance(stmt.value, Literal):
                    return False
                if isinstance(stmt, Select) and any(b.select.free_vars() - {time, row, col} - batch for b in stmt.branches):
                    return False
                if isinstance(stmt, Load) and (not stmt.is_scalar or any(e.free_vars() - {time, row, col} - batch for e in stmt.index)):
                    return False
                if isinstance(stmt, Pre) and (
                    not resident or stmt.index != (*(Var(name) for name in batch_names), Var(col), Var(row)) or extents[col] != columns
                ):
                    return False
            return True

        cells = (*((spec.write.index[-1].name, spec.write.index[-2].name) for spec in outputs), (row, col))
        if not all(owns(root, row, col) for root, (row, col) in zip(roots, cells, strict=True)):
            return None
        return cls(time, state, carrying.init[0], outputs, roots, batch_axes, rows, columns)


@dataclass(frozen=True, slots=True)
class RegisterSchedule:
    """A warp inventory and matrix fragment geometry for register storage."""

    work: Work
    tile: Tile


@dataclass(frozen=True)
class _RegisterSite(Site):
    problem: RegisterProblem
    keys = ("WORK", "TILE", "STAGE")

    @cached_property
    def options(self):
        p = self.problem
        program = p.tile.register_program
        if program is None or set(p.row) - set(self.keys) or p.row.get("STAGE", "d1/reg") != "d1/reg":
            return ()
        if not p.allow_f16 and "TILE" not in p.row:
            return ()
        candidates = []
        works = (Work.parse(p.row["WORK"]),) if "WORK" in p.row else tuple(Work(kind="warp", units=(w, 1)) for w in (1, 2, 4))
        for work in works:
            if work.kind != "warp":
                continue
            plans = (
                (Tile.parse(p.row["TILE"], work),)
                if "TILE" in p.row
                else tuple(
                    Tile(atom=atom, units=work.units, regs=(1, (program.columns + atom.atom_n - 1) // atom.atom_n), bk=4) for atom in _ATOMS
                )
            )
            for plan in plans:
                choice = RegisterSchedule(work, plan)
                if p.accepts(choice):
                    candidates.append(Schedule(choice, {}, {}))
        return tuple(candidates)


@dataclass(frozen=True)
class RegisterProblem(ScheduleProblem):
    tile: TileOp
    target: Context | None
    row: frozendict[str, str] = field(default_factory=frozendict)
    allow_f16: bool = False

    def __post_init__(self):
        object.__setattr__(self, "row", frozendict(self.row))

    @cached_property
    def sites(self):
        return (_RegisterSite(self),)

    @property
    def bounds(self):
        count = len(self.sites[0].options)
        return count, count

    def accepts(self, choice: RegisterSchedule) -> bool:
        program = self.tile.register_program
        if program is None or not isinstance(choice, RegisterSchedule):
            return False
        work, plan = choice.work, choice.tile
        return (
            work.kind == "warp"
            and work.units[1] == 1
            and work.count <= 32
            and not work.producer
            and plan.atom in _ATOMS
            and plan.units == work.units
            and plan.regs == (1, (program.columns + plan.atom.atom_n - 1) // plan.atom.atom_n)
            and (self.target is None or plan.atom.available_on(self.target))
        )

    def with_row(self, row, *, strict=False):
        # A descent narrows every offered tier with the kernel's whole row. A row naming a family
        # this tier does not own (a classic row's REDUCE, its site-scoped keys) describes another
        # tier, so this one offers no leaf for it; only a strict row must be this tier's own.
        if strict and set(row) - set(_RegisterSite.keys):
            raise ValueError("register schedule accepts only WORK, TILE and STAGE")
        return replace(self, row=frozendict(row))


@dataclass(frozen=True, slots=True)
class RegisterContext(ScheduleContext):
    _problem: RegisterProblem
    _schedule: Schedule = field(default_factory=lambda: Schedule(None, {}, {}))

    @property
    def problem(self):
        return self._problem

    @property
    def schedule(self):
        return self._schedule

    def _with_problem(self, problem):
        return RegisterContext(problem)

    def extensions(self):
        if self.schedule.kernel is None:
            yield from self.problem.sites[0].options

    def extend(self, pick):
        if (
            self.schedule.kernel is not None
            or pick.nodes
            or pick.edges
            or not self.problem.accepts(pick.kernel)
            or any(RegisterCodec(self)._encode(pick).get(k) != v for k, v in self.problem.row.items())
        ):
            raise ScheduleRefused("register schedule is outside this loop's domain")
        return replace(self, _schedule=pick)


class RegisterCodec:
    def __init__(self, context):
        self.context = context

    def keys(self):
        return _RegisterSite.keys

    def _encode(self, schedule):
        choice = schedule.kernel
        return {"WORK": choice.work.spell(), "TILE": choice.tile.spell(), "STAGE": "d1/reg"}

    def encode(self, schedule):
        return self._encode(self.context.extend(schedule).schedule)

    def decode(self, row):
        if set(row) != set(self.keys()):
            raise ValueError("a register schedule needs exactly WORK, TILE and STAGE")
        context = self.context.narrowed(row, strict=True)
        choices = tuple(context.extensions())
        if len(choices) != 1:
            raise ValueError("row does not select one register schedule")
        return context.extend(choices[0]).schedule

    def delta(self, before, after):
        return self._encode(after.schedule)


@dataclass(frozen=True, slots=True)
class RegisterMaterialization:
    """Register storage preserves the serial axis as a loop inside the kernel."""

    def validate(self, schedule, source, *, place, workers):
        if workers is not None or not place.is_mapped:
            raise ValueError("register storage needs a mapped, uniform warp inventory")
        problem = RegisterProblem(source, None, allow_f16=True)
        RegisterContext(problem).extend(schedule)


def materialize_register(tile: TileOp, schedule: Schedule, knobs: dict) -> TileOp:
    return replace(
        tile,
        place=tile.place.on_grid(),
        schedule=schedule,
        materialization=RegisterMaterialization(),
        knobs=knobs,
    )
