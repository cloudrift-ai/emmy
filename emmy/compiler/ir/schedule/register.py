"""Register storage for an ordered loop of matrix products and elementwise operations.

One warp owns sixteen rows and all columns of each stored value. The row is an independent
state coordinate; contractions may mix columns but never communicate between row owners.
The term remains the ordinary Fold tree, including its lagged buffer reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

from frozendict import frozendict

from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.schedule.base import Schedule, ScheduleContext, ScheduleProblem, ScheduleRefused, Site
from emmy.compiler.ir.schedule.choices import Tile, Work
from emmy.compiler.ir.stmt import Assign, Body, Let, Load, Select

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.pure import Fold
    from emmy.compiler.ir.tile import OutputSpec, TileOp


_ATOMS = tuple(atom for atom in ATOM_REGISTRY.values() if atom.c_to_a_repack and atom.c_to_b_repack)


@dataclass(frozen=True, slots=True)
class RegisterProgram:
    """The rectangular outputs and one state of a loop, derived before scheduling."""

    state: OutputSpec
    outputs: tuple[OutputSpec, ...]
    roots: tuple[Fold, ...]
    batch: tuple[Axis, ...]
    rows: int
    columns: int

    @classmethod
    def from_tile(cls, tile: TileOp) -> RegisterProgram | None:
        if len(tile.place.serial) != 1 or any(not a.extent.is_static for a in tile.axes):
            return None
        from emmy.compiler.ir.tile.ir import loaded_buffers  # noqa: PLC0415

        own = {load.input for load in loaded_buffers(tile.op)} & {s.write.output for s in tile.output_specs}
        states = [s for s in tile.output_specs if s.write.output in own]
        if len(states) != 1:
            return None
        state = states[0]
        # Every output is a matrix under the same batch and time coordinates. Its last
        # coordinate is the independently owned state row (the physical state is transposed).
        time = tile.place.serial[0].name
        extents = {a.name: a.extent.as_static() for a in tile.axes}
        outputs = tuple(s for s in tile.output_specs if s.write.output not in own)
        if not outputs:
            return None
        for spec in (state, *outputs):
            idx = spec.write.index
            if len(idx) < 3 or idx[0] != Var(time) or len(spec.write.values) != 1:
                return None
            if not all(isinstance(e, Var) and e.name in extents for e in idx[-2:]):
                return None
        rows = extents[state.write.index[-1].name]
        columns = extents[state.write.index[-2].name]
        if rows < 1 or columns < 1 or any(extents[s.write.index[-1].name] != rows for s in outputs):
            return None
        if tile.op.axis is not None:
            return None
        lift = tile.op.applied
        roots = tuple(
            replace(tile.op, lift=replace(lift, body=Body(lift.body.backward_cone(s.write.values).members), results=s.write.values))
            for s in (*outputs, state)
        )
        for site in tile.sites:
            node = site.node
            if node.twist is not None or node.observe is not None:
                return None
            if node.axis is not None:
                view = node.as_contraction()
                if view is None or len(node.operands) != 2 or len(node.exposes) != 1 or node.init != (0.0,):
                    return None
                if (view.product.name, view.plus.name) != ("multiply", "add"):
                    return None
            if any(not isinstance(s, (Assign, Load, Let, Select)) for s in node.lift.body):
                return None
        cells = {e.name for spec in (state, *outputs) for e in spec.write.index[-2:]}
        batch_axes = tuple(a for a in tile.place.free if a.name not in cells)
        batch = {a.name for a in batch_axes}
        if any(spec.write.index[1:-2] != state.write.index[1:-2] for spec in outputs) or any(
            e.free_vars() - batch for e in state.write.index[1:-2]
        ):
            return None
        lag = TernaryExpr(BinaryExpr(">", Var(time), Literal(0, "int")), BinaryExpr("-", Var(time), Literal(1, "int")), Literal(0, "int"))

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
                if isinstance(stmt, Load):
                    if not stmt.is_scalar or any(e.free_vars() - {time, row, col} - batch for e in stmt.index):
                        return False
                    if stmt.input == state.write.output and (
                        not resident or stmt.index != (lag, *state.write.index[1:-2], Var(col), Var(row)) or extents[col] != columns
                    ):
                        return False
            return True

        if not all(
            owns(root, spec.write.index[-1].name, spec.write.index[-2].name) for root, spec in zip(roots, (*outputs, state), strict=True)
        ):
            return None
        return cls(state, outputs, roots, batch_axes, rows, columns)


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
        if program is None or p.row.get("STAGE", "d1/reg") != "d1/reg":
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
        # A descent narrows every offered tier with the kernel's whole row, whose other families
        # (a classic row's REDUCE, its site-scoped keys) this tier does not own: they name no leaf
        # here, and the leaf match rejects them. Only a strict row must be this tier's own.
        if strict and set(row) - set(_RegisterSite.keys):
            raise ValueError("register schedule accepts only WORK, TILE and STAGE")
        return replace(self, row=frozendict({key: value for key, value in row.items() if key in _RegisterSite.keys}))


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
