"""Register storage for an ordered loop of matrix products and elementwise operations.

One warp owns sixteen rows and all columns of each stored value. The row is an independent
state coordinate; contractions may mix columns but never communicate between row owners.
The term remains the ordinary Fold tree, its state a carried one whose reads are carrier reads.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from emmy.compiler.ir.atom import ATOM_REGISTRY
from emmy.compiler.ir.expression import Literal, Var
from emmy.compiler.ir.pure import Fold, Lambda
from emmy.compiler.ir.schedule.base import Schedule
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
        # A free axis outside the carrying loop is neither a cell nor a coordinate the emitter
        # binds; the roll never makes one, and such a kernel keeps the classic schedule.
        if not tile.carries or tile.place.free or any(not a.extent.is_static for a in tile.axes):
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
            shape = tile.outputs[spec.write.output].shape
            batch_index = tuple(e for e, size in zip(idx[1:-2], shape[1:-2], strict=True) if not (e == Literal(0, "int") and size == 1))
            if batch_index != tuple(Var(name) for name in batch_names):
                return None
        lift = carrying.lift
        results = (*(spec.write.values[0] for spec in outputs), lift.results[0])
        roots = tuple(
            Fold(
                operands=carrying.operands,
                lift=Lambda.closing(lift.params[1:], Body(lift.body.backward_cone((value,)).members), (value,)),
            )
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


#: The warp inventories a register schedule's catalog offers.
REGISTER_WORKS = tuple(Work(kind="warp", units=(warps, 1)) for warps in (1, 2, 4))


def register_tiles(program: RegisterProgram, work: Work) -> tuple[Tile, ...]:
    """The fragment geometries the catalog offers ``program``'s stored values under ``work``: one per atom."""
    return tuple(Tile(atom=atom, units=work.units, regs=(1, (program.columns + atom.atom_n - 1) // atom.atom_n), bk=4) for atom in _ATOMS)


def register_accepts(program: RegisterProgram | None, work: Work, plan: Tile, target: Context | None) -> bool:
    """Whether registers can hold ``program``'s state under this warp inventory and fragment geometry."""
    return (
        program is not None
        and work.kind == "warp"
        and work.units[1] == 1
        and work.count <= 32
        and not work.producer
        and plan.atom in _ATOMS
        and plan.units == work.units
        and plan.regs == (1, (program.columns + plan.atom.atom_n - 1) // plan.atom.atom_n)
        and (target is None or plan.atom.available_on(target))
    )


def register_row(schedule: Schedule) -> dict[str, str]:
    """A register schedule as the three values that decide it — the wire form of a scheduled kernel."""
    return {"WORK": schedule.kernel.work.spell(), "TILE": schedule.kernel.tile.spell(), "STAGE": "d1/reg"}


def register_schedule(tile: TileOp, row) -> Schedule:
    """The register schedule ``row`` spells on ``tile`` (inverse of :func:`register_row`)."""
    if set(row) != {"WORK", "TILE", "STAGE"} or row["STAGE"] != "d1/reg":
        raise ValueError("a register schedule is exactly WORK, TILE and STAGE=d1/reg")
    work = Work.parse(row["WORK"])
    choice = RegisterSchedule(work, Tile.parse(row["TILE"], work))
    if not register_accepts(tile.register_program, choice.work, choice.tile, None):
        raise ValueError("row does not select one register schedule")
    return Schedule(choice, {}, {})


@dataclass(frozen=True, slots=True)
class RegisterMaterialization:
    """Register storage preserves the serial axis as a loop inside the kernel."""

    def validate(self, schedule, source, *, place, workers):
        if workers is not None or not place.is_mapped:
            raise ValueError("register storage needs a mapped, uniform warp inventory")
        kernel = schedule.kernel
        if not isinstance(kernel, RegisterSchedule) or schedule.nodes or schedule.edges:
            raise ValueError("register storage takes one kernel choice and no site choice")
        if not register_accepts(source.register_program, kernel.work, kernel.tile, None):
            raise ValueError("register schedule is outside this loop's domain")


def materialize_register(tile: TileOp, schedule: Schedule, knobs: dict) -> TileOp:
    return replace(
        tile,
        place=tile.place.on_grid(),
        schedule=schedule,
        materialization=RegisterMaterialization(),
        knobs=knobs,
    )
