"""One value feeding TWO sibling cones is declared ONCE per scope.

A ``Fold``'s operand edges splice independently (``Fold.spliced_step``), so two cones reading the
same 1-element input each carry their own copy of its ``buf[0]`` ``Load`` — same SSA name, same
``(input, index)``. Flattened into one loop body they become two C declarations of one name, which
nvcc rejects (``"in4" has already been declared in the current scope``, 11 errors on the
DeepSeek-V4 MXFP4 expert kernel, whose decode spells eleven shared scalar constants and applies
them to both halves of a fused ``gate_up`` weight).

The graph below is that shape minimized: one constant applied to a fused weight, sliced into the
two channels of one contraction. Capability-independent — it reproduced identically on sm_70,
sm_80, sm_89 and sm_120.
"""

from __future__ import annotations

import re
from dataclasses import replace
from importlib import import_module

import pytest

from emmy.compiler.context import Context
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.frontend.ir import LinearOp, SliceOp
from emmy.compiler.ir.schedule import Placement, Raster, Reduce, Stage, Tile, Work, derive_inventory
from emmy.compiler.ir.schedule.base import Schedule
from emmy.compiler.ir.schedule.classic import (
    ClassicMaterialization,
    EdgeSchedule,
    KernelSchedule,
    ProjectionSchedule,
    ReductionSchedule,
)
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.target import set_target
from tests.compiler.terms import projection, reduction, slab

materialize = import_module("emmy.compiler.pipeline.passes.lowering.kernel.010_materialize")
factor = import_module("emmy.compiler.pipeline.passes.lowering.kernel._factor")

_CAP = (8, 0)
_M, _K, _N = 1, 32, 16  # M=1 — the decode row, whose contraction folds serially per channel
_DECL = re.compile(r"^\s*(?:const\s+)?(?:float|double|half|__half|int|unsigned|long|bool)\s+(\w+)\s*(?:=|;)")


def _redeclared(source: str) -> list[str]:
    """Names declared twice within ONE brace scope — exactly what nvcc rejects."""
    scopes: list[dict[str, int]] = [{}]
    clashes: list[str] = []
    for line in source.splitlines():
        declared = _DECL.match(line)
        if declared and declared.group(1) in scopes[-1]:
            clashes.append(declared.group(1))
        elif declared:
            scopes[-1][declared.group(1)] = 1
        for char in line:
            if char == "{":
                scopes.append({})
            elif char == "}" and len(scopes) > 1:
                scopes.pop()
    return clashes


def _gate_up_graph() -> Graph:
    """``y = (x @ (w − c)[:N].T) · (x @ (w − c)[N:].T)`` — the fused gate/up shape, ``c`` shared."""
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (_M, _K), dtype=F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("w", (2 * _N, _K), dtype=F16), node_id="w")
    graph.add_node(InputOp(), [], Tensor("c", (1,), dtype=F16), node_id="c")
    graph.add_node(ElementwiseOp("subtract"), ["w", "c"], Tensor("wq", (2 * _N, _K), dtype=F16), node_id="wq")
    graph.add_node(SliceOp(shape=(_N, _K), dim=0, start=0), ["wq"], Tensor("wg", (_N, _K), dtype=F16), node_id="wg")
    graph.add_node(SliceOp(shape=(_N, _K), dim=0, start=_N), ["wq"], Tensor("wu", (_N, _K), dtype=F16), node_id="wu")
    graph.add_node(LinearOp(), ["x", "wg"], Tensor("yg", (_M, _N), dtype=F16), node_id="yg")
    graph.add_node(LinearOp(), ["x", "wu"], Tensor("yu", (_M, _N), dtype=F16), node_id="yu")
    graph.add_node(ElementwiseOp("multiply"), ["yg", "yu"], Tensor("y", (_M, _N), dtype=F16), node_id="y")
    graph.inputs, graph.outputs = ["x", "w", "c"], ["y"]
    return graph


def _serial_tile(root, free: Axis, reduce_axis: Axis) -> TileOp:
    """Stamp the all-OFF schedule used by the serial projection regressions below."""
    tile = TileOp(op=root, place=Placement(free=(free,)), axes=(free, reduce_axis))
    nodes = {
        site: ProjectionSchedule(Tile()) if tile.views[site].axis is None else ReductionSchedule(Tile(), Reduce())
        for site in tile.node_sites
    }
    work = derive_inventory((Tile(),), coop=1) or Work()
    return replace(
        tile,
        schedule=Schedule(
            KernelSchedule(work, Raster.parse("")),
            nodes,
            {edge: EdgeSchedule(Stage.direct()) for edge in tile.edge_sites},
        ),
        materialization=ClassicMaterialization({}, {}),
    )


@pytest.fixture
def _scalar_tier(monkeypatch):
    """Pin the mma family off — the channels then fold serially in ONE loop body, the scope that clashes."""
    monkeypatch.setenv("EMMY_TILE", "")
    monkeypatch.setenv("EMMY_STAGE", "")


def test_sibling_cones_share_one_declaration_of_a_broadcast_constant(_scalar_tier) -> None:
    set_target(_CAP)
    try:
        lowered = Pipeline.build(CUDA_PASSES).run(_gate_up_graph(), ctx=Context(compute_capability=_CAP))
    finally:
        set_target(None)
    sources = {node.op.kernel_name: node.op.kernel_source for node in lowered.nodes.values() if getattr(node.op, "kernel_source", None)}

    assert sources, "the graph must lower to at least one kernel"
    assert {name: _redeclared(src) for name, src in sources.items() if _redeclared(src)} == {}
    # Every reduction step reads the shared constant through ONE binding used by both sibling
    # cones: a kernel that reads it binds it once, and every use pairs up — an ILP partition
    # multiplies the uses (one pair per register copy), never the binding.
    bindings = {source: re.findall(r"__half (\w+) = c\[0\];", source) for source in sources.values()}
    assert any(bindings.values())
    assert all(len(names) == 1 for names in bindings.values() if names)
    uses = [source.count(f" - {names[0]};") for source, names in bindings.items() if names]
    assert all(count >= 2 and count % 2 == 0 for count in uses), uses


def test_a_name_rebound_to_a_different_address_survives_as_the_fault_it_is() -> None:
    """Only an exact repeat is dead. A same-name rebind of a DIFFERENT address must reach nvcc."""
    zero = (Literal(0, "int"),)
    body = Body((Load(name="in0", input="a", index=zero), Load(name="in0", input="b", index=zero)))

    assert [stmt.input for stmt in materialize._drop_repeated_declarations(body)] == ["a", "b"]


def test_sibling_cones_share_one_declaration_of_a_derived_value() -> None:
    """A cone DERIVES from what it reads, so the repeat is an ``Assign`` as readily as a ``Load``.

    DeepSeek-V4's post block is a chain of normalizations; a placement cut lands two cones of one
    ``1 / (sum + eps)`` at the kernel's own scope, and the second reciprocal redeclares the first's
    name. nvcc rejects it exactly as it rejects the repeated load."""
    zero = (Literal(0, "int"),)
    body = Body(
        (
            Load(name="in0", input="s", index=zero),
            Assign(name="v0", op="reciprocal", args=("in0",)),
            Assign(name="v0", op="reciprocal", args=("in0",)),
        )
    )

    assert [stmt.name for stmt in materialize._drop_repeated_declarations(body)] == ["in0", "v0"]


def test_a_name_rebound_to_a_different_expression_survives_as_the_fault_it_is() -> None:
    """The same rule from the other side: two VALUES under one name stay two statements."""
    body = Body((Assign(name="v0", op="reciprocal", args=("a",)), Assign(name="v0", op="reciprocal", args=("b",))))

    assert [stmt.args for stmt in materialize._drop_repeated_declarations(body)] == [("a",), ("b",)]


def test_two_cones_that_bind_one_name_to_two_values_are_re_spelled() -> None:
    """The other half of the same rule: two VALUES cannot share one declaration either.

    Sibling cones lower under one ``__aside`` tag, which keeps them off the root's names but not off
    each other's. That is invisible while the cones agree — they are one statement and the dedup
    collapses them. A placement cut breaks the tie: it leaves one cone reading the workspace it
    materialized and its structural twin computing in place, so the two derive one name from
    different operands and nvcc rejects the second declaration.
    """
    body = [
        Assign(name="v0", op="add", args=("a", "b")),
        Assign(name="out0", op="reciprocal", args=("v0",)),
        Assign(name="v0", op="add", args=("ws", "b")),
        Assign(name="out1", op="reciprocal", args=("v0",)),
    ]

    spelled = factor._one_value_per_name(body)
    names = [stmt.name for stmt in spelled]

    assert names[0] == "v0" and names[2] != "v0", names
    assert spelled[1].args == ("v0",)  # a use BEFORE the rebinding still means the first value
    assert spelled[3].args == (names[2],)  # one after it means the second


def test_two_cones_that_agree_keep_one_spelling() -> None:
    """Nothing moves while the cones agree, or the dedup above would stop collapsing them."""
    body = [Assign(name="v0", op="add", args=("a", "b"))] * 2

    assert [stmt.name for stmt in factor._one_value_per_name(body)] == ["v0", "v0"]


def test_serial_nested_projections_re_spell_distinct_values_in_one_scope() -> None:
    """A cut workspace and its fused twin may derive different values under one original SSA name."""
    free, reduce_axis = Axis("m", 4), Axis("k", 8)
    red = reduction(
        reduce_axis,
        (slab("x_e", "x", "m", "k"),),
        (Assign(name="acc__v", op="copy", args=("x_e",)),),
        ("acc",),
    )
    inner = projection(
        (red, slab("ws", "cutbuf", "m"), slab("eps", "eps")),
        (
            Assign(name="v", op="add", args=("ws", "eps")),
            Assign(name="inner_out", op="multiply", args=("acc", "v")),
        ),
        results=("inner_out",),
    )
    root = projection(
        (inner, slab("base", "base", "m")),
        (
            Assign(name="v", op="add", args=("inner_out", "base")),
            Assign(name="out", op="reciprocal", args=("v",)),
        ),
        results=("out",),
    )
    bound = factor.factorize(_serial_tile(root, free, reduce_axis), root=None)
    additions = [stmt for stmt in bound.body if isinstance(stmt, Assign) and stmt.op.name == "add"]
    reciprocal = next(stmt for stmt in bound.body if isinstance(stmt, Assign) and stmt.name == "out")

    assert len(additions) == 2
    assert additions[0].name != additions[1].name
    assert reciprocal.args == (additions[1].name,)


def test_serial_root_and_projection_tail_re_spell_distinct_values_in_one_scope() -> None:
    """A serial fold's hoisted provider and its projection tail cannot redeclare one SSA name."""
    free, reduce_axis = Axis("m", 4), Axis("k", 8)
    provider = projection(
        (slab("base", "base", "m"), slab("eps", "eps")),
        (Assign(name="v", op="add", args=("base", "eps")),),
        results=("v",),
    )
    red = reduction(
        reduce_axis,
        (slab("x_e", "x", "m", "k"), provider),
        (Assign(name="acc__v", op="multiply", args=("x_e", "v")),),
        ("acc",),
    )
    root = projection(
        (red, slab("ws", "cutbuf", "m")),
        (
            Assign(name="v", op="add", args=("acc", "ws")),
            Assign(name="out", op="reciprocal", args=("v",)),
        ),
        results=("out",),
    )

    bound = factor.factorize(_serial_tile(root, free, reduce_axis), root=None)
    additions = [stmt for stmt in bound.body if isinstance(stmt, Assign) and stmt.op.name == "add"]
    reciprocal = next(stmt for stmt in bound.body if isinstance(stmt, Assign) and stmt.name == "out")

    assert len(additions) == 2
    assert additions[0].name != additions[1].name
    assert reciprocal.args == (additions[1].name,)


@pytest.mark.parametrize("coop", [1, 32, 64])
def test_projection_lowers_overlapping_carriers_together(coop) -> None:
    free, reduce_axis = Axis("m", 4), Axis("k", 8)
    shared = reduction(
        reduce_axis,
        (slab("x_e", "x", "m", "k"),),
        tuple(Assign(name=f"{name}__v", op="multiply", args=("x_e", "x_e")) for name in ("a", "b", "c")),
        ("a", "b", "c"),
    )
    left = projection((shared,), (Assign(name="left", op="add", args=("a", "b")),))
    root = projection((shared, left), (Assign(name="out", op="multiply", args=("c", "left")),))

    tile = _serial_tile(root, free, reduce_axis)
    tile = replace(
        tile,
        schedule=replace(
            tile.schedule,
            kernel=KernelSchedule(derive_inventory((Tile(),), coop=coop) or Work(), Raster.parse("")),
            nodes={
                site: replace(choice, reduce=Reduce.of(coop=coop)) if isinstance(choice, ReductionSchedule) else choice
                for site, choice in tile.schedule.nodes.items()
            },
        ),
    )
    bound = factor.factorize(tile, root=None)
    accumulators = [stmt.name for stmt in bound.body.iter() if isinstance(stmt, Accum)]
    assert len(accumulators) == len(set(accumulators))
