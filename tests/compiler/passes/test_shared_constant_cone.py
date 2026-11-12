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
from emmy.compiler.ir.expression import Var
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
from emmy.compiler.ir.stmt import Accum, Assign, normalize_body, parse_body
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


def _names(stmt) -> tuple[str, ...]:
    """The names an ``Assign``'s value reads as direct operands (a bare-name value is its one operand)."""
    operands = (stmt.value,) if isinstance(stmt.value, Var) else stmt.value.operands
    return tuple(operand.name for operand in operands)


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
    # A use is any read of the binding past its declaration, whatever spells the subtraction (``-`` or ``__hsub_rn``).
    uses = [len(re.findall(rf"\b{names[0]}\b", source)) - 1 for source, names in bindings.items() if names]
    assert all(count >= 2 and count % 2 == 0 for count in uses), uses


def _normal_form(text: str) -> list[str]:
    """The statements of ``text``'s normal form, the form every kernel body takes at construction."""
    return [str(stmt) for stmt in normalize_body(parse_body(text))]


def test_sibling_cones_share_one_declaration_of_a_derived_value() -> None:
    """A cone DERIVES from what it reads, so the repeat is an ``Assign`` as readily as a ``Load``.

    DeepSeek-V4's post block is a chain of normalizations; a placement cut lands two cones of one
    ``1 / (sum + eps)`` at the kernel's own scope, and the second reciprocal rebinds the first's
    name to the same value. The normal form holds the value once."""
    body = _normal_form(
        """
        in0 = load s[0]
        v0 = reciprocal(in0)
        v0 = reciprocal(in0)
        out[0] = v0
        """
    )

    assert len(body) == 3, body


@pytest.mark.parametrize("value", [3840.0, 1e-6])
def test_sibling_cones_share_one_typed_literal_declaration(value) -> None:
    body = _normal_form(
        f"""
        x = load s[0]
        let f32 v0 = {value!r}
        first = multiply(x, v0)
        let f32 v0 = {value!r}
        second = multiply(x, v0)
        out[0] = first
        out[1] = second
        """
    )

    assert sum("*" in stmt for stmt in body) == 1, body


def test_a_same_dtype_copy_of_the_bound_value_is_a_repeat() -> None:
    """A placement cut on softmax's maximum binds ``v0`` to the workspace read in one cone and to a
    same-dtype copy of a second read of it in the sibling: one value spelled twice."""
    body = _normal_form(
        """
        f16 v0 = load ws[0]
        f16 acc0 = load ws[0]
        f16 v0 = acc0
        out[0] = v0
        """
    )

    assert sum("load ws" in stmt for stmt in body) == 1, body


def test_a_name_rebound_to_a_different_value_keeps_both_values() -> None:
    """Only an exact repeat is one value. A name bound twice to two values reads lexically: each read takes the
    binding ahead of it, and the normal form gives the two values two names."""
    body = _normal_form(
        """
        in0 = load a[0]
        out[0] = in0
        in0 = load b[0]
        out[1] = in0
        """
    )

    assert sum("load a[0]" in stmt for stmt in body) == 1 and sum("load b[0]" in stmt for stmt in body) == 1, body
    names = {stmt.split(" = ")[0].split()[-1] for stmt in body if "load" in stmt}
    assert len(names) == 2, body


def test_two_cones_that_bind_one_name_to_two_values_are_re_spelled() -> None:
    """The other half of the same rule: two VALUES cannot share one declaration either.

    Sibling cones lower under one ``__aside`` tag, which keeps them off the root's names but not off
    each other's. That is invisible while the cones agree — they are one statement and the dedup
    collapses them. A placement cut breaks the tie: it leaves one cone reading the workspace it
    materialized and its structural twin computing in place, so the two derive one name from
    different operands and nvcc rejects the second declaration.
    """
    body = list(
        parse_body(
            """
            v0 = add(a, b)
            out0 = reciprocal(v0)
            v0 = add(ws, b)
            out1 = reciprocal(v0)
            """
        )
    )

    spelled = factor._one_value_per_name(body)
    names = [stmt.name for stmt in spelled]

    assert names[0] == "v0" and names[2] != "v0", names
    assert _names(spelled[1]) == ("v0",)  # a use BEFORE the rebinding still means the first value
    assert _names(spelled[3]) == (names[2],)  # one after it means the second


def test_two_cones_that_agree_keep_one_spelling() -> None:
    """Nothing moves while the cones agree, or the dedup above would stop collapsing them."""
    body = list(parse_body("v0 = add(a, b)")) * 2

    assert [stmt.name for stmt in factor._one_value_per_name(body)] == ["v0", "v0"]


def test_serial_nested_projections_re_spell_distinct_values_in_one_scope() -> None:
    """A cut workspace and its fused twin may derive different values under one original SSA name."""
    free, reduce_axis = Axis("m", 4), Axis("k", 8)
    red = reduction(reduce_axis, (slab("x_e", "x", "m", "k"),), parse_body("acc__v = copy(x_e)"), ("acc",))
    inner = projection(
        (red, slab("ws", "cutbuf", "m"), slab("eps", "eps")),
        parse_body(
            """
            v = add(ws, eps)
            inner_out = multiply(acc, v)
            """
        ),
        results=("inner_out",),
    )
    root = projection(
        (inner, slab("base", "base", "m")),
        parse_body(
            """
            v = add(inner_out, base)
            out = reciprocal(v)
            """
        ),
        results=("out",),
    )
    bound = factor.factorize(_serial_tile(root, free, reduce_axis), root=None)
    additions = [stmt for stmt in bound.body if isinstance(stmt, Assign) and stmt.value.operation.name == "add"]
    reciprocal = next(stmt for stmt in bound.body if isinstance(stmt, Assign) and stmt.name == "out")

    assert len(additions) == 2
    assert additions[0].name != additions[1].name
    assert _names(reciprocal) == (additions[1].name,)


def test_serial_root_and_projection_tail_re_spell_distinct_values_in_one_scope() -> None:
    """A serial fold's hoisted provider and its projection tail cannot redeclare one SSA name."""
    free, reduce_axis = Axis("m", 4), Axis("k", 8)
    provider = projection((slab("base", "base", "m"), slab("eps", "eps")), parse_body("v = add(base, eps)"), results=("v",))
    red = reduction(reduce_axis, (slab("x_e", "x", "m", "k"), provider), parse_body("acc__v = multiply(x_e, v)"), ("acc",))
    root = projection(
        (red, slab("ws", "cutbuf", "m")),
        parse_body(
            """
            v = add(acc, ws)
            out = reciprocal(v)
            """
        ),
        results=("out",),
    )

    bound = factor.factorize(_serial_tile(root, free, reduce_axis), root=None)
    additions = [stmt for stmt in bound.body if isinstance(stmt, Assign) and stmt.value.operation.name == "add"]
    reciprocal = next(stmt for stmt in bound.body if isinstance(stmt, Assign) and stmt.name == "out")

    assert len(additions) == 2
    assert additions[0].name != additions[1].name
    assert _names(reciprocal) == (additions[1].name,)


@pytest.mark.parametrize("coop", [1, 32, 64])
def test_projection_lowers_overlapping_carriers_together(coop) -> None:
    free, reduce_axis = Axis("m", 4), Axis("k", 8)
    shared = reduction(
        reduce_axis,
        (slab("x_e", "x", "m", "k"),),
        parse_body("\n".join(f"{name}__v = multiply(x_e, x_e)" for name in ("a", "b", "c"))),
        ("a", "b", "c"),
    )
    left = projection((shared,), parse_body("left = add(a, b)"))
    root = projection((shared, left), parse_body("out = multiply(c, left)"))

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
