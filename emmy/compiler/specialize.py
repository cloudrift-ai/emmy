"""Bind symbolic dimensions in persisted compiler programs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.expression import Expression, Interval, Literal, SimplifyCtx, Var
from emmy.compiler.ir.frontend.ir import ReshapeOp, SliceOp
from emmy.compiler.ir.stmt import Body
from emmy.compiler.wire import rewrite


def _rewrite_graph(graph: Graph, fn: Callable[[object], object | None]) -> Graph:
    """A copy of ``graph`` with ``fn`` applied (:func:`~emmy.compiler.wire.rewrite`) to every op and output buffer."""
    out = graph.copy()
    for node in out.nodes.values():
        node.op = rewrite(node.op, fn)
        node.outputs = tuple(rewrite(tensor, fn) for tensor in node.outputs)
    return out


def _rewrite_body(body: Body, fn: Callable[[object], object | None]) -> Body:
    """``body`` with ``fn`` applied to every expression and inside every loop axis: what the rewrite reaches in a
    body, which is no wire object's fields but the statement module's own."""
    mapped = body.map(lambda stmt: stmt.map_expressions(lambda expression: rewrite(expression, fn)))
    return Body(stmt.rewrite(lambda name: name, axis_fn=lambda axis: rewrite(axis, fn)) for stmt in mapped)


def _bound_expression(expression: Expression, bindings: Mapping[str, int], *, extent: bool) -> Expression:
    """Bind the named dimensions inside one expression and simplify what that fixes.

    ``extent`` says the expression IS a dimension, so every name still free in it is a
    tensor extent and simplification may use the one fact an extent carries: it is at
    least 1. Every other expression indexes a tensor rather than sizing one, and its free
    names are output coordinates and loop variables that start at 0 — reading those as
    extents folds a real predicate away (an IndexMap's ``out_coord_1 < 1`` becomes false,
    silently dropping that source), so they simplify with no range at all.
    """
    specialized = expression.substitute({name: Literal(size, "int") for name, size in bindings.items()})
    ranges = {name: Interval(1, 1 << 30) for name in specialized.free_vars()} if extent else {}
    return specialized.simplify(SimplifyCtx(ranges))


def _bound_dim(dim: Dim, bindings: Mapping[str, int]) -> Dim:
    if dim.is_static:
        return dim
    if isinstance(dim.expression, Var):
        return Dim(bindings[dim.expression.name]) if dim.expression.name in bindings else dim
    expression = _bound_expression(dim.expression, bindings, extent=True)
    return Dim(int(expression.value)) if isinstance(expression, Literal) and expression.dtype == "int" else Dim(expression, hint=dim.hint)


def specialize_program(graph: Graph, bindings: Mapping[str, int]) -> Graph:
    """Return a copy of ``graph`` with the named symbolic dimensions bound: every dim, every expression — an index,
    a predicate, a context value — and the names a reshape or slice spells its shape with."""
    if not bindings:
        return graph.copy()
    invalid = {
        name: value for name, value in bindings.items() if not isinstance(name, str) or not name or type(value) is not int or value <= 0
    }
    if invalid:
        raise ValueError(f"dimension bindings must map non-empty names to positive integers: {invalid!r}")

    def bind(value):
        if isinstance(value, Graph):
            return _rewrite_graph(value, bind)
        if isinstance(value, Body):
            return _rewrite_body(value, bind)
        if isinstance(value, Dim):
            return _bound_dim(value, bindings)
        if isinstance(value, Expression):
            return _bound_expression(value, bindings, extent=False)
        if isinstance(value, (ReshapeOp, SliceOp)):
            shape = tuple(bindings.get(dim, dim) if isinstance(dim, str) else rewrite(dim, bind) for dim in value.shape)
            return replace(value, shape=shape)
        return None

    return bind(graph)


def rehint_program(graph: Graph, sizes: Mapping[str, int]) -> Graph:
    """``graph`` with its symbolic dims' hints set to ``sizes`` — the sizes a measurement bound them to — so the
    program stays symbolic and a bench of it binds those sizes (``wire.symbolic_bindings``). Binding them
    instead (:func:`specialize_program`) makes the dims static, another kernel. A dim spelled as an expression
    takes the expression's value at those sizes, and one over a name ``sizes`` lacks beside one it names is an
    error; a dim over no name ``sizes`` names keeps its hint."""

    def rehint(value):
        if isinstance(value, Graph):
            return _rewrite_graph(value, rehint)
        if isinstance(value, Body):
            return _rewrite_body(value, rehint)
        if not isinstance(value, Dim) or value.is_static or not value.expression.free_vars() & sizes.keys():
            return None
        if isinstance(value.expression, Var):
            return Dim(value.expression, hint=sizes[value.expression.name])
        if missing := sorted(value.expression.free_vars() - set(sizes)):
            raise ValueError(f"no size for {', '.join(missing)} in the dim {value.expression}")
        return Dim(value.expression, hint=int(value.expression.eval(dict(sizes))))

    return rehint(graph)


__all__ = ["rehint_program", "specialize_program"]
