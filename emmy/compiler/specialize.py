"""Bind symbolic dimensions in persisted compiler programs."""

from __future__ import annotations

from collections.abc import Mapping

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import Interval, Literal, SimplifyCtx
from emmy.compiler.wire import decode, encode

_EXPR_TAGS = {"var", "literal", "binary", "builtin", "call", "ternary", "cast"}
_NAMED_SHAPE_OPS = {"torch.reshape", "torch.slice"}


def _specialize_expr(value: Mapping, bindings: Mapping[str, int], *, extent: bool = False) -> dict:
    """Bind the named dimensions inside one wire expression and simplify what that fixes.

    ``extent`` says the expression IS a dimension, so every name still free in it is a
    tensor extent and simplification may use the one fact an extent carries: it is at
    least 1. Every other expression indexes a tensor rather than sizing one, and its free
    names are output coordinates and loop variables that start at 0 — reading those as
    extents folds a real predicate away (an IndexMap's ``out_coord_1 < 1`` becomes false,
    silently dropping that source), so they simplify with no range at all.
    """
    return encode(_bound_expr(value, bindings, extent=extent))


def _bound_expr(value: Mapping, bindings: Mapping[str, int], *, extent: bool):
    expr = decode(dict(value))
    replacements = {name: Literal(size, "int") for name, size in bindings.items()}
    specialized = expr.substitute(replacements)
    ranges = {name: Interval(1, 1 << 30) for name in specialized.free_vars()} if extent else {}
    return specialized.simplify(SimplifyCtx(ranges))


def _specialize_dim(value, bindings: Mapping[str, int]):
    """A dim on the wire — ``int``, ``{sym, hint}`` or ``{expr, hint}`` — with its names bound."""
    if not isinstance(value, Mapping):
        return value
    if "sym" in value:
        return bindings.get(value["sym"], dict(value))
    expr = _bound_expr(value["expr"], bindings, extent=True)
    if isinstance(expr, Literal) and expr.dtype == "int":
        return int(expr.value)
    return {"expr": encode(expr), **({"hint": value["hint"]} if "hint" in value else {})}


def _specialize_named_shape(value, bindings: Mapping[str, int]):
    if isinstance(value, str):
        return bindings.get(value, value)
    if isinstance(value, list):
        return [_specialize_named_shape(item, bindings) for item in value]
    return value


def _specialize_wire(value, bindings: Mapping[str, int]):
    if isinstance(value, list):
        return [_specialize_wire(item, bindings) for item in value]
    if not isinstance(value, Mapping):
        return value
    keys = set(value)
    if len(value) == 1 and keys <= _EXPR_TAGS:
        return _specialize_expr(value, bindings)
    if ("sym" in keys and keys <= {"sym", "hint"}) or ("expr" in keys and keys <= {"expr", "hint"}):
        return _specialize_dim(value, bindings)
    if keys == {"dim"}:
        return {"dim": _specialize_dim(value["dim"], bindings)}
    specialized = {key: _specialize_wire(item, bindings) for key, item in value.items()}
    attrs = specialized.get("attrs")
    if specialized.get("op") in _NAMED_SHAPE_OPS and isinstance(attrs, Mapping) and "shape" in attrs:
        specialized["attrs"] = {**attrs, "shape": _specialize_named_shape(attrs["shape"], bindings)}
    return specialized


def specialize_program(graph: Graph, bindings: Mapping[str, int]) -> Graph:
    """Return a copy of ``graph`` with the named symbolic dimensions bound."""
    if not bindings:
        return graph.copy()
    invalid = {
        name: value for name, value in bindings.items() if not isinstance(name, str) or not name or type(value) is not int or value <= 0
    }
    if invalid:
        raise ValueError(f"dimension bindings must map non-empty names to positive integers: {invalid!r}")
    return Graph.from_wire(_specialize_wire(graph.to_wire(), bindings))


def rehint_program(wire: dict, sizes: Mapping[str, int]) -> dict:
    """``wire`` with its symbolic dims' hints set to ``sizes`` — the sizes a measurement bound them to — so the
    program stays symbolic and a bench of it binds those sizes (``wire.symbolic_bindings``). Binding them
    instead (:func:`specialize_program`) makes the dims static, another kernel. A dim spelled as an expression
    takes the expression's value at those sizes, and one over a name ``sizes`` lacks is an error; a plain symbolic
    dim ``sizes`` does not name keeps its hint."""

    def walk(value):
        if isinstance(value, list):
            return [walk(item) for item in value]
        if not isinstance(value, Mapping):
            return value
        keys = set(value)
        # A hinted dim is the one two-key mapping a wire holds: a body's tagged values have one key each.
        if keys == {"sym", "hint"}:
            return {**value, "hint": sizes.get(value["sym"], value["hint"])}
        if keys == {"expr", "hint"}:
            expr = decode(dict(value["expr"]))
            missing = sorted(set(expr.free_vars()) - set(sizes))
            if missing:
                raise ValueError(f"no size for {', '.join(missing)} in the dim {expr.pretty()}")
            return {"expr": value["expr"], "hint": int(expr.eval(dict(sizes)))}
        return {key: walk(item) for key, item in value.items()}

    return walk(wire)


__all__ = ["rehint_program", "specialize_program"]
